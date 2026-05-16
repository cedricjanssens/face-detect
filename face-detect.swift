// face-detect.swift — macOS CLI for face detection + recognition embeddings.
// Build: swiftc -O -framework AVFoundation -framework CoreML face-detect.swift -o face-detect
// Requires: macOS 14+, Apple Silicon recommended (Neural Engine).
//
// Embeddings: AdaFace (Core ML, 512-dim L2-normalized) for face recognition.
//   Models: IR-18 (fast, default) or IR-50 (ResNet-50, more discriminant).
// Fallback: Vision VNGenerateImageFeaturePrintRequest (768-dim, generic image similarity).
//
// Model search paths (in order):
//   1. $FACE_DETECT_MODEL_PATH
//   2. <binary_dir>/AdaFace_<variant>.mlpackage
//   3. /opt/homebrew/share/face-detect/AdaFace_<variant>.mlpackage
//   4. /usr/local/share/face-detect/AdaFace_<variant>.mlpackage

import AppKit
import AVFoundation
import CoreML
import Foundation
import Vision

let VERSION = "0.6.0"

// MARK: - Utilities

func die(_ msg: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(code)
}

func shell(_ cmd: String) -> Int32 {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/bin/sh")
    p.arguments = ["-c", cmd]
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    do { try p.run(); p.waitUntilExit(); return p.terminationStatus }
    catch { return -1 }
}

func emit<T: Encodable>(_ value: T, pretty: Bool = false) {
    let enc = JSONEncoder()
    enc.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted] : [.sortedKeys]
    guard let data = try? enc.encode(value) else { die("encode failed") }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
}

// MARK: - Output models

struct ImageResult: Encodable {
    let image: String
    let width: Int
    let height: Int
    let elapsed_ms: Int
    let engine: String
    let engine_dim: Int
    let model: String?
    let description: String?
    let tags: [TagResult]?
    let faces: [FaceResult]?
    let error: String?
}

struct TagResult: Encodable {
    let label: String
    let confidence: Float
}

// Output format for the embedding vector. `.float` (default, back-compat) emits a
// JSON array of numbers rounded to 6 significant figures (≈30% smaller than the
// IEEE round-trip representation, error <1e-6 → negligible for cosine similarity).
// `.b64` emits a single base64 string of little-endian Float32 bytes (≈70% smaller
// vs default), under field `embedding_b64`. Mitigates macOS FIFO ~16 KB saturation
// on multi-face responses. v0.5.5+.
enum EmbeddingFormat: String { case float, b64 }
var embeddingFormat: EmbeddingFormat = .float

func roundSig(_ x: Float, sig: Int) -> Float {
    guard x.isFinite, x != 0 else { return x }
    let d = Double(abs(x))
    let mag = pow(10.0, Double(sig - 1) - floor(log10(d)))
    return Float((Double(x) * mag).rounded() / mag)
}

struct FaceResult: Encodable {
    let bbox: [Double]          // [x, y, w, h] normalized 0-1, origin bottom-left
    let confidence: Float
    let quality: Float?         // faceCaptureQuality 0-1
    let roll: Double?
    let yaw: Double?
    let pitch: Double?
    let embedding: [Float]      // 512 floats (AdaFace) or 768 (Vision)
    let landmarks: [String: [[Double]]]?

    enum CodingKeys: String, CodingKey {
        case bbox, confidence, quality, roll, yaw, pitch
        case embedding, embedding_b64
        case landmarks
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(bbox, forKey: .bbox)
        try c.encode(confidence, forKey: .confidence)
        try c.encodeIfPresent(quality, forKey: .quality)
        try c.encodeIfPresent(roll, forKey: .roll)
        try c.encodeIfPresent(yaw, forKey: .yaw)
        try c.encodeIfPresent(pitch, forKey: .pitch)
        switch embeddingFormat {
        case .float:
            let compact = embedding.map { roundSig($0, sig: 6) }
            try c.encode(compact, forKey: .embedding)
        case .b64:
            let bytes = embedding.withUnsafeBufferPointer { Data(buffer: $0) }
            try c.encode(bytes.base64EncodedString(), forKey: .embedding_b64)
        }
        try c.encodeIfPresent(landmarks, forKey: .landmarks)
    }
}

struct BenchResult: Encodable {
    let images: Int
    let faces: Int
    let total_ms: Int
    let avg_ms: Double
    let fps: Double
    let embedding_dim: Int
}

// MARK: - Watch protocol

struct WatchRequest: Decodable {
    let id: String?
    let image: String?
    let ping: Bool?
    let shutdown: Bool?
}

struct PongResponse: Encodable {
    let pong: Bool
    let id: String?
    let uptime_ms: Int
    let processed: Int
    let engine: String
    let engine_dim: Int
    let model: String?
}

struct ShutdownResponse: Encodable {
    let shutdown: Bool
    let id: String?
    let uptime_ms: Int
    let processed: Int
}

// Wrapper to add id to ImageResult without changing the base struct.
// We encode manually so the field order is deterministic and id
// is omitted when absent (back-compat with single-path input format).
struct IdentifiedImageResult: Encodable {
    let id: String?
    let result: ImageResult

    enum CodingKeys: String, CodingKey {
        case id
    }

    func encode(to encoder: Encoder) throws {
        // Encode ImageResult fields at top level, plus id when present.
        try result.encode(to: encoder)
        if let id = id {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
        }
    }
}

// MARK: - Image loading

func loadCGImage(path: String) -> (CGImage, Int, Int)? {
    let url = URL(fileURLWithPath: path)
    // Use CGImageSource for broader format support (HEIC, JPEG, PNG, TIFF)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }
    return (cg, cg.width, cg.height)
}

// MARK: - Landmark extraction

let landmarkRegions: [(String, (VNFaceLandmarks2D) -> VNFaceLandmarkRegion2D?)] = [
    ("faceContour",  { $0.faceContour }),
    ("leftEye",      { $0.leftEye }),
    ("rightEye",     { $0.rightEye }),
    ("leftEyebrow",  { $0.leftEyebrow }),
    ("rightEyebrow", { $0.rightEyebrow }),
    ("nose",         { $0.nose }),
    ("noseCrest",    { $0.noseCrest }),
    ("medianLine",   { $0.medianLine }),
    ("outerLips",    { $0.outerLips }),
    ("innerLips",    { $0.innerLips }),
    ("leftPupil",    { $0.leftPupil }),
    ("rightPupil",   { $0.rightPupil }),
]

func extractLandmarks(face: VNFaceObservation) -> [String: [[Double]]]? {
    guard let lm = face.landmarks else { return nil }
    let bb = face.boundingBox
    var result: [String: [[Double]]] = [:]
    for (name, accessor) in landmarkRegions {
        guard let region = accessor(lm) else { continue }
        let points = region.normalizedPoints
        var coords: [[Double]] = []
        for i in 0..<region.pointCount {
            let p = points[i]
            coords.append([
                Double(bb.minX + CGFloat(p.x) * bb.width),
                Double(bb.minY + CGFloat(p.y) * bb.height)
            ])
        }
        result[name] = coords
    }
    return result.isEmpty ? nil : result
}

// MARK: - Embedding engines & model variants

enum EmbeddingEngine: String {
    case adaface
    case vision
}

enum AdaFaceVariant: String {
    case ir18
    case ir50

    var modelFileName: String {
        switch self {
        case .ir18: return "AdaFace_IR18.mlpackage"
        case .ir50: return "AdaFace_IR50.mlpackage"
        }
    }
}

enum DescriptionLang: String {
    case fr, en
}

// Loaded once at startup, reused for all images.
var adaFaceModel: VNCoreMLModel?
var activeEngine: EmbeddingEngine = .adaface
var adaFaceVariant: AdaFaceVariant = .ir18
var descLang: DescriptionLang = .fr
var minQuality: Float = 0.0
var globalTimeoutSec: Int = 30
var idleTimeoutSec: UInt32 = 1800  // 30 min default for watch mode
var predictTimeoutSec: UInt32 = 60 // per-image watchdog (watch mode); exits if a single predict blocks > N sec
var watchdogInPredict: Int32 = 0   // sig_atomic_t: 1 inside processImage(), 0 otherwise. Reads in signal handler must be lock-free.

func modelLabel() -> String? {
    activeEngine == .adaface ? adaFaceVariant.rawValue : nil
}

func locateModel() -> URL? {
    let fileName = adaFaceVariant.modelFileName
    let binaryDir = (CommandLine.arguments.first.map { ($0 as NSString).deletingLastPathComponent } ?? "")
    let candidates: [String] = [
        ProcessInfo.processInfo.environment["FACE_DETECT_MODEL_PATH"] ?? "",
        binaryDir.isEmpty ? "" : binaryDir + "/" + fileName,
        "/opt/homebrew/share/face-detect/" + fileName,
        "/usr/local/share/face-detect/" + fileName,
    ]
    for path in candidates where !path.isEmpty {
        if FileManager.default.fileExists(atPath: path) {
            return URL(fileURLWithPath: path)
        }
    }
    return nil
}

func loadAdaFaceModel() {
    guard let modelURL = locateModel() else {
        let v = adaFaceVariant.rawValue
        FileHandle.standardError.write(Data("face-detect: AdaFace \(v) model not found, falling back to Vision (768d)\n".utf8))
        FileHandle.standardError.write(Data("  install: make face-detect-models-\(v)  OR  helpers/face-detect/scripts/install-models.sh \(v)\n".utf8))
        activeEngine = .vision
        return
    }

    // Detect ANE contention: only MLX runners actually hold the Neural Engine.
    // "ollama serve" alone, or with Metal/GGUF runners (--ollama-engine), uses GPU
    // not ANE — no conflict with CoreML. Pre-0.5.4 we skipped ANE for any Ollama
    // process, which was overly conservative (penalized nomic-embed-text, GGUF LLMs).
    // Overrides:
    //   FACE_DETECT_NO_ANE=1    → always skip ANE (paranoia / known-good fallback)
    //   FACE_DETECT_FORCE_ANE=1 → always use ANE, bypass MLX detection
    let env = ProcessInfo.processInfo.environment
    let forceSkip = env["FACE_DETECT_NO_ANE"] == "1"
    let forceANE = env["FACE_DETECT_FORCE_ANE"] == "1"
    let mlxRunner = !forceANE && (shell("pgrep -f 'ollama.runner.*mlx'") == 0)

    let config = MLModelConfiguration()
    if forceSkip || mlxRunner {
        config.computeUnits = .cpuAndGPU
        let reason = forceSkip ? "FACE_DETECT_NO_ANE" : "Ollama MLX runner detected"
        FileHandle.standardError.write(Data("face-detect: ANE skipped (\(reason)), using CPU+GPU\n".utf8))
    } else {
        config.computeUnits = .all  // Neural Engine when available
    }

    // Load model with timeout — compileModel can deadlock on Neural Engine
    // when another process (Ollama/MLX) holds the ANE.
    let sem = DispatchSemaphore(value: 0)
    var loadedModel: VNCoreMLModel?
    var loadError: Error?

    DispatchQueue.global(qos: .userInitiated).async {
        do {
            let compiled = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiled, configuration: config)
            loadedModel = try VNCoreMLModel(for: model)
        } catch {
            loadError = error
        }
        sem.signal()
    }

    let timeoutSec = 15
    if sem.wait(timeout: .now() + .seconds(timeoutSec)) == .timedOut {
        FileHandle.standardError.write(Data("face-detect: model load timed out (\(timeoutSec)s), falling back to Vision\n".utf8))
        activeEngine = .vision
        return
    }

    if let model = loadedModel {
        adaFaceModel = model
    } else {
        FileHandle.standardError.write(Data("face-detect: model load failed (\(loadError?.localizedDescription ?? "unknown")), falling back to Vision\n".utf8))
        activeEngine = .vision
    }
}

func croppedFaceImage(cgImage: CGImage, bbox: CGRect) -> CGImage? {
    // Expand bbox by 20% for context (hair, chin)
    let pad = CGFloat(0.2)
    let expanded = CGRect(
        x: max(0, bbox.minX - bbox.width * pad),
        y: max(0, bbox.minY - bbox.height * pad),
        width: min(1.0, bbox.width * (1 + 2 * pad)),
        height: min(1.0, bbox.height * (1 + 2 * pad))
    ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))

    let pixelRect = VNImageRectForNormalizedRect(expanded, cgImage.width, cgImage.height)
    guard pixelRect.width > 0, pixelRect.height > 0 else { return nil }
    return cgImage.cropping(to: pixelRect)
}

// MARK: - Face alignment (ArcFace/AdaFace canonical template)

// ArcFace/AdaFace canonical eye positions in 112x112 (CG bottom-left origin)
private let kAlignLeftEye  = CGPoint(x: 38.2946, y: 112.0 - 51.6963)
private let kAlignRightEye = CGPoint(x: 73.5318, y: 112.0 - 51.5014)

/// Produce a 112x112 face image aligned via similarity transform on eye positions.
/// Falls back to simple bbox crop if landmarks are unavailable.
func alignedFaceImage(cgImage: CGImage, face: VNFaceObservation) -> CGImage? {
    guard let lm = face.landmarks,
          let lp = lm.leftPupil, lp.pointCount > 0,
          let rp = lm.rightPupil, rp.pointCount > 0 else {
        return croppedFaceImage(cgImage: cgImage, bbox: face.boundingBox)
    }

    let bb = face.boundingBox
    let imgW = CGFloat(cgImage.width)
    let imgH = CGFloat(cgImage.height)

    // Source eye positions in pixel coords (CG bottom-left origin)
    let sl = CGPoint(
        x: (bb.minX + CGFloat(lp.normalizedPoints[0].x) * bb.width) * imgW,
        y: (bb.minY + CGFloat(lp.normalizedPoints[0].y) * bb.height) * imgH)
    let sr = CGPoint(
        x: (bb.minX + CGFloat(rp.normalizedPoints[0].x) * bb.width) * imgW,
        y: (bb.minY + CGFloat(rp.normalizedPoints[0].y) * bb.height) * imgH)

    // Similarity transform: template = [[a,-b],[b,a]] * source + [tx,ty]
    // Solved from 2 point pairs (left eye, right eye)
    let dx = sr.x - sl.x, dy = sr.y - sl.y
    let denom = dx * dx + dy * dy
    guard denom > 1 else { return croppedFaceImage(cgImage: cgImage, bbox: bb) }

    let tdx = kAlignRightEye.x - kAlignLeftEye.x
    let tdy = kAlignRightEye.y - kAlignLeftEye.y
    let a = (tdx * dx + tdy * dy) / denom
    let b = (tdy * dx - tdx * dy) / denom
    let tx = kAlignLeftEye.x - a * sl.x + b * sl.y
    let ty = kAlignLeftEye.y - b * sl.x - a * sl.y

    // CGAffineTransform: x'=a*x+c*y+tx, y'=b*x+d*y+ty
    let xform = CGAffineTransform(a: a, b: b, c: -b, d: a, tx: tx, ty: ty)

    // Render into 112x112 context (CG bottom-left origin matches VN)
    let sz = 112
    guard let ctx = CGContext(data: nil, width: sz, height: sz,
                              bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
        return croppedFaceImage(cgImage: cgImage, bbox: bb)
    }
    ctx.interpolationQuality = .high
    ctx.concatenate(xform)
    ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: imgW, height: imgH))
    return ctx.makeImage()
}

// AdaFace: 512-dim L2-normalized embedding via Core ML (Neural Engine when possible)
func extractAdaFaceEmbedding(cgImage: CGImage, face: VNFaceObservation) -> [Float] {
    guard let model = adaFaceModel else { return [] }
    // Use aligned face when landmarks available, fall back to bbox crop
    guard let aligned = alignedFaceImage(cgImage: cgImage, face: face) else { return [] }

    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill
    let handler = VNImageRequestHandler(cgImage: aligned)
    do {
        try handler.perform([request])
    } catch {
        return []
    }
    guard let result = request.results?.first as? VNCoreMLFeatureValueObservation,
          let array = result.featureValue.multiArrayValue else {
        return []
    }
    let count = array.count
    var floats = [Float](repeating: 0, count: count)
    // Bulk copy when data is already Float32 (avoids NSNumber boxing per element)
    if array.dataType == .float32 {
        memcpy(&floats, array.dataPointer, count * MemoryLayout<Float>.size)
    } else {
        for i in 0..<count { floats[i] = Float(array[i].doubleValue) }
    }
    return floats
}

// Vision FeaturePrint: 768-dim generic image similarity (fallback)
func extractVisionEmbedding(cgImage: CGImage, bbox: CGRect) -> [Float] {
    guard let cropped = croppedFaceImage(cgImage: cgImage, bbox: bbox) else { return [] }
    let req = VNGenerateImageFeaturePrintRequest()
    let handler = VNImageRequestHandler(cgImage: cropped)
    do { try handler.perform([req]) } catch { return [] }
    guard let obs = req.results?.first as? VNFeaturePrintObservation else { return [] }
    let count = obs.elementCount
    var floats = [Float](repeating: 0, count: count)
    obs.data.withUnsafeBytes { rawBuf in
        guard let ptr = rawBuf.baseAddress else { return }
        memcpy(&floats, ptr, count * MemoryLayout<Float>.size)
    }
    return floats
}

func extractEmbedding(cgImage: CGImage, face: VNFaceObservation) -> [Float] {
    switch activeEngine {
    case .adaface: return extractAdaFaceEmbedding(cgImage: cgImage, face: face)
    case .vision:  return extractVisionEmbedding(cgImage: cgImage, bbox: face.boundingBox)
    }
}

func engineDim() -> Int {
    switch activeEngine {
    case .adaface: return 512
    case .vision:  return 768
    }
}

// MARK: - Description generation (i18n)

struct DescStrings {
    let person: String
    let baby: String
    let child: String
    let groupFmt: String       // must contain %d
    let groupChildFmt: String  // must contain %d
    let outdoor: String
    let indoor: String
    let fallback: String
    let context: [(tag: String, word: String)]
}

let descTable: [DescriptionLang: DescStrings] = [
    .fr: DescStrings(
        person: "personne", baby: "bébé", child: "enfant",
        groupFmt: "groupe de %d personnes",
        groupChildFmt: "groupe de %d personnes avec enfant(s)",
        outdoor: "en extérieur", indoor: "en intérieur", fallback: "image",
        context: [
            ("food", "repas"), ("drink", "boisson"), ("cake", "gâteau"),
            ("beach", "plage"), ("snow", "neige"), ("mountain", "montagne"),
            ("water", "eau"), ("pool", "piscine"), ("garden", "jardin"),
            ("grass", "herbe"), ("tree", "arbre"), ("flower", "fleur"),
            ("car", "voiture"), ("vehicle", "véhicule"),
            ("sport", "sport"), ("ball", "ballon"),
            ("animal", "animal"), ("dog", "chien"), ("cat", "chat"),
            ("celebration", "fête"), ("party", "fête"),
        ]
    ),
    .en: DescStrings(
        person: "person", baby: "baby", child: "child",
        groupFmt: "group of %d people",
        groupChildFmt: "group of %d people with child(ren)",
        outdoor: "outdoors", indoor: "indoors", fallback: "image",
        context: [
            ("food", "meal"), ("drink", "drink"), ("cake", "cake"),
            ("beach", "beach"), ("snow", "snow"), ("mountain", "mountain"),
            ("water", "water"), ("pool", "pool"), ("garden", "garden"),
            ("grass", "grass"), ("tree", "tree"), ("flower", "flower"),
            ("car", "car"), ("vehicle", "vehicle"),
            ("sport", "sport"), ("ball", "ball"),
            ("animal", "animal"), ("dog", "dog"), ("cat", "cat"),
            ("celebration", "celebration"), ("party", "party"),
        ]
    ),
]

func generateDescription(faceCount: Int, tags: [TagResult]) -> String {
    let s = descTable[descLang]!
    let tagSet = Set(tags.map { $0.label })

    // Subject
    var subject: String
    if faceCount == 0 {
        subject = ""
    } else if faceCount == 1 {
        if tagSet.contains("baby") { subject = s.baby }
        else if tagSet.contains("child") { subject = s.child }
        else { subject = s.person }
    } else {
        if tagSet.contains("baby") || tagSet.contains("child") {
            subject = String(format: s.groupChildFmt, faceCount)
        } else {
            subject = String(format: s.groupFmt, faceCount)
        }
    }

    // Setting
    var setting = ""
    if tagSet.contains("outdoor") || tagSet.contains("sky") || tagSet.contains("land") {
        setting = s.outdoor
    } else if tagSet.contains("structure") || tagSet.contains("furniture") || tagSet.contains("room") {
        setting = s.indoor
    }

    // Activity / context keywords (first match wins)
    var details: [String] = []
    for (tag, word) in s.context {
        if tagSet.contains(tag) { details.append(word); break }
    }

    // Assemble
    var parts: [String] = []
    if !subject.isEmpty { parts.append(subject) }
    if !setting.isEmpty { parts.append(setting) }
    parts.append(contentsOf: details)

    if parts.isEmpty {
        if let first = tags.first { return first.label }
        return s.fallback
    }

    return parts.joined(separator: ", ")
}

// MARK: - Core pipeline

func processImage(path: String, cgImage: CGImage) -> ImageResult {
    let start = DispatchTime.now()
    let w = cgImage.width
    let h = cgImage.height

    // Phase 1: face detection + landmarks + image classification (single pass)
    let landmarksReq = VNDetectFaceLandmarksRequest()
    let classifyReq = VNClassifyImageRequest()
    let handler = VNImageRequestHandler(cgImage: cgImage)

    do {
        try handler.perform([landmarksReq, classifyReq])
    } catch {
        let ms = elapsedMs(since: start)
        return ImageResult(image: path, width: w, height: h, elapsed_ms: ms,
                           engine: activeEngine.rawValue, engine_dim: engineDim(),
                           model: modelLabel(),
                           description: nil, tags: nil, faces: nil,
                           error: "vision failed: \(error.localizedDescription)")
    }

    // Extract tags (confidence > 0.3, sorted by confidence)
    let tags: [TagResult] = (classifyReq.results ?? [])
        .filter { $0.confidence > 0.3 }
        .sorted { $0.confidence > $1.confidence }
        .map { TagResult(label: $0.identifier, confidence: $0.confidence) }

    let observations = landmarksReq.results ?? []

    // Phase 2: face quality
    let qualityReq = VNDetectFaceCaptureQualityRequest()
    qualityReq.inputFaceObservations = observations
    var qualities: [Float?] = Array(repeating: nil, count: observations.count)
    if let _ = try? handler.perform([qualityReq]) {
        for (i, qobs) in (qualityReq.results ?? []).enumerated() {
            if i < qualities.count {
                qualities[i] = qobs.faceCaptureQuality
            }
        }
    }

    // Phase 3: per-face embedding + assembly (skip faces below min-quality)
    var faces: [FaceResult] = []
    for (i, obs) in observations.enumerated() {
        let q = i < qualities.count ? qualities[i] : nil
        if let q = q, minQuality > 0, q < minQuality { continue }

        let bb = obs.boundingBox
        let embedding = extractEmbedding(cgImage: cgImage, face: obs)
        let lm = extractLandmarks(face: obs)

        let face = FaceResult(
            bbox: [Double(bb.origin.x), Double(bb.origin.y),
                   Double(bb.width), Double(bb.height)],
            confidence: obs.confidence,
            quality: q,
            roll: obs.roll?.doubleValue,
            yaw: obs.yaw?.doubleValue,
            pitch: obs.pitch?.doubleValue,
            embedding: embedding,
            landmarks: lm
        )
        faces.append(face)
    }

    let ms = elapsedMs(since: start)
    let desc = generateDescription(faceCount: faces.count, tags: tags)
    return ImageResult(image: path, width: w, height: h, elapsed_ms: ms,
                       engine: activeEngine.rawValue, engine_dim: engineDim(),
                       model: modelLabel(),
                       description: desc, tags: tags, faces: faces, error: nil)
}

func elapsedMs(since start: DispatchTime) -> Int {
    Int((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
}

// MARK: - Mode: single image

func cmdSingle(_ path: String) {
    guard let (cg, _, _) = loadCGImage(path: path) else {
        die("cannot load image: \(path)")
    }
    let result = processImage(path: path, cgImage: cg)
    emit(result, pretty: true)
}

// MARK: - Mode: batch (stdin → NDJSON stdout)

func cmdBatch() {
    var count = 0
    while let line = readLine() {
        let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { continue }
        count += 1
        autoreleasepool {
            if let (cg, _, _) = loadCGImage(path: path) {
                let result = processImage(path: path, cgImage: cg)
                let faceCount = result.faces?.count ?? 0
                FileHandle.standardError.write(Data("[\(count)] \(path): \(faceCount) faces, \(result.elapsed_ms)ms\n".utf8))
                emit(result)
            } else {
                FileHandle.standardError.write(Data("[\(count)] \(path): FAILED (cannot load)\n".utf8))
                emit(ImageResult(image: path, width: 0, height: 0, elapsed_ms: 0,
                                 engine: activeEngine.rawValue, engine_dim: engineDim(),
                                 model: modelLabel(),
                                 description: nil, tags: nil, faces: nil, error: "cannot load image"))
            }
        }
    }
}

// MARK: - Mode: watch (FIFO or Unix socket daemon)

// Shared signal-handler installer for watch modes. Idle/predict alarm semantics
// are identical between transports.
func installWatchSignalHandlers() {
    signal(SIGPIPE, SIG_IGN)

    signal(SIGTERM) { _ in
        let msg: StaticString = "face-detect: SIGTERM, exiting\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
        _exit(0)
    }
    signal(SIGINT) { _ in
        let msg: StaticString = "face-detect: SIGINT, exiting\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
        _exit(0)
    }

    // POSIX alarm() serves two roles, distinguished by the watchdogInPredict flag:
    //   - watchdogInPredict == 0 → idle timeout (no input for N sec). Default 30 min.
    //   - watchdogInPredict == 1 → predict watchdog (one image stuck > N sec, likely
    //     ANE deadlock or CoreML zombification). Exit 124 to signal "respawn me" to
    //     the client. Default 60s, configurable via --predict-timeout.
    signal(SIGALRM) { _ in
        if watchdogInPredict != 0 {
            let msg: StaticString = "face-detect: predict watchdog fired (>predict-timeout, likely ANE/CoreML deadlock), exiting 124\n"
            _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
            _exit(124)
        } else {
            let msg: StaticString = "face-detect: idle timeout, exiting\n"
            _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
            _exit(0)
        }
    }
}

// Per-session loop: reads newline-delimited JSON commands from inHandle, writes
// responses to outHandle (which may be the same fd for socket streams). Returns
// when the peer disconnects (EOF). Updates `processed` and rearms alarms.
// `clientLabel` is used only for stderr logs.
func runWatchSession(
    inHandle: FileHandle,
    outHandle: FileHandle,
    decoder: JSONDecoder,
    startTime: DispatchTime,
    processed: inout Int,
    clientLabel: String
) {
    func uptimeMs() -> Int {
        Int((DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000)
    }
    func logStderr(_ msg: String) {
        FileHandle.standardError.write(Data("face-detect: \(msg)\n".utf8))
    }
    func emitJSON<T: Encodable>(_ value: T, to handle: FileHandle) {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        guard let data = try? enc.encode(value) else { return }
        handle.write(data)
        handle.write(Data([0x0a]))
    }

    var buffer = Data()
    while true {
        let chunk = inHandle.availableData
        if chunk.isEmpty { break } // EOF — peer disconnected
        buffer.append(chunk)

        if buffer.count > 65536 {
            logStderr("input buffer overflow (>64KiB without newline), dropping (\(clientLabel))")
            buffer.removeAll(keepingCapacity: true)
            continue
        }

        while let nlRange = buffer.range(of: Data([0x0a])) {
            let lineData = buffer[buffer.startIndex..<nlRange.lowerBound]
            buffer.removeSubrange(buffer.startIndex...nlRange.lowerBound)

            guard let line = String(data: lineData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  !line.isEmpty else { continue }

            alarm(idleTimeoutSec)

            autoreleasepool {
                var requestId: String? = nil
                var imagePath: String? = nil
                var isPing = false
                var isShutdown = false

                if line.hasPrefix("{") {
                    if let data = line.data(using: .utf8),
                       let req = try? decoder.decode(WatchRequest.self, from: data) {
                        requestId = req.id
                        imagePath = req.image
                        isPing = req.ping ?? false
                        isShutdown = req.shutdown ?? false
                    } else {
                        logStderr("malformed JSON (parse failed): \(line.prefix(120))")
                        return
                    }
                } else {
                    imagePath = line
                }

                if isShutdown {
                    logStderr("shutdown requested (processed=\(processed))")
                    emitJSON(ShutdownResponse(
                        shutdown: true,
                        id: requestId,
                        uptime_ms: uptimeMs(),
                        processed: processed
                    ), to: outHandle)
                    outHandle.synchronizeFile()
                    outHandle.closeFile()
                    _exit(0)
                }

                if isPing {
                    emitJSON(PongResponse(
                        pong: true,
                        id: requestId,
                        uptime_ms: uptimeMs(),
                        processed: processed,
                        engine: activeEngine.rawValue,
                        engine_dim: engineDim(),
                        model: modelLabel()
                    ), to: outHandle)
                    return
                }

                guard let path = imagePath else {
                    logStderr("malformed request (no image): \(line.prefix(120))")
                    return
                }

                let result: ImageResult
                let t0 = DispatchTime.now()
                if let (cg, _, _) = loadCGImage(path: path) {
                    logStderr("processing \(path)")
                    watchdogInPredict = 1
                    alarm(predictTimeoutSec)
                    result = processImage(path: path, cgImage: cg)
                    watchdogInPredict = 0
                    alarm(idleTimeoutSec)
                } else {
                    result = ImageResult(
                        image: path, width: 0, height: 0, elapsed_ms: 0,
                        engine: activeEngine.rawValue, engine_dim: engineDim(),
                        model: modelLabel(),
                        description: nil, tags: nil, faces: nil, error: "cannot load image"
                    )
                }
                processed += 1
                let took = Int((DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000)
                let faceCount = result.faces?.count ?? 0
                let errPart = result.error.map { " error=\($0)" } ?? ""
                logStderr("done \(path) in \(took)ms (\(faceCount) faces)\(errPart)")

                emitJSON(IdentifiedImageResult(id: requestId, result: result), to: outHandle)
            }
        }
    }
}

func cmdWatch(inPath: String, outPath: String) {
    installWatchSignalHandlers()
    alarm(idleTimeoutSec)

    let startTime = DispatchTime.now()
    var processed = 0
    let decoder = JSONDecoder()

    FileHandle.standardError.write(Data("face-detect: ready (engine=\(activeEngine.rawValue), model=\(adaFaceVariant.rawValue), dim=\(engineDim()), idle_timeout=\(idleTimeoutSec)s, predict_timeout=\(predictTimeoutSec)s, embedding_format=\(embeddingFormat.rawValue), transport=fifo, in=\(inPath), out=\(outPath))\n".utf8))

    while true {
        guard let inHandle = FileHandle(forReadingAtPath: inPath) else {
            die("cannot open input FIFO: \(inPath)")
        }
        guard let outHandle = FileHandle(forWritingAtPath: outPath) else {
            die("cannot open output FIFO: \(outPath)")
        }

        runWatchSession(
            inHandle: inHandle,
            outHandle: outHandle,
            decoder: decoder,
            startTime: startTime,
            processed: &processed,
            clientLabel: "fifo"
        )

        inHandle.closeFile()
        outHandle.closeFile()
        FileHandle.standardError.write(Data("face-detect: writer disconnected (processed=\(processed)), waiting for reconnect\n".utf8))
        alarm(idleTimeoutSec)
    }
}

// Unix domain socket transport. Buffer = 1 MB (SO_SNDBUF/SO_RCVBUF) — eliminates
// the ~16 KB macOS FIFO PIPE_SIZE stall on multi-face responses. One client at a
// time (listen backlog 1) to preserve single-threaded CoreML semantics.
func cmdWatchSocket(socketPath: String) {
    installWatchSignalHandlers()
    alarm(idleTimeoutSec)

    // Remove stale socket file from previous run (e.g. crash, no graceful unlink).
    // unlink() returning ENOENT is harmless.
    _ = unlink(socketPath)

    let serverFd = socket(AF_UNIX, SOCK_STREAM, 0)
    if serverFd < 0 { die("socket() failed: \(String(cString: strerror(errno)))") }

    // Set buffers BEFORE bind/listen so any accepted child fd inherits them.
    var bufSize: Int32 = 1024 * 1024
    _ = setsockopt(serverFd, SOL_SOCKET, SO_SNDBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))
    _ = setsockopt(serverFd, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathLen = socketPath.utf8.count
    // sun_path is 104 bytes on Darwin; -1 for the NUL terminator.
    if pathLen >= 104 {
        die("socket path too long (\(pathLen) >= 104 bytes): \(socketPath)")
    }
    withUnsafeMutablePointer(to: &addr.sun_path.0) { dst in
        socketPath.utf8CString.withUnsafeBufferPointer { src in
            _ = memcpy(dst, src.baseAddress, src.count)
        }
    }
    let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
    let bindOK = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
            bind(serverFd, sa, addrLen)
        }
    }
    if bindOK != 0 { die("bind(\(socketPath)) failed: \(String(cString: strerror(errno)))") }
    if listen(serverFd, 1) != 0 { die("listen() failed: \(String(cString: strerror(errno)))") }

    // chmod 0600 — only owner may connect. (umask may have stripped bits.)
    _ = chmod(socketPath, 0o600)

    FileHandle.standardError.write(Data("face-detect: ready (engine=\(activeEngine.rawValue), model=\(adaFaceVariant.rawValue), dim=\(engineDim()), idle_timeout=\(idleTimeoutSec)s, predict_timeout=\(predictTimeoutSec)s, embedding_format=\(embeddingFormat.rawValue), transport=unix-socket, socket=\(socketPath), sndbuf=\(bufSize))\n".utf8))

    let startTime = DispatchTime.now()
    var processed = 0
    let decoder = JSONDecoder()

    while true {
        let clientFd = accept(serverFd, nil, nil)
        if clientFd < 0 {
            // EINTR from SIGALRM is benign — alarm handler will _exit if it fires
            // for real timeout reasons. Just retry.
            if errno == EINTR { continue }
            FileHandle.standardError.write(Data("face-detect: accept() failed: \(String(cString: strerror(errno)))\n".utf8))
            continue
        }
        // Reinforce buffer sizes on accepted fd (Darwin doesn't always inherit).
        _ = setsockopt(clientFd, SOL_SOCKET, SO_SNDBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))
        _ = setsockopt(clientFd, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))

        // One FileHandle for the duplex socket fd. closeOnDealloc=false so we
        // control the lifecycle and can close() exactly once below.
        let handle = FileHandle(fileDescriptor: clientFd, closeOnDealloc: false)

        runWatchSession(
            inHandle: handle,
            outHandle: handle,
            decoder: decoder,
            startTime: startTime,
            processed: &processed,
            clientLabel: "socket"
        )

        close(clientFd)
        FileHandle.standardError.write(Data("face-detect: client disconnected (processed=\(processed)), waiting for reconnect\n".utf8))
        alarm(idleTimeoutSec)
    }
}

// MARK: - Mode: video

func cmdVideo(path: String, fps: Double) {
    guard fps > 0 else { die("--fps must be > 0 (got \(fps))") }

    let url = URL(fileURLWithPath: path)
    let asset = AVURLAsset(url: url)

    // Get duration synchronously
    let semaphore = DispatchSemaphore(value: 0)
    var duration: Double = 0
    Task {
        do {
            let d = try await asset.load(.duration)
            duration = CMTimeGetSeconds(d)
        } catch {}
        semaphore.signal()
    }
    semaphore.wait()

    guard duration > 0 else { die("cannot read video duration: \(path)") }

    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
    generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)

    let interval = 1.0 / fps
    var t = 0.0
    while t < duration {
        let time = CMTime(seconds: t, preferredTimescale: 600)
        let label = String(format: "%@@%.1fs", path, t)
        autoreleasepool {
            let sem = DispatchSemaphore(value: 0)
            var frameImage: CGImage?
            var frameError: Error?
            generator.generateCGImagesAsynchronously(forTimes: [NSValue(time: time)]) { _, img, _, _, err in
                frameImage = img
                frameError = err
                sem.signal()
            }
            sem.wait()
            if let cg = frameImage {
                let result = processImage(path: label, cgImage: cg)
                emit(result)
            } else {
                emit(ImageResult(image: label, width: 0, height: 0, elapsed_ms: 0,
                                 engine: activeEngine.rawValue, engine_dim: engineDim(),
                                 model: modelLabel(),
                                 description: nil, tags: nil, faces: nil, error: frameError?.localizedDescription ?? "frame extraction failed"))
            }
        }
        t += interval
    }
}

// MARK: - Mode: benchmark

func cmdBench(dir: String) {
    let fm = FileManager.default
    let extensions: Set<String> = ["heic", "jpg", "jpeg", "png", "tiff", "tif"]

    guard let files = try? fm.contentsOfDirectory(atPath: dir) else {
        die("cannot list directory: \(dir)")
    }

    let images = files.filter { f in
        extensions.contains((f as NSString).pathExtension.lowercased())
    }.map { (dir as NSString).appendingPathComponent($0) }

    guard !images.isEmpty else { die("no images found in \(dir)") }

    var totalFaces = 0
    var totalMs = 0
    var embeddingDim = 0

    for (i, path) in images.enumerated() {
        autoreleasepool {
            if let (cg, _, _) = loadCGImage(path: path) {
                let result = processImage(path: path, cgImage: cg)
                let faceCount = result.faces?.count ?? 0
                totalFaces += faceCount
                totalMs += result.elapsed_ms
                if embeddingDim == 0, let first = result.faces?.first {
                    embeddingDim = first.embedding.count
                }
                FileHandle.standardError.write(
                    Data("[\(i+1)/\(images.count)] \(path): \(faceCount) faces, \(result.elapsed_ms)ms\n".utf8))
            } else {
                FileHandle.standardError.write(Data("[\(i+1)/\(images.count)] \(path): FAILED\n".utf8))
            }
        }
    }

    let avg = images.count > 0 ? Double(totalMs) / Double(images.count) : 0
    let fps = avg > 0 ? 1000.0 / avg : 0

    emit(BenchResult(
        images: images.count,
        faces: totalFaces,
        total_ms: totalMs,
        avg_ms: (avg * 10).rounded() / 10,
        fps: (fps * 10).rounded() / 10,
        embedding_dim: embeddingDim
    ), pretty: true)
}

// MARK: - Argument parsing & dispatch

func extractGlobalFlags(_ args: [String]) -> [String] {
    var rest: [String] = []
    var i = 0
    while i < args.count {
        let a = args[i]
        if a == "--engine", i + 1 < args.count {
            if let eng = EmbeddingEngine(rawValue: args[i + 1]) {
                activeEngine = eng
            } else {
                die("invalid --engine: \(args[i + 1]) (use adaface or vision)")
            }
            i += 2
        } else if a == "--model", i + 1 < args.count {
            if let v = AdaFaceVariant(rawValue: args[i + 1]) {
                adaFaceVariant = v
            } else {
                die("invalid --model: \(args[i + 1]) (use ir18 or ir50)")
            }
            i += 2
        } else if a == "--lang", i + 1 < args.count {
            if let l = DescriptionLang(rawValue: args[i + 1]) {
                descLang = l
            } else {
                die("invalid --lang: \(args[i + 1]) (use fr or en)")
            }
            i += 2
        } else if a == "--min-quality", i + 1 < args.count {
            let v = Float(args[i + 1]) ?? 0.0
            minQuality = max(0.0, min(1.0, v))
            i += 2
        } else if a == "--timeout", i + 1 < args.count {
            globalTimeoutSec = max(1, Int(args[i + 1]) ?? 30)
            i += 2
        } else if a == "--idle-timeout", i + 1 < args.count {
            let raw = Int(args[i + 1]) ?? 1800
            let clamped = max(60, raw)
            if clamped != raw {
                FileHandle.standardError.write(Data("face-detect: --idle-timeout clamped from \(raw) to \(clamped)s (minimum 60)\n".utf8))
            }
            idleTimeoutSec = UInt32(clamped)
            i += 2
        } else if a == "--embedding-format", i + 1 < args.count {
            if let f = EmbeddingFormat(rawValue: args[i + 1].lowercased()) {
                embeddingFormat = f
            } else {
                FileHandle.standardError.write(Data("face-detect: invalid --embedding-format '\(args[i + 1])', expected 'float' or 'b64'\n".utf8))
                exit(2)
            }
            i += 2
        } else if a == "--predict-timeout", i + 1 < args.count {
            let raw = Int(args[i + 1]) ?? 60
            let clamped = max(5, raw)
            if clamped != raw {
                FileHandle.standardError.write(Data("face-detect: --predict-timeout clamped from \(raw) to \(clamped)s (minimum 5)\n".utf8))
            }
            predictTimeoutSec = UInt32(clamped)
            i += 2
        } else {
            rest.append(a)
            i += 1
        }
    }
    return rest
}

let rawArgs = Array(CommandLine.arguments.dropFirst())
let args = extractGlobalFlags(rawArgs)

let helpText = """
face-detect \(VERSION) — face detection + recognition embeddings via Apple Vision + AdaFace.

USAGE
  face-detect [FLAGS] <image>                              Single image → JSON
  face-detect [FLAGS] --batch                              stdin → NDJSON
  face-detect [FLAGS] --watch --socket <path>              Unix socket daemon (recommended, v0.6.0+)
  face-detect [FLAGS] --watch --in <fifo> --out <fifo>     FIFO daemon (back-compat)
  face-detect [FLAGS] --video <file> [--fps <rate>]        Video frames → NDJSON
  face-detect [FLAGS] --bench <folder>                     Throughput benchmark

GLOBAL FLAGS
  --engine adaface|vision    Embedding engine (default: adaface, fallback vision)
  --model ir18|ir50          AdaFace variant (default: ir18; ir50 = ResNet-50, more discriminant)
  --lang fr|en               Description language (default: fr)
  --min-quality 0.0-1.0      Skip faces below threshold (default: 0)
  --timeout <seconds>        SIGALRM kill after N seconds (default: 30, CLI modes)
  --idle-timeout <seconds>   Auto-exit if no request (default: 1800, --watch only, min 60)
  --predict-timeout <sec>    Exit 124 if a single image processing exceeds N sec (default: 60, --watch only, min 5)
                             Signals UE/ANE deadlock to the client → respawn the daemon.
  --embedding-format <fmt>   Embedding JSON format: 'float' (default, array of numbers, 6 sig figs)
                             or 'b64' (base64 Float32 little-endian under "embedding_b64").
                             Use 'b64' to halve response size and avoid macOS FIFO saturation
                             on multi-face images (>16 KB ≈ 4+ faces triggers kernel pipe stall).

SAFETY
  CLI modes (single, batch, video, bench) are DISABLED by default to prevent
  zombie processes from Neural Engine deadlocks. Set FACE_DETECT_ALLOW_CLI=1.
  Even with override, alarm() kills the process after --timeout seconds.

WATCH PROTOCOL (transport-agnostic — same JSON over FIFO or Unix socket)
  Request:  {"image":"/path"} or {"ping":true} or {"shutdown":true}
            Optional "id" field propagated as "id" in response.
  Response: ImageResult JSON, PongResponse, or ShutdownResponse, newline-terminated.
  Idle timeout:    Process exits 0 after --idle-timeout seconds without activity.
  Predict watchdog: Process exits 124 if a single image exceeds --predict-timeout.
                    Indicates ANE/CoreML deadlock — client should respawn the daemon.

WATCH TRANSPORTS
  --socket <path>  Unix domain socket (recommended). 1 MB SO_SNDBUF/SO_RCVBUF,
                   eliminates macOS FIFO 16 KB stall on multi-face responses.
                   One client at a time (listen backlog 1). chmod 0600 set on socket.
                   Connect from Node: net.createConnection({ path: '/tmp/face.sock' })
  --in/--out       Named FIFOs (legacy). Kept for back-compat with existing clients.
                   Subject to macOS PIPE_SIZE saturation — see FAQ.md.

SUPPORTED FORMATS
  HEIC, JPEG, PNG, TIFF

EMBEDDING ENGINES
  adaface (default): AdaFace Core ML, 512-dim L2-normalized,
    face-recognition specific. Best for identity clustering.
    Variants: ir18 (fast, default), ir50 (ResNet-50, more discriminant).
    Model loaded from FACE_DETECT_MODEL_PATH or
    /opt/homebrew/share/face-detect/AdaFace_<variant>.mlpackage
  vision: VNGenerateImageFeaturePrintRequest, 768-dim generic image similarity.
    Fallback when AdaFace model not found. Not face-specific.
"""

if args.isEmpty {
    die(helpText)
}

// Safety: only --watch, --help, and --version are allowed by default.
// CLI modes (single, batch, video, bench) can spawn zombie processes
// if the Neural Engine deadlocks. Override with FACE_DETECT_ALLOW_CLI=1.
let isWatchMode = args[0] == "--watch"
let isHelpMode = args[0] == "--help" || args[0] == "-h"
let isVersionMode = args[0] == "--version" || args[0] == "-V"
if !isWatchMode && !isHelpMode && !isVersionMode {
    let allowCLI = ProcessInfo.processInfo.environment["FACE_DETECT_ALLOW_CLI"] == "1"
    if !allowCLI {
        die("CLI mode disabled (zombie risk). Use --watch daemon or set FACE_DETECT_ALLOW_CLI=1 to override.")
    }
    // Nuclear timeout: POSIX alarm() sends SIGALRM at kernel level.
    // Works even if GCD is deadlocked or process is orphaned by sandbox.
    signal(SIGALRM) { _ in
        let msg: StaticString = "face-detect: SIGALRM timeout, force exit\n"
        _ = write(STDERR_FILENO, msg.utf8Start, msg.utf8CodeUnitCount)
        _exit(2)
    }
    alarm(UInt32(globalTimeoutSec))
}

// Load AdaFace model if needed (silent fallback to Vision on failure)
if activeEngine == .adaface && !isHelpMode && !isVersionMode {
    loadAdaFaceModel()
}

switch args[0] {
case "--batch":
    cmdBatch()

case "--watch":
    var inPath: String?
    var outPath: String?
    var socketPath: String?
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--in":     i += 1; if i < args.count { inPath = args[i] }
        case "--out":    i += 1; if i < args.count { outPath = args[i] }
        case "--socket": i += 1; if i < args.count { socketPath = args[i] }
        default: break
        }
        i += 1
    }
    if let sock = socketPath {
        if inPath != nil || outPath != nil {
            die("usage: --socket is mutually exclusive with --in/--out")
        }
        cmdWatchSocket(socketPath: sock)
    } else if let inp = inPath, let outp = outPath {
        cmdWatch(inPath: inp, outPath: outp)
    } else {
        die("usage: face-detect --watch (--socket <path> | --in <fifo> --out <fifo>)")
    }

case "--video":
    guard args.count >= 2 else {
        die("usage: face-detect --video <file> [--fps <rate>]")
    }
    let videoPath = args[1]
    var fps = 1.0
    var i = 2
    while i < args.count {
        if args[i] == "--fps", i + 1 < args.count, let f = Double(args[i + 1]) {
            fps = f; i += 2
        } else { i += 1 }
    }
    cmdVideo(path: videoPath, fps: fps)

case "--bench":
    guard args.count >= 2 else {
        die("usage: face-detect --bench <folder>")
    }
    cmdBench(dir: args[1])

case "--help", "-h":
    print(helpText)

case "--version", "-V":
    print("face-detect \(VERSION) (engine=\(activeEngine.rawValue), model=\(adaFaceVariant.rawValue), dim=\(engineDim()))")

default:
    cmdSingle(args[0])
}
