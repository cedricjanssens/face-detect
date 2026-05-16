// face-detect.swift — macOS CLI for face detection + recognition embeddings.
// Build: swiftc -O -framework AVFoundation -framework CoreML face-detect.swift -o face-detect
// Requires: macOS 14+, Apple Silicon recommended (Neural Engine).
//
// Embeddings: AdaFace IR-18 (Core ML, 512-dim L2-normalized) for face recognition.
// Fallback: Vision VNGenerateImageFeaturePrintRequest (768-dim, generic image similarity).
//
// Model search paths (in order):
//   1. $FACE_DETECT_MODEL_PATH
//   2. <binary_dir>/AdaFace_IR18.mlpackage
//   3. /opt/homebrew/share/face-detect/AdaFace_IR18.mlpackage
//   4. /usr/local/share/face-detect/AdaFace_IR18.mlpackage

import AppKit
import AVFoundation
import CoreML
import Foundation
import Vision

// MARK: - Utilities

func die(_ msg: String, code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
    exit(code)
}

func emit<T: Encodable>(_ value: T) {
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys]
    guard let data = try? enc.encode(value) else { die("encode failed") }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0a]))
}

func emitPretty<T: Encodable>(_ value: T) {
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys, .prettyPrinted]
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
    let description: String?
    let tags: [TagResult]?
    let faces: [FaceResult]?
    let error: String?
}

struct TagResult: Encodable {
    let label: String
    let confidence: Float
}

struct FaceResult: Encodable {
    let bbox: [Double]          // [x, y, w, h] normalized 0-1, origin bottom-left
    let confidence: Float
    let quality: Float?         // faceCaptureQuality 0-1
    let roll: Double?
    let yaw: Double?
    let pitch: Double?
    let embedding: [Float]      // 768 floats (macOS 14+) or empty on failure
    let landmarks: [String: [[Double]]]?
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

// Output for non-image control commands (ping). For image requests,
// we emit the normal ImageResult, optionally with `request_id` propagated.
struct PongResponse: Encodable {
    let pong: Bool
    let request_id: String?
    let uptime_ms: Int
    let processed: Int
    let engine: String
    let engine_dim: Int
}

struct ShutdownResponse: Encodable {
    let shutdown: Bool
    let request_id: String?
    let uptime_ms: Int
    let processed: Int
}

// Wrapper to add request_id to ImageResult without changing the base struct.
// We encode manually so the field order is deterministic and request_id
// is omitted when absent (back-compat with single-path input format).
struct IdentifiedImageResult: Encodable {
    let request_id: String?
    let result: ImageResult

    enum CodingKeys: String, CodingKey {
        case request_id
    }

    func encode(to encoder: Encoder) throws {
        // Encode ImageResult fields at top level, plus request_id when present.
        try result.encode(to: encoder)
        if let id = request_id {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .request_id)
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

// MARK: - Embedding engines

enum EmbeddingEngine: String {
    case adaface
    case vision
}

// Loaded once at startup, reused for all images.
var adaFaceModel: VNCoreMLModel?
var activeEngine: EmbeddingEngine = .adaface
var minQuality: Float = 0.0
var globalTimeoutSec: Int = 30
var idleTimeoutSec: UInt32 = 1800  // 30 min default for watch mode

func locateModel() -> URL? {
    let candidates: [String] = [
        ProcessInfo.processInfo.environment["FACE_DETECT_MODEL_PATH"] ?? "",
        Bundle.main.bundlePath + "/AdaFace_IR18.mlpackage",
        ((CommandLine.arguments.first as NSString?)?.deletingLastPathComponent ?? "") + "/AdaFace_IR18.mlpackage",
        "/opt/homebrew/share/face-detect/AdaFace_IR18.mlpackage",
        "/usr/local/share/face-detect/AdaFace_IR18.mlpackage",
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
        FileHandle.standardError.write(Data("face-detect: AdaFace model not found, falling back to Vision FeaturePrint (768d)\n".utf8))
        activeEngine = .vision
        return
    }

    // Load model with timeout — compileModel can deadlock on Neural Engine
    // when another process (Ollama/MLX) holds the ANE.
    let sem = DispatchSemaphore(value: 0)
    var loadedModel: VNCoreMLModel?
    var loadError: Error?

    DispatchQueue.global(qos: .userInitiated).async {
        do {
            let compiled = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiled)
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

// AdaFace: 512-dim L2-normalized embedding via Core ML (Neural Engine when possible)
func extractAdaFaceEmbedding(cgImage: CGImage, bbox: CGRect) -> [Float] {
    guard let model = adaFaceModel,
          let cropped = croppedFaceImage(cgImage: cgImage, bbox: bbox) else {
        return []
    }
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .scaleFill
    let handler = VNImageRequestHandler(cgImage: cropped)
    do {
        try handler.perform([request])
    } catch {
        return []
    }
    guard let result = request.results?.first as? VNCoreMLFeatureValueObservation,
          let array = result.featureValue.multiArrayValue else {
        return []
    }
    var floats = [Float](repeating: 0, count: array.count)
    for i in 0..<array.count {
        floats[i] = Float(array[i].doubleValue)
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

func extractEmbedding(cgImage: CGImage, bbox: CGRect) -> [Float] {
    switch activeEngine {
    case .adaface: return extractAdaFaceEmbedding(cgImage: cgImage, bbox: bbox)
    case .vision:  return extractVisionEmbedding(cgImage: cgImage, bbox: bbox)
    }
}

func engineDim() -> Int {
    switch activeEngine {
    case .adaface: return 512
    case .vision:  return 768
    }
}

// MARK: - Description generation

func generateDescription(faceCount: Int, tags: [TagResult]) -> String {
    let tagSet = Set(tags.map { $0.label })

    // Subject
    var subject: String
    if faceCount == 0 {
        subject = ""
    } else if faceCount == 1 {
        if tagSet.contains("baby") { subject = "bébé" }
        else if tagSet.contains("child") { subject = "enfant" }
        else { subject = "personne" }
    } else {
        if tagSet.contains("baby") || tagSet.contains("child") {
            subject = "groupe de \(faceCount) personnes avec enfant(s)"
        } else {
            subject = "groupe de \(faceCount) personnes"
        }
    }

    // Setting
    var setting = ""
    if tagSet.contains("outdoor") || tagSet.contains("sky") || tagSet.contains("land") {
        setting = "en extérieur"
    } else if tagSet.contains("structure") || tagSet.contains("furniture") || tagSet.contains("room") {
        setting = "en intérieur"
    }

    // Activity / context keywords
    var details: [String] = []
    let contextMap: [(String, String)] = [
        ("food", "repas"), ("drink", "boisson"), ("cake", "gâteau"),
        ("beach", "plage"), ("snow", "neige"), ("mountain", "montagne"),
        ("water", "eau"), ("pool", "piscine"), ("garden", "jardin"),
        ("grass", "herbe"), ("tree", "arbre"), ("flower", "fleur"),
        ("car", "voiture"), ("vehicle", "véhicule"),
        ("sport", "sport"), ("ball", "ballon"),
        ("animal", "animal"), ("dog", "chien"), ("cat", "chat"),
        ("celebration", "fête"), ("party", "fête"),
    ]
    for (tag, fr) in contextMap {
        if tagSet.contains(tag) { details.append(fr); break }
    }

    // Assemble
    var parts: [String] = []
    if !subject.isEmpty { parts.append(subject) }
    if !setting.isEmpty { parts.append(setting) }
    parts.append(contentsOf: details)

    if parts.isEmpty {
        // Fallback: top tag
        if let first = tags.first { return first.label }
        return "image"
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
        let embedding = extractEmbedding(cgImage: cgImage, bbox: bb)
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
    emitPretty(result)
}

// MARK: - Mode: batch (stdin → NDJSON stdout)

func cmdBatch() {
    while let line = readLine() {
        let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { continue }
        autoreleasepool {
            if let (cg, _, _) = loadCGImage(path: path) {
                emit(processImage(path: path, cgImage: cg))
            } else {
                emit(ImageResult(image: path, width: 0, height: 0, elapsed_ms: 0,
                                 engine: activeEngine.rawValue, engine_dim: engineDim(),
                                 description: nil, tags: nil, faces: nil, error: "cannot load image"))
            }
        }
    }
}

// MARK: - Mode: watch (FIFO daemon)

func cmdWatch(inPath: String, outPath: String) {
    signal(SIGPIPE, SIG_IGN)

    // Graceful shutdown via POSIX signal handlers (not GCD — fires while blocked in read).
    // Closures must NOT capture context (C function pointer requirement).
    signal(SIGTERM) { _ in
        "face-detect: SIGTERM, exiting\n".withCString { _ = write(STDERR_FILENO, $0, strlen($0)) }
        _exit(0)
    }
    signal(SIGINT) { _ in
        "face-detect: SIGINT, exiting\n".withCString { _ = write(STDERR_FILENO, $0, strlen($0)) }
        _exit(0)
    }

    // Idle timeout via POSIX alarm() — kernel-level, fires even if blocked in read().
    // Rearms after each message. Default 30 min, configurable via --idle-timeout.
    signal(SIGALRM) { _ in
        "face-detect: idle timeout, exiting\n".withCString { _ = write(STDERR_FILENO, $0, strlen($0)) }
        _exit(0)
    }
    alarm(idleTimeoutSec)

    let startTime = DispatchTime.now()
    var processed = 0

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

    logStderr("ready (engine=\(activeEngine.rawValue), dim=\(engineDim()), idle_timeout=\(idleTimeoutSec)s, in=\(inPath), out=\(outPath))")

    while true {
        // Opens block until writer/reader connect
        guard let inHandle = FileHandle(forReadingAtPath: inPath) else {
            die("cannot open input FIFO: \(inPath)")
        }
        guard let outHandle = FileHandle(forWritingAtPath: outPath) else {
            die("cannot open output FIFO: \(outPath)")
        }

        var buffer = Data()
        while true {
            let chunk = inHandle.availableData
            if chunk.isEmpty { break } // EOF — writer disconnected
            buffer.append(chunk)

            // Cap buffer growth at 64 KiB (paths > PATH_MAX are pathological)
            if buffer.count > 65536 {
                logStderr("input buffer overflow (>64KiB without newline), dropping")
                buffer.removeAll(keepingCapacity: true)
                continue
            }

            while let nlRange = buffer.range(of: Data([0x0a])) {
                let lineData = buffer[buffer.startIndex..<nlRange.lowerBound]
                buffer.removeSubrange(buffer.startIndex...nlRange.lowerBound)

                guard let line = String(data: lineData, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines),
                      !line.isEmpty else { continue }

                // Rearm idle timeout on any input
                alarm(idleTimeoutSec)

                autoreleasepool {
                    // Parse: JSON command, or plain path (back-compat)
                    var requestId: String? = nil
                    var imagePath: String? = nil
                    var isPing = false
                    var isShutdown = false

                    if line.hasPrefix("{") {
                        if let data = line.data(using: .utf8),
                           let req = try? JSONDecoder().decode(WatchRequest.self, from: data) {
                            requestId = req.id
                            imagePath = req.image
                            isPing = req.ping ?? false
                            isShutdown = req.shutdown ?? false
                        } else {
                            imagePath = line
                        }
                    } else {
                        imagePath = line
                    }

                    // Handle shutdown
                    if isShutdown {
                        logStderr("shutdown requested (processed=\(processed))")
                        emitJSON(ShutdownResponse(
                            shutdown: true,
                            request_id: requestId,
                            uptime_ms: uptimeMs(),
                            processed: processed
                        ), to: outHandle)
                        outHandle.closeFile()
                        _exit(0)
                    }

                    // Handle ping
                    if isPing {
                        emitJSON(PongResponse(
                            pong: true,
                            request_id: requestId,
                            uptime_ms: uptimeMs(),
                            processed: processed,
                            engine: activeEngine.rawValue,
                            engine_dim: engineDim()
                        ), to: outHandle)
                        return
                    }

                    guard let path = imagePath else {
                        logStderr("malformed request: \(line.prefix(100))")
                        return
                    }

                    let result: ImageResult
                    let t0 = DispatchTime.now()
                    if let (cg, _, _) = loadCGImage(path: path) {
                        logStderr("processing \(path)")
                        result = processImage(path: path, cgImage: cg)
                    } else {
                        result = ImageResult(
                            image: path, width: 0, height: 0, elapsed_ms: 0,
                            engine: activeEngine.rawValue, engine_dim: engineDim(),
                            description: nil, tags: nil, faces: nil, error: "cannot load image"
                        )
                    }
                    processed += 1
                    let took = Int((DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000)
                    let faceCount = result.faces?.count ?? 0
                    let errPart = result.error.map { " error=\($0)" } ?? ""
                    logStderr("done \(path) in \(took)ms (\(faceCount) faces)\(errPart)")

                    emitJSON(IdentifiedImageResult(request_id: requestId, result: result), to: outHandle)
                }
            }
        }

        inHandle.closeFile()
        outHandle.closeFile()
        logStderr("writer disconnected (processed=\(processed)), waiting for reconnect")
        // Rearm idle timeout after disconnect too
        alarm(idleTimeoutSec)
    }
}

// MARK: - Mode: video

func cmdVideo(path: String, fps: Double) {
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

    emitPretty(BenchResult(
        images: images.count,
        faces: totalFaces,
        total_ms: totalMs,
        avg_ms: (avg * 10).rounded() / 10,
        fps: (fps * 10).rounded() / 10,
        embedding_dim: embeddingDim
    ))
}

// MARK: - Argument parsing & dispatch

// Extract global flags (--engine, --min-quality) and return remaining args.
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
        } else if a == "--min-quality", i + 1 < args.count {
            let v = Float(args[i + 1]) ?? 0.0
            minQuality = max(0.0, min(1.0, v))
            i += 2
        } else if a == "--timeout", i + 1 < args.count {
            globalTimeoutSec = max(1, Int(args[i + 1]) ?? 30)
            i += 2
        } else if a == "--idle-timeout", i + 1 < args.count {
            idleTimeoutSec = UInt32(max(60, Int(args[i + 1]) ?? 1800))
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

if args.isEmpty {
    die("""
        usage: face-detect [GLOBAL_FLAGS] <image>
               face-detect [GLOBAL_FLAGS] --batch
               face-detect [GLOBAL_FLAGS] --watch --in <fifo> --out <fifo>
               face-detect [GLOBAL_FLAGS] --video <file> [--fps <rate>]
               face-detect [GLOBAL_FLAGS] --bench <folder>

        GLOBAL_FLAGS:
               --engine adaface|vision   (default: adaface)
               --min-quality 0.0-1.0     (default: 0, no filtering)
               --timeout <seconds>       (default: 30, CLI modes only)
               --idle-timeout <seconds>  (default: 1800, --watch only)

        NOTE: CLI modes (single, batch, video, bench) are disabled by default.
              Set FACE_DETECT_ALLOW_CLI=1 to override.
        """)
}

// Safety: only --watch (daemon) and --help are allowed by default.
// CLI modes (single, batch, video, bench) can spawn zombie processes
// if the Neural Engine deadlocks. Override with FACE_DETECT_ALLOW_CLI=1.
let isWatchMode = args[0] == "--watch"
let isHelpMode = args[0] == "--help" || args[0] == "-h"
if !isWatchMode && !isHelpMode {
    let allowCLI = ProcessInfo.processInfo.environment["FACE_DETECT_ALLOW_CLI"] == "1"
    if !allowCLI {
        die("CLI mode disabled (zombie risk). Use --watch daemon or set FACE_DETECT_ALLOW_CLI=1 to override.")
    }
    // Nuclear timeout: POSIX alarm() sends SIGALRM at kernel level.
    // Works even if GCD is deadlocked or process is orphaned by sandbox.
    signal(SIGALRM) { _ in
        let msg = "face-detect: SIGALRM timeout, force exit\n"
        msg.withCString { _ = write(STDERR_FILENO, $0, strlen($0)) }
        _exit(2)
    }
    alarm(UInt32(globalTimeoutSec))
}

// Load AdaFace model if needed (silent fallback to Vision on failure)
if activeEngine == .adaface {
    loadAdaFaceModel()
}

switch args[0] {
case "--batch":
    cmdBatch()

case "--watch":
    var inPath: String?
    var outPath: String?
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--in":  i += 1; if i < args.count { inPath = args[i] }
        case "--out": i += 1; if i < args.count { outPath = args[i] }
        default: break
        }
        i += 1
    }
    guard let inp = inPath, let outp = outPath else {
        die("usage: face-detect --watch --in <fifo> --out <fifo>")
    }
    cmdWatch(inPath: inp, outPath: outp)

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
    print("""
    face-detect — face detection + recognition embeddings via Apple Vision + AdaFace.

    USAGE
      face-detect [FLAGS] <image>                              Single image → JSON
      face-detect [FLAGS] --batch                              stdin → NDJSON
      face-detect [FLAGS] --watch --in <fifo> --out <fifo>     FIFO daemon
      face-detect [FLAGS] --video <file> [--fps <rate>]        Video frames → NDJSON
      face-detect [FLAGS] --bench <folder>                     Throughput benchmark

    GLOBAL FLAGS
      --engine adaface|vision    Embedding engine (default: adaface, fallback vision)
      --min-quality 0.0-1.0      Skip faces below threshold (default: 0)
      --timeout <seconds>        SIGALRM kill after N seconds (default: 30, CLI modes)
      --idle-timeout <seconds>   Auto-exit if no request (default: 1800, --watch only)

    SAFETY
      CLI modes (single, batch, video, bench) are DISABLED by default to prevent
      zombie processes from Neural Engine deadlocks. Set FACE_DETECT_ALLOW_CLI=1.
      Even with override, alarm() kills the process after --timeout seconds.

    WATCH PROTOCOL
      Input (FIFO):  {"image":"/path"} or {"ping":true} or {"shutdown":true}
                     Optional "id" field propagated as "request_id" in response.
      Output (FIFO): ImageResult JSON, PongResponse, or ShutdownResponse.
      Idle timeout:  Process exits after --idle-timeout seconds without activity.

    SUPPORTED FORMATS
      HEIC, JPEG, PNG, TIFF

    EMBEDDING ENGINES
      adaface (default): AdaFace IR-18 Core ML, 512-dim L2-normalized,
        face-recognition specific. Best for identity clustering.
        Model loaded from FACE_DETECT_MODEL_PATH or
        /opt/homebrew/share/face-detect/AdaFace_IR18.mlpackage
      vision: VNGenerateImageFeaturePrintRequest, 768-dim generic image similarity.
        Fallback when AdaFace model not found. Not face-specific.
    """)

default:
    cmdSingle(args[0])
}
