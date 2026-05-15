// face-detect.swift — macOS CLI for face detection + embeddings via Apple Vision Framework.
// Build: swiftc -O -framework AVFoundation face-detect.swift -o face-detect
// Requires: macOS 14+, Apple Silicon recommended (Neural Engine).

import AppKit
import AVFoundation
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
    let faces: [FaceResult]?
    let error: String?
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

// MARK: - Embedding extraction

func extractEmbedding(cgImage: CGImage, bbox: CGRect) -> [Float] {
    // Expand bbox by 20% for context (hair, chin)
    let pad = CGFloat(0.2)
    let expanded = CGRect(
        x: max(0, bbox.minX - bbox.width * pad),
        y: max(0, bbox.minY - bbox.height * pad),
        width: min(1.0, bbox.width * (1 + 2 * pad)),
        height: min(1.0, bbox.height * (1 + 2 * pad))
    ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))

    let pixelRect = VNImageRectForNormalizedRect(expanded, cgImage.width, cgImage.height)
    guard pixelRect.width > 0, pixelRect.height > 0,
          let cropped = cgImage.cropping(to: pixelRect) else {
        return []
    }

    let req = VNGenerateImageFeaturePrintRequest()
    let handler = VNImageRequestHandler(cgImage: cropped)
    do {
        try handler.perform([req])
    } catch {
        return []
    }

    guard let obs = req.results?.first as? VNFeaturePrintObservation else {
        return []
    }

    let count = obs.elementCount
    var floats = [Float](repeating: 0, count: count)
    obs.data.withUnsafeBytes { rawBuf in
        guard let ptr = rawBuf.baseAddress else { return }
        memcpy(&floats, ptr, count * MemoryLayout<Float>.size)
    }
    return floats
}

// MARK: - Core pipeline

func processImage(path: String, cgImage: CGImage) -> ImageResult {
    let start = DispatchTime.now()
    let w = cgImage.width
    let h = cgImage.height

    // Phase 1: face detection + landmarks (single pass)
    let landmarksReq = VNDetectFaceLandmarksRequest()
    let handler = VNImageRequestHandler(cgImage: cgImage)

    do {
        try handler.perform([landmarksReq])
    } catch {
        let ms = elapsedMs(since: start)
        return ImageResult(image: path, width: w, height: h, elapsed_ms: ms, faces: nil,
                           error: "vision failed: \(error.localizedDescription)")
    }

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

    // Phase 3: per-face embedding + assembly
    var faces: [FaceResult] = []
    for (i, obs) in observations.enumerated() {
        let bb = obs.boundingBox
        let embedding = extractEmbedding(cgImage: cgImage, bbox: bb)
        let lm = extractLandmarks(face: obs)

        let face = FaceResult(
            bbox: [Double(bb.origin.x), Double(bb.origin.y),
                   Double(bb.width), Double(bb.height)],
            confidence: obs.confidence,
            quality: i < qualities.count ? qualities[i] : nil,
            roll: obs.roll?.doubleValue,
            yaw: obs.yaw?.doubleValue,
            pitch: obs.pitch?.doubleValue,
            embedding: embedding,
            landmarks: lm
        )
        faces.append(face)
    }

    let ms = elapsedMs(since: start)
    return ImageResult(image: path, width: w, height: h, elapsed_ms: ms, faces: faces, error: nil)
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
                                 faces: nil, error: "cannot load image"))
            }
        }
    }
}

// MARK: - Mode: watch (FIFO daemon)

func cmdWatch(inPath: String, outPath: String) {
    signal(SIGPIPE, SIG_IGN)

    // Graceful shutdown
    let shutdownSrc = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    shutdownSrc.setEventHandler { exit(0) }
    shutdownSrc.resume()
    signal(SIGTERM, SIG_IGN) // let DispatchSource handle it

    let intSrc = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    intSrc.setEventHandler { exit(0) }
    intSrc.resume()
    signal(SIGINT, SIG_IGN)

    FileHandle.standardError.write(Data("face-detect: watch mode started\n".utf8))

    while true {
        // Open blocks until a writer connects
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

            // Process complete lines
            while let nlRange = buffer.range(of: Data([0x0a])) {
                let lineData = buffer[buffer.startIndex..<nlRange.lowerBound]
                buffer.removeSubrange(buffer.startIndex...nlRange.lowerBound)

                guard let line = String(data: lineData, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines),
                      !line.isEmpty else { continue }

                autoreleasepool {
                    let result: ImageResult
                    if let (cg, _, _) = loadCGImage(path: line) {
                        result = processImage(path: line, cgImage: cg)
                    } else {
                        result = ImageResult(image: line, width: 0, height: 0, elapsed_ms: 0,
                                             faces: nil, error: "cannot load image")
                    }
                    let enc = JSONEncoder()
                    enc.outputFormatting = [.sortedKeys]
                    if let json = try? enc.encode(result) {
                        outHandle.write(json)
                        outHandle.write(Data([0x0a]))
                    }
                }
            }
        }

        inHandle.closeFile()
        outHandle.closeFile()
        FileHandle.standardError.write(Data("face-detect: writer disconnected, waiting for reconnect\n".utf8))
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
                                 faces: nil, error: frameError?.localizedDescription ?? "frame extraction failed"))
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

let args = Array(CommandLine.arguments.dropFirst())

if args.isEmpty {
    die("""
        usage: face-detect <image>
               face-detect --batch
               face-detect --watch --in <fifo> --out <fifo>
               face-detect --video <file> [--fps <rate>]
               face-detect --bench <folder>
        """)
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
    face-detect — detect faces and generate embeddings via Apple Vision Framework.

    USAGE
      face-detect <image>                              Single image → JSON stdout
      face-detect --batch                              stdin paths → NDJSON stdout
      face-detect --watch --in <fifo> --out <fifo>     FIFO daemon (long-running)
      face-detect --video <file> [--fps <rate>]        Video frames → NDJSON stdout
      face-detect --bench <folder>                     Benchmark throughput

    SUPPORTED FORMATS
      HEIC, JPEG, PNG, TIFF

    OUTPUT
      Each result is a JSON object with: image, width, height, elapsed_ms, faces[], error.
      Each face has: bbox [x,y,w,h], confidence, quality, roll, yaw, pitch,
      embedding (768 floats), landmarks {region: [[x,y],...]}.

    EMBEDDING
      768-dimensional feature vector from VNGenerateImageFeaturePrintRequest.
      Generated by cropping each detected face (+20% padding) and computing
      an image feature print. Useful for face similarity/clustering.
    """)

default:
    cmdSingle(args[0])
}
