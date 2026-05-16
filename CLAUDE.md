# CLAUDE.md

## What this is

macOS CLI for face detection + recognition embeddings. Single-file Swift binary using Apple Vision (detection, landmarks, quality, scene tags) and AdaFace Core ML (512-dim identity embeddings for clustering). No cloud calls, no dependencies beyond macOS frameworks.

## Build & test

```bash
make build            # → bin/face-detect
make test             # 45 assertions, requires AdaFace IR-18 model installed
make install          # → /opt/homebrew/bin/face-detect
make models           # download AdaFace IR-18 (~42 MB)
make models-ir50      # download AdaFace IR-50 (~83 MB)
```

Requirements: macOS 14+, Xcode Command Line Tools, Apple Silicon recommended.

CLI modes require `FACE_DETECT_ALLOW_CLI=1` (test script sets this automatically).

## Architecture

Single file: `face-detect.swift` (~1100 lines). No modules, no packages.

Key sections (grep for `// MARK:`):
- Output models: `ImageResult`, `FaceResult`, `TagResult`, `BenchResult`
- Watch protocol: `WatchRequest`, `PongResponse`, `ShutdownResponse`
- Embedding: `EmbeddingEngine` enum, AdaFace CoreML, Vision fallback
- Processing: `processImage()` — detection + landmarks + quality + embeddings
- Classification: `classifyImage()` — `VNClassifyImageRequest`
- Description: `describeImage()` — French/English synthesis
- Commands: `cmdSingle`, `cmdBatch`, `cmdWatch`, `cmdVideo`, `cmdBench`
- Entry: `extractGlobalFlags()`, dispatch switch

## Watch mode protocol (consumed by archiviste)

Daemon via named pipes (FIFOs):

```bash
face-detect --watch --in /tmp/face-in --out /tmp/face-out
```

**Input** (one JSON per line):
```json
{"image":"/path/to/photo.jpg","id":"req-001"}
{"ping":true,"id":"hb-1"}
{"shutdown":true,"id":"bye"}
```

**Output** (one JSON per line):
- Image → `ImageResult` with `id` propagated
- Ping → `{"pong":true,"id":"...","uptime_ms":...,"model":"ir18"}`
- Shutdown → `{"shutdown":true,"id":"...","processed":N}`

Single-threaded by design. One daemon serves all clients sequentially.

## Models

- IR-18 (default): `/opt/homebrew/share/face-detect/AdaFace_IR18.mlpackage`
- IR-50 (optional): `/opt/homebrew/share/face-detect/AdaFace_IR50.mlpackage`
- Override: `FACE_DETECT_MODEL_PATH=/path/to/model.mlpackage`

## Known limitations

- Neural Engine contention: auto-detects Ollama, skips ANE if running. Never run 2+ face-detect processes simultaneously.
- FIFO output buffer: 64KB kernel-side. Client MUST drain responses before sending next request (see FAQ.md).
- Children under 3: degraded recognition (all face models).
- CoreML can enter Uninterruptible state on ANE deadlock (only reboot fixes).

## Conventions

- Single-file Swift: everything in `face-detect.swift`, no splitting
- Versioning: SemVer in `VERSION` constant (line 21)
- Commit format: conventional commits (`feat`/`fix`/`chore`/`test`/`docs`: description)
- Output: JSON to stdout, diagnostics to stderr
- No package managers (SPM, CocoaPods) — `swiftc` only
- After `make build`, always install: `make install` (archiviste uses `/opt/homebrew/bin/face-detect`)
