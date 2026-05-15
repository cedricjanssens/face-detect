# Changelog

All notable changes to `face-detect` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/).

---

## [0.4.0] — 2026-05-16

### 🛡️ Hardened watch mode for single-daemon multiplexing

Watch mode becomes the recommended interface for clients (e.g. archiviste)
to avoid Neural Engine contention with other CoreML consumers (Ollama, Photos).

#### Added
- JSON request format: `{"id":"req-001","image":"/path"}` with `request_id` propagation in output
- Health check: `{"ping":true}` → `{"pong":true, "uptime_ms", "processed", "engine", "engine_dim"}`
- Structured stderr logging: `ready`, `processing`, `done` with timing and face count
- Input buffer cap at 64 KiB (drops pathological inputs without newline)

#### Changed
- Back-compat preserved: plain `path\n` input still works
- Watch protocol documented in README with concurrency rationale

#### Known limitations (require client-side handling)
- No per-image cancel/timeout (CoreML syscalls are not interruptible from Swift)
- If CoreML hangs in kernel space (state UE), client must kill the process and restart
- Daemon is single-threaded — multiplexing happens via FIFO queue, not concurrent threads

---

## [0.3.0] — 2026-05-16

### 🧬 Switch to face-specific embeddings

Major model change for face identity clustering.

#### Added
- **AdaFace IR-18 Core ML** embedding engine (default) — 512-dim L2-normalized vectors trained on WebFace4M
- `--engine adaface|vision` flag to choose embedding engine
- `--min-quality 0.0-1.0` flag to filter out low-quality face captures
- `engine` and `engine_dim` fields in JSON output
- Model search via `$FACE_DETECT_MODEL_PATH` env var
- `make face-detect-model` target to auto-download AdaFace IR-18 (~42 MB)

#### Changed
- Default embedding: **AdaFace 512d** instead of Vision FeaturePrint 768d
- Embedding now face-recognition specific — proper cosine similarity discrimination (intra-person > 0.4, inter-person < 0.1)
- Build now links CoreML framework (`-framework CoreML`)
- README rewritten with badges, emoji, integration examples, architecture diagram

#### Fallback behavior
- If AdaFace model is missing or fails to load → automatic fallback to Vision FeaturePrint (768d) with warning on stderr
- No crash, no breaking change for consumers that don't depend on a specific dimensionality

#### Known limitations
- Children under ~3 years: degraded recall (fundamental limitation of all open-source face models)
- Identical twins: similar embeddings (~0.5-0.6 cosine sim) — distinguish via metadata
- Model adds ~48 MB to install footprint

---

## [0.2.0] — 2026-05-16

### 🏷️ Image tags + auto-description

#### Added
- **Scene tags** via `VNClassifyImageRequest` — labels with confidence (people, child, outdoor, food, etc.)
- **Auto-description** synthesis combining face count + scene tags, in French
  - Examples: `"groupe de 3 personnes avec enfant(s), en intérieur"`, `"personne, en extérieur, herbe"`
- New JSON fields: `description` (string) and `tags` (array of `{label, confidence}`)

#### Performance
- Classification runs in the same Vision pass as face detection — near-zero overhead

---

## [0.1.0] — 2026-05-15

### 🎬 Initial release

#### Added
- Face detection via `VNDetectFaceLandmarksRequest` (bbox, confidence, head pose, landmarks)
- Face quality scoring via `VNDetectFaceCaptureQualityRequest`
- 768-dim image feature print via `VNGenerateImageFeaturePrintRequest`
- 12 landmark regions: faceContour, leftEye, rightEye, leftEyebrow, rightEyebrow, nose, noseCrest, medianLine, outerLips, innerLips, leftPupil, rightPupil
- 5 invocation modes:
  - `face-detect <image>` — single image, JSON to stdout
  - `face-detect --batch` — NDJSON streaming from stdin
  - `face-detect --watch --in <fifo> --out <fifo>` — FIFO daemon with auto-reconnect
  - `face-detect --video <file> --fps <rate>` — video frame extraction via AVFoundation
  - `face-detect --bench <folder>` — throughput benchmark
- Format support: HEIC, JPEG, PNG, TIFF
- Installation to `/opt/homebrew/bin/face-detect` for system-wide use
- Throughput: ~12 images/s on M4 Pro (M-series Neural Engine)
