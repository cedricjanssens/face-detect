# Changelog

All notable changes to `face-detect` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · Versioning: [SemVer](https://semver.org/).

---

## [0.6.1] — 2026-05-17

### 🛡️ Hardening after code review

#### Fixed
- **Watchdog flag race** — rearm idle alarm BEFORE clearing `watchdogInPredict`.
  Previous order could misclassify a SIGALRM delivered right at the predict
  deadline: handler reads flag=0 → exit 0 (idle) instead of exit 124 (watchdog).
  Hard to trigger but real, especially under high load.
- **Socket file leak on signal-driven exit** — SIGTERM/SIGINT/SIGALRM handlers
  now `unlink(socketCleanupPath)` before `_exit()`. `unlink(2)` is async-signal-safe,
  the path is held as a heap-allocated C string (strdup) to avoid touching Swift
  String machinery from a signal handler. Startup `unlink()` continues to cover
  the SIGKILL / crash case.
- Shutdown via JSON now also unlinks the socket file (parity with signal exits).

No protocol changes. Drop-in upgrade from 0.6.0.

---

## [0.5.5] — 2026-05-16

### 📉 Compact embedding output (mitigate macOS FIFO 16 KB stall)

The macOS named-pipe kernel buffer is ~16 KB (`PIPE_SIZE`, may dynamically grow
to 64 KB but shrinks back). Responses with 3+ faces (~32 KB) saturate the
buffer → daemon's `write()` blocks → client sees truncated JSON or hangs.
This release halves typical response size to push the threshold higher; the
real fix (Unix domain socket) is planned for 0.6.0.

#### Changed
- Embedding values now serialized at **6 significant figures** by default
  (was IEEE 754 round-trip precision, ~9 sig figs). Average ~10% reduction
  per response, cosine-similarity error < 1e-6 (negligible for face matching).

#### Added
- **`--embedding-format <fmt>`** flag: `float` (default, back-compat) or
  `b64` (base64-encoded little-endian Float32). When `b64`, the field
  becomes `embedding_b64` (string) instead of `embedding` (array).
- 1-face response: 10.5 KB (float, IEEE) → 9.7 KB (float, 6 sig figs) → 7.0 KB (b64)
- 4-face response (typical worst case): ~42 KB → ~38 KB → ~22 KB

#### Client adoption (archiviste)
Pass `--embedding-format b64` to halve response size. Decode in Node:
```js
const buf = Buffer.from(face.embedding_b64, 'base64')
const embedding = new Float32Array(buf.buffer, buf.byteOffset, 512)
```
B64 still doesn't fully eliminate the FIFO stall for 5+ face images. Plan
to migrate to Unix socket transport in 0.6.0 (see roadmap).

---

## [0.5.4] — 2026-05-16

### 🎯 Refined Ollama/ANE detection + predict watchdog

#### Changed — ANE detection (no more false positives)
Pre-0.5.4 skipped ANE for any `ollama` process. That was overly conservative:
only MLX runners actually hold the Neural Engine. `ollama serve` alone, or
GGUF runners (`--ollama-engine`, Metal GPU), do not conflict with CoreML.

- ANE skip now triggers only on `pgrep -f 'ollama.runner.*mlx'` (MLX runner)
- Stderr message clarified: `ANE skipped (Ollama MLX runner detected)`
- New env override: `FACE_DETECT_FORCE_ANE=1` (bypass MLX detection)

Common Ollama setups load embedding models (e.g. `nomic-embed-text`, 100% GPU
via llama.cpp). Before 0.5.4, face-detect would silently fall back to CPU+GPU
in those cases, costing ~20-50ms/image for no real reason.

#### Added — predict watchdog (watch mode)
A per-image timeout that exits the daemon with code **124** when a single
`processImage()` call exceeds the threshold. Designed to signal ANE/CoreML
zombification ("Uninterruptible state") to the parent client, so it can
detect a dead daemon and respawn — instead of hanging on heartbeat timeout.

- New flag: `--predict-timeout <sec>` (default 60s, min 5s)
- Exit code 124 = predict watchdog fired (vs. 0 = idle timeout, 0 = clean shutdown)
- Reuses the existing `SIGALRM` infrastructure with a context flag
  (`watchdogInPredict`) read in async-signal-safe form

**Client contract**: treat exit 124 as "daemon was stuck on a single image,
respawn". Do not retry the same image on the new daemon — it likely triggered
the deadlock. Skip and continue.

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
