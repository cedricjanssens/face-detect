# 👤 face-detect

> Fast face detection + recognition embeddings for macOS — Apple Vision + AdaFace Core ML.

[![macOS 14+](https://img.shields.io/badge/macOS-14%2B-blue)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Neural%20Engine-success)](https://www.apple.com/mac/)
[![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)](https://swift.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Single-file Swift CLI that combines Apple Vision (detection, landmarks, quality, scene tags) with **AdaFace IR-18** Core ML model for **face identity embeddings** suitable for clustering — without any cloud calls or third-party services.

---

## ✨ Features

- 🎯 **Face detection** — bbox, confidence, head pose (roll/yaw/pitch)
- 🧬 **Face recognition embeddings** — AdaFace IR-18, 512-dim L2-normalized
- 📍 **Landmarks** — 12 facial regions (eyes, eyebrows, nose, lips, contour, pupils)
- 📊 **Quality scoring** — Apple's face capture quality (0-1)
- 🏷️ **Scene tags** — image classification (people, outdoor, food...) via Vision
- 📝 **Auto-description** — French synthesis "groupe de 3 personnes en intérieur"
- 🎬 **Video support** — frame extraction at custom FPS via AVFoundation
- 🚀 **5 invocation modes** — single, batch, watch (FIFO daemon), video, bench
- 🧠 **Neural Engine** — runs on M-series Apple Silicon
- 📦 **Zero dependencies** — single `.swift` file, builds with `swiftc`

---

## 🚀 Quick start

```bash
make face-detect          # builds bin/face-detect
make face-detect-model    # downloads AdaFace IR-18 (~42 MB)
make install              # → /usr/local/bin/ or /opt/homebrew/bin/

face-detect photo.jpg     # JSON output
```

**Requirements**: macOS 14+, Xcode CLI Tools, Apple Silicon recommended.

---

## 📖 Usage

### Single image

```bash
face-detect photo.jpg | jq .
```

```json
{
  "image": "photo.jpg",
  "width": 4032, "height": 3024,
  "engine": "adaface", "engine_dim": 512,
  "elapsed_ms": 142,
  "description": "groupe de 2 personnes avec enfant(s), en extérieur, herbe",
  "tags": [
    {"label": "people", "confidence": 0.95},
    {"label": "child", "confidence": 0.93},
    {"label": "outdoor", "confidence": 0.71}
  ],
  "faces": [
    {
      "bbox": [0.32, 0.41, 0.18, 0.24],
      "confidence": 0.98,
      "quality": 0.85,
      "roll": -0.05, "yaw": 0.12, "pitch": -0.03,
      "embedding": [0.048, -0.028, ...],
      "landmarks": {
        "leftEye": [[0.38, 0.55], ...],
        "rightEye": [...],
        "nose": [...]
      }
    }
  ]
}
```

### Batch (NDJSON streaming)

```bash
find ~/Photos -name "*.jpg" | face-detect --batch > results.ndjson
```

One JSON object per line. Errors inline — never stops the stream.

### Watch mode (FIFO daemon)

The recommended pattern for high-throughput processing: a single daemon
processes requests one at a time, model loaded once.

```bash
mkfifo /tmp/face-in /tmp/face-out
face-detect --watch --in /tmp/face-in --out /tmp/face-out &
```

**Input formats** (one per line):

```bash
# Simple: just a path (back-compat)
echo "/path/to/photo.jpg" > /tmp/face-in

# With request ID for correlation
echo '{"id":"req-001","image":"/path/to/photo.jpg"}' > /tmp/face-in

# Health check
echo '{"ping":true}' > /tmp/face-in
echo '{"ping":true,"id":"hb-42"}' > /tmp/face-in
```

**Output** (NDJSON on `/tmp/face-out`):

- Image request → standard `ImageResult` JSON, with `id` field if provided
- Ping → `{"pong": true, "id": "...", "uptime_ms": 12345, "processed": 87, "engine": "adaface", "engine_dim": 512, "model": "ir18"}`

**Daemon behavior**:
- 🔁 Reconnects automatically when writer disconnects
- 🛑 Graceful shutdown on SIGTERM / SIGINT
- 📊 Stderr logging: `ready`, `processing`, `done`, errors
- 🛡️ Buffer cap at 64 KiB (rejects pathological inputs)
- 🚫 Ignores SIGPIPE (survives reader disconnects mid-write)

**Concurrency note**: face-detect is single-threaded by design. For concurrent
clients, multiplex through this daemon rather than spawning parallel binaries —
running N face-detect processes contends for the Neural Engine and can deadlock
when other CoreML consumers (Ollama MLX, Photos.app, etc.) are active.

### Video frames

```bash
face-detect --video vacation.mp4 --fps 0.1   # 1 frame / 10s
face-detect --video interview.mp4 --fps 1    # 1 frame / s
```

Frame label in JSON: `"file.mp4@12.5s"`.

### Benchmark

```bash
face-detect --bench ~/Pictures/faces/
```

```json
{
  "images": 89,
  "faces": 128,
  "total_ms": 7671,
  "avg_ms": 86.2,
  "fps": 11.6,
  "embedding_dim": 512
}
```

### Global flags

```bash
face-detect --engine adaface photo.jpg      # default: AdaFace 512d
face-detect --engine vision photo.jpg       # fallback: Vision 768d
face-detect --min-quality 0.4 photo.jpg     # skip faces below threshold
```

---

## 🔬 Embedding engine

| Engine | Dim | Source | Notes |
|--------|-----|--------|-------|
| **adaface** (default) | 512 | AdaFace IR-18 Core ML, WebFace4M | Face-specific, best for identity clustering |
| **vision** (fallback) | 768 | `VNGenerateImageFeaturePrintRequest` | Generic image similarity, used if model missing |

**AdaFace metrics** (from upstream paper):
- LFW: 99.53% accuracy
- AgeDB: 96.47%
- License: MIT

**Cosine similarity thresholds** (with AdaFace):
- `> 0.4` → likely same person
- `< 0.1` → likely different people
- `0.1 - 0.4` → ambiguous (especially across ages or for children)

---

## 📦 Model installation

The Core ML model is searched in this order:

1. `$FACE_DETECT_MODEL_PATH` environment variable
2. `<binary_dir>/AdaFace_IR18.mlpackage`
3. `/opt/homebrew/share/face-detect/AdaFace_IR18.mlpackage`
4. `/usr/local/share/face-detect/AdaFace_IR18.mlpackage`

Auto-install via:

```bash
make face-detect-model
```

Or manually:

```bash
curl -fL https://github.com/john-rocky/CoreML-Models/releases/download/adaface-v1/AdaFace_IR18.mlpackage.zip \
  -o /tmp/AdaFace.zip
unzip /tmp/AdaFace.zip -d /opt/homebrew/share/face-detect/
```

---

## ⚠️ Caveats

### 👶 Children's faces

All open-source face recognition models degrade significantly on children under ~3 years old. AdaFace + WebFace4M is more age-diverse than MS-Celeb-1M alternatives, but infant clustering remains fundamentally limited.

**Recommendation**: for family photo libraries spanning many years, cluster **per age-bucket** (0-2, 3-5, 6-10, 11+) rather than globally per person.

### 🎭 Identical twins

AdaFace produces similar embeddings for identical twins (~0.5-0.6 cosine similarity, vs 0.4 threshold). This is a property of the model, not a bug. Distinguish via metadata or manual labeling.

### 📸 Image formats

Supported: **HEIC**, **JPEG**, **PNG**, **TIFF** (via macOS ImageIO).

---

## 🛠️ Integration

### Node.js (child_process)

```javascript
const { execFileSync, spawn } = require('child_process');

// Single image
const result = JSON.parse(execFileSync('face-detect', ['photo.jpg']));
console.log(`${result.faces.length} faces, ${result.engine} ${result.engine_dim}d`);

// Batch streaming
const proc = spawn('face-detect', ['--batch']);
proc.stdout.on('data', chunk => {
  for (const line of chunk.toString().split('\n').filter(Boolean)) {
    const result = JSON.parse(line);
    // ... process
  }
});
proc.stdin.write('/path/to/img.jpg\n');
proc.stdin.end();
```

### Python

```python
import json, subprocess

result = json.loads(subprocess.check_output(['face-detect', 'photo.jpg']))
for face in result['faces']:
    print(f"bbox={face['bbox']}, quality={face['quality']:.2f}, emb={len(face['embedding'])}d")
```

### Shell pipelines

```bash
# Cluster faces by similarity
find ~/Photos -name "*.jpg" \
  | face-detect --batch \
  | jq -c 'select(.faces | length > 0)' \
  > faces.ndjson
```

---

## 📐 Output reference

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `image` | string | Input path (or `path@Ts` for video frames) |
| `width` / `height` | int | Image dimensions in pixels |
| `elapsed_ms` | int | Processing time |
| `engine` | string | `"adaface"` or `"vision"` |
| `engine_dim` | int | Embedding dimensionality (512 or 768) |
| `description` | string | Auto-generated French description |
| `tags` | array | Scene labels with confidence |
| `faces` | array | Detected faces (`null` on error) |
| `error` | string | Error message (`null` on success) |

### Face fields

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | `[x,y,w,h]` | Normalized 0-1, origin bottom-left |
| `confidence` | float | Detection confidence |
| `quality` | float | Capture quality 0-1 |
| `roll` / `yaw` / `pitch` | float | Head rotation in radians |
| `embedding` | array | 512 or 768 floats |
| `landmarks` | object | 12 named regions (eyes, nose, lips...) |

### Landmark regions

`faceContour` · `leftEye` · `rightEye` · `leftEyebrow` · `rightEyebrow` · `nose` · `noseCrest` · `medianLine` · `outerLips` · `innerLips` · `leftPupil` · `rightPupil`

---

## 🏗️ Architecture

```
┌──────────────┐
│  CGImage     │  (HEIC/JPEG/PNG/TIFF via CGImageSource)
└──────┬───────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│  Vision      │    │  Vision          │
│  Landmarks   │    │  Classify        │
│  + Quality   │    │  (tags)          │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       │                     │
       ▼                     │
┌──────────────┐             │
│ Per-face     │             │
│ crop (+20%)  │             │
└──────┬───────┘             │
       │                     │
       ▼                     │
┌──────────────┐             │
│  AdaFace     │             │
│  IR-18       │             │
│  Core ML     │             │
│  (512d emb)  │             │
└──────┬───────┘             │
       │                     │
       └─────────┬───────────┘
                 ▼
         ┌──────────────┐
         │  JSON / NDJSON│
         └──────────────┘
```

---

## 📜 License

MIT. AdaFace model: MIT (upstream [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)).
Core ML conversion: [john-rocky/CoreML-Models](https://github.com/john-rocky/CoreML-Models).

---

## Troubleshooting

See [FAQ.md](FAQ.md) — covers common issues:
- Watch mode hangs (FIFO buffer saturation)
- Neural Engine / Ollama contention
- Integration patterns (Node.js, Python)
- Model selection (IR-18 vs IR-50)

---

See [CHANGELOG.md](CHANGELOG.md) for version history.
