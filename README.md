# face-detect

macOS CLI for face detection and embedding generation via Apple Vision Framework.
Designed for Apple Silicon (Neural Engine acceleration).

## Build

```bash
swiftc -O -framework AVFoundation face-detect.swift -o face-detect
```

Or from the repo root:

```bash
make face-detect          # builds bin/face-detect
make install              # copies to /usr/local/bin/
```

Requirements: macOS 14+, Xcode Command Line Tools.

## Usage

### Single image

```bash
face-detect photo.jpg
```

```json
{
  "image": "photo.jpg",
  "width": 4032,
  "height": 3024,
  "elapsed_ms": 142,
  "faces": [
    {
      "bbox": [0.32, 0.41, 0.18, 0.24],
      "confidence": 0.98,
      "quality": 0.85,
      "roll": -0.05,
      "yaw": 0.12,
      "pitch": -0.03,
      "embedding": [0.012, -0.034, ...],
      "landmarks": {
        "leftEye": [[0.38, 0.55], ...],
        "rightEye": [[0.44, 0.56], ...],
        "nose": [[0.41, 0.48], ...],
        "outerLips": [[0.39, 0.42], ...]
      }
    }
  ]
}
```

### Batch (stdin → NDJSON)

```bash
find ~/Photos -name "*.jpg" | face-detect --batch > results.ndjson
```

One JSON per line, one line per image. Errors are inline (never stops the batch).

### Watch mode (FIFO daemon)

```bash
mkfifo /tmp/face-in /tmp/face-out
face-detect --watch --in /tmp/face-in --out /tmp/face-out &

# From another process:
echo "/path/to/photo.jpg" > /tmp/face-in
read result < /tmp/face-out
```

Reconnects automatically when the writer disconnects.

### Video frame extraction

```bash
# 1 frame every 10 seconds
face-detect --video vacation.mp4 --fps 0.1

# 1 frame per second
face-detect --video interview.mp4 --fps 1
```

Output: NDJSON with `image` field as `"file.mp4@12.5s"`.

### Benchmark

```bash
face-detect --bench ~/Pictures/faces/
```

```json
{
  "images": 50,
  "faces": 73,
  "total_ms": 7200,
  "avg_ms": 144.0,
  "fps": 6.9,
  "embedding_dim": 768
}
```

### From Node.js

```javascript
const { execFileSync, spawn } = require('child_process');

// Single image
const result = JSON.parse(execFileSync('face-detect', ['photo.jpg']));
console.log(`Found ${result.faces.length} faces`);

// Batch streaming
const proc = spawn('face-detect', ['--batch']);
proc.stdout.on('data', chunk => {
  for (const line of chunk.toString().split('\n').filter(Boolean)) {
    const result = JSON.parse(line);
    console.log(result.image, result.faces?.length ?? 0, 'faces');
  }
});
proc.stdin.write('/path/to/img1.jpg\n');
proc.stdin.write('/path/to/img2.jpg\n');
proc.stdin.end();
```

## Output format

| Field | Type | Description |
|-------|------|-------------|
| `image` | string | Input path (or `path@Ts` for video frames) |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `elapsed_ms` | int | Processing time |
| `faces` | array | Detected faces (null on error) |
| `error` | string | Error message (null on success) |

### Face fields

| Field | Type | Description |
|-------|------|-------------|
| `bbox` | [x,y,w,h] | Normalized 0-1, origin bottom-left |
| `confidence` | float | Detection confidence |
| `quality` | float | Capture quality 0-1 (lighting, blur, pose) |
| `roll/yaw/pitch` | float | Head rotation in radians |
| `embedding` | [float] | 768-dim feature vector for similarity |
| `landmarks` | object | Named regions with [x,y] point arrays |

### Landmark regions

`faceContour`, `leftEye`, `rightEye`, `leftEyebrow`, `rightEyebrow`,
`nose`, `noseCrest`, `medianLine`, `outerLips`, `innerLips`,
`leftPupil`, `rightPupil`

## Embedding details

The embedding is a **768-dimensional** float vector from `VNGenerateImageFeaturePrintRequest`.
Each face is cropped (+20% padding) and processed independently.
Use cosine similarity or L2 distance for face matching/clustering.

This is Apple's general image feature print applied to face crops — not a
dedicated face recognition model. It works well for similarity/clustering
but is not a FaceNet/ArcFace-class embedding.

## Supported formats

HEIC, JPEG, PNG, TIFF (via macOS ImageIO).
