#!/usr/bin/env python3
"""Convert AdaFace IR-50 to Core ML .mlpackage matching IR-18 spec.

IR-18 input: imageType named 'face_image', 112x112 RGB (colorSpace=30)
IR-18 output: multiArrayType named 'embedding'
"""
import sys
import os
import torch
import torch.nn as nn

REPO_DIR = "/tmp/AdaFace"
WEIGHTS_PATH = "/tmp/adaface_ir50.safetensors"

# 1. Load architecture from AdaFace repo
sys.path.insert(0, REPO_DIR)
from net import build_model

print("1/5 Building IR-50 architecture...")
model = build_model('ir_50')

# 2. Load weights with proper prefix mapping
print("2/5 Loading weights...")
from safetensors.torch import load_file
raw_sd = load_file(WEIGHTS_PATH)

# The safetensors keys have 'net.' prefix, model expects without
stripped_sd = {}
skipped = 0
for k, v in raw_sd.items():
    new_key = k
    # HF safetensors uses 'model.net.' prefix; strip to match AdaFace Backbone keys
    for prefix in ["model.net.", "net.", "model."]:
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            break
    # Skip num_batches_tracked (not in model)
    if "num_batches_tracked" in new_key:
        skipped += 1
        continue
    stripped_sd[new_key] = v

missing, unexpected = model.load_state_dict(stripped_sd, strict=False)
print(f"   Keys loaded: {len(stripped_sd)}, skipped: {skipped}")
print(f"   Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"   Missing sample: {missing[:3]}")
if unexpected:
    print(f"   Unexpected sample: {unexpected[:3]}")

if len(missing) > 0:
    print("   ERROR: model has missing weights — embeddings would be random!")
    # Debug: compare key patterns
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(stripped_sd.keys())
    print(f"   Model expects {len(model_keys)} keys, got {len(loaded_keys)}")
    print(f"   Model sample: {sorted(model_keys)[:5]}")
    print(f"   Loaded sample: {sorted(loaded_keys)[:5]}")
    sys.exit(1)

model.eval()

# 3. Verify output shape
print("3/5 Verifying model output...")
dummy = torch.randn(1, 3, 112, 112)
with torch.no_grad():
    out = model(dummy)
    if isinstance(out, (tuple, list)):
        print(f"   Raw output: tuple of {len(out)}, shapes: {[o.shape for o in out]}")
        class EmbeddingOnly(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x)[0]
        model = EmbeddingOnly(model)
        model.eval()
        out = model(dummy)
    print(f"   Final output shape: {out.shape}")
    assert out.shape == (1, 512), f"Expected (1, 512), got {out.shape}"

# 4. Trace and convert
print("4/5 Converting to Core ML (imageType input, matching IR-18 spec)...")
traced = torch.jit.trace(model, dummy)

import coremltools as ct

# Match IR-18 exactly: imageType input named 'face_image', scale for [-1, 1] normalization
# AdaFace expects: (pixel / 255.0 - 0.5) / 0.5 = pixel / 127.5 - 1.0
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="face_image",
        shape=(1, 3, 112, 112),
        scale=1.0 / 127.5,
        bias=[-1.0, -1.0, -1.0],
        color_layout="BGR",  # match IR-18 (john-rocky); VNCoreMLRequest feeds BGR
    )],
    outputs=[ct.TensorType(name="embedding")],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
)

# 5. Save
output_path = "/opt/homebrew/share/face-detect/AdaFace_IR50.mlpackage"
print(f"5/5 Saving to {output_path}...")
mlmodel.save(output_path)

size_mb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fns in os.walk(output_path)
    for f in fns
) / 1024 / 1024
print(f"Done! IR-50 model saved ({size_mb:.1f} MB)")
