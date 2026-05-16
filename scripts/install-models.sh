#!/bin/bash
# install-models.sh — Download and install AdaFace Core ML models
#
# Usage:
#   install-models.sh [ir18|ir50|all] [--force]
#
# Models are installed to /opt/homebrew/share/face-detect/
# IR-18: direct download (~42 MB)
# IR-50: download weights + convert via Python (~167 MB download, needs torch+coremltools)
set -euo pipefail

MODEL_DIR="/opt/homebrew/share/face-detect"
IR18_URL="https://github.com/john-rocky/CoreML-Models/releases/download/adaface-v1/AdaFace_IR18.mlpackage.zip"
IR50_WEIGHTS_URL="https://huggingface.co/minchul/cvlface_adaface_ir50_ms1mv2/resolve/main/model.safetensors"
ADAFACE_REPO="https://github.com/mk-minchul/AdaFace.git"
CONVERT_VENV="/tmp/adaface-convert-venv"

# ─── Colors & progress ────────────────────────────────────────────
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
BLUE='\033[34m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

step_total=0
step_current=0

start_steps() { step_total=$1; step_current=0; }

step() {
    step_current=$((step_current + 1))
    local pct=$((step_current * 100 / step_total))
    local filled=$((pct / 5))
    local empty=$((20 - filled))
    local bar="${GREEN}"
    for ((i=0; i<filled; i++)); do bar+="█"; done
    bar+="${DIM}"
    for ((i=0; i<empty; i++)); do bar+="░"; done
    bar+="${RESET}"
    printf "\r  ${bar} ${BOLD}%3d%%${RESET}  %s" "$pct" "$*"
    if [[ $step_current -eq $step_total ]]; then echo ""; fi
}

ok()   { printf "  ${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "  ${YELLOW}!${RESET} %s\n" "$*"; }
fail() { printf "  ${RED}✗${RESET} %s\n" "$*"; exit 1; }
info() { printf "  ${BLUE}→${RESET} %s\n" "$*"; }

# ─── Download with progress bar ───────────────────────────────────
download() {
    local url="$1" dest="$2" label="$3"
    info "Downloading $label..."
    curl -fL --progress-bar --output "$dest" "$url"
}

# ─── IR-18 install ────────────────────────────────────────────────
install_ir18() {
    local dest="$MODEL_DIR/AdaFace_IR18.mlpackage"

    if [[ -d "$dest" && "$FORCE" != "1" ]]; then
        ok "IR-18 already installed ($(du -sh "$dest" | cut -f1 | tr -d ' '))"
        return
    fi

    printf "\n${BOLD}Installing AdaFace IR-18${RESET}\n"
    start_steps 4

    step "Creating directory..."
    mkdir -p "$MODEL_DIR"

    step "Downloading model (42 MB)..."
    download "$IR18_URL" /tmp/AdaFace_IR18.mlpackage.zip "AdaFace IR-18"

    step "Extracting..."
    [[ -d "$dest" ]] && rm -rf "$dest"
    cd "$MODEL_DIR" && unzip -q -o /tmp/AdaFace_IR18.mlpackage.zip
    rm -f /tmp/AdaFace_IR18.mlpackage.zip

    step "Done!"
    ok "IR-18 installed ($(du -sh "$dest" | cut -f1 | tr -d ' '))"
}

# ─── IR-50 install (conversion from PyTorch) ──────────────────────
install_ir50() {
    local dest="$MODEL_DIR/AdaFace_IR50.mlpackage"

    if [[ -d "$dest" && "$FORCE" != "1" ]]; then
        ok "IR-50 already installed ($(du -sh "$dest" | cut -f1 | tr -d ' '))"
        return
    fi

    printf "\n${BOLD}Installing AdaFace IR-50 (ResNet-50)${RESET}\n"
    info "Requires: Python 3, ~2 GB disk for conversion dependencies"
    start_steps 7

    # Step 1: Python venv
    step "Creating Python venv..."
    if [[ ! -d "$CONVERT_VENV" ]]; then
        python3 -m venv "$CONVERT_VENV" 2>/dev/null || fail "python3 -m venv failed. Install Python 3."
    fi

    # Step 2: Install dependencies
    step "Installing torch + coremltools (may take a while)..."
    "$CONVERT_VENV/bin/pip" install --quiet torch coremltools safetensors 2>/dev/null || fail "pip install failed"

    # Step 3: Clone AdaFace repo
    step "Cloning AdaFace architecture..."
    if [[ ! -d /tmp/AdaFace ]]; then
        git clone --depth 1 --quiet "$ADAFACE_REPO" /tmp/AdaFace 2>/dev/null
    fi

    # Step 4: Download weights
    step "Downloading IR-50 weights (167 MB)..."
    if [[ ! -f /tmp/adaface_ir50.safetensors ]]; then
        download "$IR50_WEIGHTS_URL" /tmp/adaface_ir50.safetensors "AdaFace IR-50 weights"
    fi

    # Step 5: Convert
    step "Converting to Core ML (Neural Engine optimized)..."
    mkdir -p "$MODEL_DIR"
    [[ -d "$dest" ]] && rm -rf "$dest"

    # Use the conversion script from the repo if available, otherwise inline
    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"
    if [[ -f "$script_dir/convert_ir50.py" ]]; then
        "$CONVERT_VENV/bin/python3" "$script_dir/convert_ir50.py" >/dev/null 2>&1 \
            || fail "Conversion failed. Check: $script_dir/convert_ir50.py"
    else
        # Inline minimal conversion
        "$CONVERT_VENV/bin/python3" - <<'PYEOF' 2>/dev/null || fail "Conversion failed"
import sys, os, torch, torch.nn as nn
sys.path.insert(0, "/tmp/AdaFace")
from net import build_model
from safetensors.torch import load_file
import coremltools as ct

model = build_model('ir_50')
raw = load_file("/tmp/adaface_ir50.safetensors")
sd = {}
for k, v in raw.items():
    nk = k
    for p in ["model.net.", "net.", "model."]:
        if k.startswith(p): nk = k[len(p):]; break
    if "num_batches_tracked" not in nk: sd[nk] = v
model.load_state_dict(sd, strict=True)
model.eval()

class W(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x): return self.m(x)[0]

w = W(model)
w.eval()
traced = torch.jit.trace(w, torch.randn(1, 3, 112, 112))
ml = ct.convert(traced,
    inputs=[ct.ImageType(name="face_image", shape=(1,3,112,112),
            scale=1/127.5, bias=[-1,-1,-1], color_layout="BGR")],
    outputs=[ct.TensorType(name="embedding")],
    convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14)
ml.save("/opt/homebrew/share/face-detect/AdaFace_IR50.mlpackage")
PYEOF
    fi

    # Step 6: Verify
    step "Verifying model..."
    if [[ ! -d "$dest" ]]; then
        fail "Model not found after conversion"
    fi
    local size
    size=$(du -sh "$dest" | cut -f1 | tr -d ' ')

    # Step 7: Cleanup
    step "Done!"
    ok "IR-50 installed ($size)"
}

# ─── Status ───────────────────────────────────────────────────────
status() {
    printf "\n${BOLD}AdaFace model status${RESET}\n"
    local ir18="$MODEL_DIR/AdaFace_IR18.mlpackage"
    local ir50="$MODEL_DIR/AdaFace_IR50.mlpackage"

    if [[ -d "$ir18" ]]; then
        ok "IR-18  $(du -sh "$ir18" | cut -f1 | tr -d ' ')  $ir18"
    else
        warn "IR-18  NOT INSTALLED  (run: $0 ir18)"
    fi

    if [[ -d "$ir50" ]]; then
        ok "IR-50  $(du -sh "$ir50" | cut -f1 | tr -d ' ')  $ir50"
    else
        warn "IR-50  NOT INSTALLED  (run: $0 ir50)"
    fi
    echo ""
}

# ─── Main ─────────────────────────────────────────────────────────
FORCE=0
TARGET="${1:-status}"
[[ "${2:-}" == "--force" ]] && FORCE=1

case "$TARGET" in
    ir18)    install_ir18 ;;
    ir50)    install_ir50 ;;
    all)     install_ir18; install_ir50 ;;
    status)  status ;;
    *)
        echo "Usage: $0 [ir18|ir50|all|status] [--force]"
        exit 1
        ;;
esac
