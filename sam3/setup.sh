#!/usr/bin/env bash
set -euo pipefail

# SAM 3 Setup Script
# Prerequisites: uv, CUDA-capable GPU (16 GB+ VRAM recommended)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"

echo "=== SAM 3 Setup ==="

# 1. Install Python dependencies via uv
echo "[1/3] Installing dependencies..."
cd "$PROJECT_DIR"
uv add huggingface-hub
uv add transformers

# 2. Authenticate with Hugging Face (needed for gated model)
echo "[2/3] Hugging Face authentication..."
echo "SAM 3 is a gated model. You need to:"
echo "  1. Go to https://huggingface.co/facebook/sam3 and accept the license"
echo "  2. Create a token at https://huggingface.co/settings/tokens"
echo ""
if hf auth whoami &>/dev/null; then
    echo "Already logged in to Hugging Face."
else
    echo "Please log in to Hugging Face:"
    hf auth login
fi

# 3. Download the SAM 3 checkpoint
echo "[3/3] Downloading SAM 3 checkpoint..."
mkdir -p "$CHECKPOINT_DIR"
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='facebook/sam3',
    local_dir='$CHECKPOINT_DIR/sam3',
    ignore_patterns=['*.md', '*.txt', '.gitattributes'],
)
print('Download complete.')
"

echo ""
echo "=== Setup complete ==="
echo "Checkpoint saved to: $CHECKPOINT_DIR/sam3/"
echo "Run inference with:  uv run python sam3/detect_person.py"
