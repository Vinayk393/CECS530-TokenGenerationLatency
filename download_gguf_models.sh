#!/usr/bin/env bash
# download_gguf_models.sh
# Downloads TinyLlama GGUF variants needed for bench-07-llamacpp (measured Q4/Q8).
# Usage: bash download_gguf_models.sh
# Output: models/model-F16.gguf, models/model-Q8_0.gguf, models/model-Q4_K_M.gguf
set -e
mkdir -p models
BASE="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main"
echo "[1/3] Downloading F16 (~2.2 GB)..."
curl -L --progress-bar \
  "${BASE}/tinyllama-1.1b-chat-v1.0.F16.gguf" \
  -o models/model-F16.gguf
echo "[2/3] Downloading Q8_0 (~1.2 GB)..."
curl -L --progress-bar \
  "${BASE}/tinyllama-1.1b-chat-v1.0.Q8_0.gguf" \
  -o models/model-Q8_0.gguf
echo "[3/3] Downloading Q4_K_M (~670 MB)..."
curl -L --progress-bar \
  "${BASE}/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -o models/model-Q4_K_M.gguf
echo ""
echo "Done. To run measured quantization benchmark:"
echo "  make bench-07-llamacpp GGUF_DIR=models/"