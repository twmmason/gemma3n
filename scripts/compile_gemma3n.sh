#!/usr/bin/env bash
# compile_gemma3n.sh - Compile & Quantise Gemma-3 n E4B-it to WebGPU artifacts
#
# Usage: ./scripts/compile_gemma3n.sh
# Run this once on a CUDA-capable machine containing the Gemma weights.
#
# After creating the script remember to make it executable:
#   chmod +x scripts/compile_gemma3n.sh
#
# This script follows build book §2 “Compile & Quantise”.

set -euo pipefail

MODEL=gemma-3n-e4b-it
HF_REPO=https://huggingface.co/google/$MODEL

# 2.0 Clone FP16 weights
git lfs clone "$HF_REPO"

mkdir -p build && cd build

# 2.1 Compile & 4-bit-quantise to WebGPU artifacts
python -m mlc_llm.build \
  --model-path ../"$MODEL" \
  --target webgpu \
  --quantization q4f16_ft \
  --max-seq-len 8192 \
  --artifact-path ./gemma3n_art

# 2.2 Package & shard to stay below Chrome’s 2 GB ArrayBuffer cap
python -m mlc_llm.package \
  --artifact-path ./gemma3n_art \
  --embed-cache \
  --zip \
  --output ../frontend/public/gemma3n