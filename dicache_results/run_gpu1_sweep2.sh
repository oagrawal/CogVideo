#!/usr/bin/env bash
# GPU 1 — all 33 prompts, mode cog_dc_fixed_0.50
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu1_sweep2_0.50.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=1 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 0 \
    --end-idx 33 \
    --modes "cog_dc_fixed_0.50" \
    2>&1 | tee "$LOG"

echo 'GPU 1 (cog_dc_fixed_0.50) done.'
