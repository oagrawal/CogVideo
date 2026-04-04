#!/usr/bin/env bash
# GPU 3 — all 33 prompts, mode cog_dc_fixed_0.70
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu3_sweep2_0.70.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=3 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 0 \
    --end-idx 33 \
    --modes "cog_dc_fixed_0.70" \
    2>&1 | tee "$LOG"

echo 'GPU 3 (cog_dc_fixed_0.70) done.'
