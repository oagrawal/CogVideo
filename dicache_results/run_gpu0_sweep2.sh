#!/usr/bin/env bash
# GPU 0 — all 33 prompts, mode cog_dc_fixed_0.40
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu0_sweep2_0.40.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=0 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 0 \
    --end-idx 33 \
    --modes "cog_dc_fixed_0.40" \
    2>&1 | tee "$LOG"

echo 'GPU 0 (cog_dc_fixed_0.40) done.'
