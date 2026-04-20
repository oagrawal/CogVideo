#!/usr/bin/env bash
# GPU 0 — prompts 0–8, all 4 adaptive modes
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu0_adaptive_sweep.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=0 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 0 \
    --end-idx 8 \
    --modes "cog_dc_adaptive_hi0.60_lo0.10,cog_dc_adaptive_hi0.50_lo0.10,cog_dc_adaptive_hi0.70_lo0.10,cog_dc_adaptive_hi0.60_lo0.10_late" \
    2>&1 | tee "$LOG"

echo "GPU 0 adaptive sweep done."
