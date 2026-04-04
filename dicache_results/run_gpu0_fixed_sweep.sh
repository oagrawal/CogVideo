#!/usr/bin/env bash
# GPU 0 — prompts 0–9, all fixed modes + baseline
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu0_fixed_sweep.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=0 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 0 \
    --end-idx 9 \
    --modes "cog_dc_baseline,cog_dc_fixed_0.05,cog_dc_fixed_0.10,cog_dc_fixed_0.15,cog_dc_fixed_0.20,cog_dc_fixed_0.25,cog_dc_fixed_0.30" \
    2>&1 | tee "$LOG"

echo "GPU 0 done."
