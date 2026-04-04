#!/usr/bin/env bash
# GPU 1 — prompts 9–17, all fixed modes + baseline
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu1_fixed_sweep.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=1 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 9 \
    --end-idx 17 \
    --modes "cog_dc_baseline,cog_dc_fixed_0.05,cog_dc_fixed_0.10,cog_dc_fixed_0.15,cog_dc_fixed_0.20,cog_dc_fixed_0.25,cog_dc_fixed_0.30" \
    2>&1 | tee "$LOG"

echo "GPU 1 done."
