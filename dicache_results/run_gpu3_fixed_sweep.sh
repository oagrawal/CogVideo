#!/usr/bin/env bash
# GPU 3 — prompts 25–33, all fixed modes + baseline
set -euo pipefail

LOG=/workspace/cogvideo/dicache_results/logs/gpu3_fixed_sweep.log
mkdir -p "$(dirname "$LOG")"

cd /workspace/cogvideo

CUDA_VISIBLE_DEVICES=3 python3 dicache_results/batch_generate_cogvideo_dicache.py \
    --start-idx 25 \
    --end-idx 33 \
    --modes "cog_dc_baseline,cog_dc_fixed_0.05,cog_dc_fixed_0.10,cog_dc_fixed_0.15,cog_dc_fixed_0.20,cog_dc_fixed_0.25,cog_dc_fixed_0.30" \
    2>&1 | tee "$LOG"

echo "GPU 3 done."
