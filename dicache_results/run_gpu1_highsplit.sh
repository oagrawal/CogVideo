#!/usr/bin/env bash
# GPU 1 — prompts 9–17 (8 prompts), modes 0.40–0.70
set -euo pipefail
LOG=/workspace/cogvideo/dicache_results/logs/gpu1_highsplit.log
mkdir -p "$(dirname "$LOG")"
cd /workspace/cogvideo
CUDA_VISIBLE_DEVICES=1 python3 dicache_results/batch_generate_cogvideo_dicache.py     --start-idx 9 --end-idx 17     --modes "cog_dc_fixed_0.40,cog_dc_fixed_0.50,cog_dc_fixed_0.60,cog_dc_fixed_0.70"     2>&1 | tee "$LOG"
echo 'GPU 1 highsplit done.'
