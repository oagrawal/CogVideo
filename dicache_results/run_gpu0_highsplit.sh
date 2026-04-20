#!/usr/bin/env bash
# GPU 0 — prompts 0–9 (9 prompts), modes 0.40–0.70
set -euo pipefail
LOG=/workspace/cogvideo/dicache_results/logs/gpu0_highsplit.log
mkdir -p "$(dirname "$LOG")"
cd /workspace/cogvideo
CUDA_VISIBLE_DEVICES=0 python3 dicache_results/batch_generate_cogvideo_dicache.py     --start-idx 0 --end-idx 9     --modes "cog_dc_fixed_0.40,cog_dc_fixed_0.50,cog_dc_fixed_0.60,cog_dc_fixed_0.70"     2>&1 | tee "$LOG"
echo 'GPU 0 highsplit done.'
