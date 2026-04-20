#!/bin/bash
MODES="cog_dc_fixed_0.35,cog_dc_adaptive_hi0.25_lo0.05_late,cog_dc_adaptive_hi0.30_lo0.05_late,cog_dc_adaptive_hi0.35_lo0.10_late,cog_dc_adaptive_hi0.25_lo0.10_early"

MODES="cog_dc_fixed_0.35,cog_dc_adaptive_hi0.25_lo0.05_late,cog_dc_adaptive_hi0.30_lo0.05_late,cog_dc_adaptive_hi0.35_lo0.10_late,cog_dc_adaptive_hi0.25_lo0.10_early"

docker exec -w /workspace/cogvideo/dicache_results -e CUDA_VISIBLE_DEVICES=0 cogvideo python3 batch_generate_cogvideo_dicache.py \
    --start-idx 0 --end-idx 8 \
    --modes "$MODES"
