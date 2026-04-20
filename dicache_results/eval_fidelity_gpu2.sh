#!/bin/bash
MODE="cog_dc_adaptive_hi0.30_lo0.05_late"
docker exec -w /nfs/oagrawal/CogVideo/dicache_results/metrics -e CUDA_VISIBLE_DEVICES=2 hv_eval_wan python3 eval_with_json.py \
    --mode_name $MODE \
    --gt_video_dir ../videos/cog_dc_baseline \
    --generated_video_dir ../videos/$MODE \
    --output_json ../fidelity_metrics/${MODE}_vs_cog_dc_baseline.json
