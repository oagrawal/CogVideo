CUDA_VISIBLE_DEVICES=2 python3 batch_generate_cogvideo_dicache.py \
    --start-idx 16 \
    --end-idx 24 \
    --modes "cog_dc_adaptive_hi0.60_lo0.20,cog_dc_adaptive_hi0.50_lo0.20,cog_dc_adaptive_hi0.70_lo0.20,cog_dc_adaptive_hi0.60_lo0.20_late"
