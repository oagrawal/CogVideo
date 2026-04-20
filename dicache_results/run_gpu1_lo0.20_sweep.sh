CUDA_VISIBLE_DEVICES=1 python3 batch_generate_cogvideo_dicache.py \
    --start-idx 8 \
    --end-idx 16 \
    --modes "cog_dc_adaptive_hi0.60_lo0.20,cog_dc_adaptive_hi0.50_lo0.20,cog_dc_adaptive_hi0.70_lo0.20,cog_dc_adaptive_hi0.60_lo0.20_late"
