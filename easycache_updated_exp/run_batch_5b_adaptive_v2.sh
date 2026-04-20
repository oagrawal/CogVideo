#!/bin/bash
# Split 4 adaptive configurations across 4 GPUs, full 33 prompts each.

# GPU 0
MODE0="cog_ec_adaptive_hi0.10_lo0.075_f9_l6"
# GPU 1
MODE1="cog_ec_adaptive_hi0.125_lo0.075_f9_l6"
# GPU 2
MODE2="cog_ec_adaptive_hi0.10_lo0.075_f13_l8"
# GPU 3
MODE3="cog_ec_adaptive_hi0.125_lo0.075_f13_l8"

tmux kill-session -t cogvideo_adaptive_v2 2>/dev/null || true
tmux new-session -d -s cogvideo_adaptive_v2

tmux send-keys -t cogvideo_adaptive_v2 "docker exec -e CUDA_VISIBLE_DEVICES=0 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b_adaptive.py --start-idx 0 --end-idx 33 --modes $MODE0 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_v2_gpu0.log 2>&1 &" C-m

tmux send-keys -t cogvideo_adaptive_v2 "docker exec -e CUDA_VISIBLE_DEVICES=1 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b_adaptive.py --start-idx 0 --end-idx 33 --modes $MODE1 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_v2_gpu1.log 2>&1 &" C-m

tmux send-keys -t cogvideo_adaptive_v2 "docker exec -e CUDA_VISIBLE_DEVICES=2 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b_adaptive.py --start-idx 0 --end-idx 33 --modes $MODE2 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_v2_gpu2.log 2>&1 &" C-m

tmux send-keys -t cogvideo_adaptive_v2 "docker exec -e CUDA_VISIBLE_DEVICES=3 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b_adaptive.py --start-idx 0 --end-idx 33 --modes $MODE3 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_v2_gpu3.log 2>&1 &" C-m

echo "All 4 GPU runs have been launched inside the 'cogvideo_adaptive_v2' tmux session."
echo "Tail logs with:"
echo "  tail -f /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_v2_gpu{0..3}.log"
