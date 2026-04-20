#!/bin/bash

# Split 33 prompts across 4 GPUs
# Ranges: 0-8, 8-16, 16-24, 24-33

# Create a tmux session named "cogvideo_easycache_batch"
tmux new-session -d -s cogvideo_easycache_batch

# GPU 0
tmux send-keys -t cogvideo_easycache_batch 'docker exec -e CUDA_VISIBLE_DEVICES=0 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b.py --start-idx 0 --end-idx 8 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_gpu0.log 2>&1 &' C-m

# GPU 1
tmux send-keys -t cogvideo_easycache_batch 'docker exec -e CUDA_VISIBLE_DEVICES=1 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b.py --start-idx 8 --end-idx 16 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_gpu1.log 2>&1 &' C-m

# GPU 2
tmux send-keys -t cogvideo_easycache_batch 'docker exec -e CUDA_VISIBLE_DEVICES=2 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b.py --start-idx 16 --end-idx 24 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_gpu2.log 2>&1 &' C-m

# GPU 3
tmux send-keys -t cogvideo_easycache_batch 'docker exec -e CUDA_VISIBLE_DEVICES=3 cogvideo python3 /workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b.py --start-idx 24 --end-idx 33 --output-dir /workspace/cogvideo/easycache_updated_exp/videos > /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_gpu3.log 2>&1 &' C-m

echo "All 4 GPU runs have been launched inside the 'cogvideo_easycache_batch' tmux session."
echo "You can check the logs at /nfs/oagrawal/CogVideo/easycache_updated_exp/batch_gpuX.log"
