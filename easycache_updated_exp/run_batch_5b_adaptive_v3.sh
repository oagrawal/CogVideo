#!/bin/bash
# Split 6 new configurations across 4 GPUs, full 33 prompts each.

# GPU 0
MODES0="cog_ec_fixed_0.15,cog_ec_fixed_0.20"
# GPU 1
MODES1="cog_ec_adaptive_hi0.15_lo0.075_f9_l6"
# GPU 2
MODES2="cog_ec_adaptive_hi0.15_lo0.075_f13_l8"
# GPU 3
MODES3="cog_ec_adaptive_hi0.20_lo0.075_f9_l6,cog_ec_adaptive_hi0.20_lo0.075_f13_l8"

tmux kill-session -t cogvideo_adaptive_v3 2>/dev/null || true
tmux new-session -d -s cogvideo_adaptive_v3

LOG_DIR="/nfs/oagrawal/CogVideo/easycache_updated_exp"
GEN_SCRIPT="/workspace/cogvideo/easycache_updated_exp/easycache_batch_generate_5b_adaptive.py"
OUT_DIR="/workspace/cogvideo/easycache_updated_exp/videos"

tmux send-keys -t cogvideo_adaptive_v3 "docker exec -e CUDA_VISIBLE_DEVICES=0 cogvideo python3 $GEN_SCRIPT --start-idx 0 --end-idx 33 --modes $MODES0 --output-dir $OUT_DIR > $LOG_DIR/batch_v3_gpu0.log 2>&1 &" C-m

tmux send-keys -t cogvideo_adaptive_v3 "docker exec -e CUDA_VISIBLE_DEVICES=1 cogvideo python3 $GEN_SCRIPT --start-idx 0 --end-idx 33 --modes $MODES1 --output-dir $OUT_DIR > $LOG_DIR/batch_v3_gpu1.log 2>&1 &" C-m

tmux send-keys -t cogvideo_adaptive_v3 "docker exec -e CUDA_VISIBLE_DEVICES=2 cogvideo python3 $GEN_SCRIPT --start-idx 0 --end-idx 33 --modes $MODES2 --output-dir $OUT_DIR > $LOG_DIR/batch_v3_gpu2.log 2>&1 &" C-m

tmux send-keys -t cogvideo_adaptive_v3 "docker exec -e CUDA_VISIBLE_DEVICES=3 cogvideo python3 $GEN_SCRIPT --start-idx 0 --end-idx 33 --modes $MODES3 --output-dir $OUT_DIR > $LOG_DIR/batch_v3_gpu3.log 2>&1 &" C-m

echo "All 4 GPU runs (v3) have been launched inside the 'cogvideo_adaptive_v3' tmux session."
echo "Tail logs with:"
echo "  tail -f $LOG_DIR/batch_v3_gpu{0..3}.log"
