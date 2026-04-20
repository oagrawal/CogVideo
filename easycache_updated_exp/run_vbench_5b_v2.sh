#!/bin/bash
# Split VBench evaluation of 4 new adaptive modes across 4 GPUs.

SCRIPT="/nfs/oagrawal/CogVideo/easycache_updated_exp/run_vbench_eval.py"
LOG_DIR="/nfs/oagrawal/CogVideo/easycache_updated_exp"
CONTAINER="hv_eval_wan"

# Kill any old session
tmux kill-session -t cogvideo_vbench_v2 2>/dev/null || true
tmux new-session -d -s cogvideo_vbench_v2

# GPU 0
tmux send-keys -t cogvideo_vbench_v2 \
  "docker exec -e CUDA_VISIBLE_DEVICES=0 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_adaptive_hi0.10_lo0.075_f9_l6 \
   > ${LOG_DIR}/vbench_v2_gpu0.log 2>&1 &" C-m

# GPU 1
tmux send-keys -t cogvideo_vbench_v2 \
  "docker exec -e CUDA_VISIBLE_DEVICES=1 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_adaptive_hi0.125_lo0.075_f9_l6 \
   > ${LOG_DIR}/vbench_v2_gpu1.log 2>&1 &" C-m

# GPU 2
tmux send-keys -t cogvideo_vbench_v2 \
  "docker exec -e CUDA_VISIBLE_DEVICES=2 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_adaptive_hi0.10_lo0.075_f13_l8 \
   > ${LOG_DIR}/vbench_v2_gpu2.log 2>&1 &" C-m

# GPU 3
tmux send-keys -t cogvideo_vbench_v2 \
  "docker exec -e CUDA_VISIBLE_DEVICES=3 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_adaptive_hi0.125_lo0.075_f13_l8 \
   > ${LOG_DIR}/vbench_v2_gpu3.log 2>&1 &" C-m

echo "VBench GPU jobs launched for the 4 adaptive v2 modes in tmux session 'cogvideo_vbench_v2'."
echo "Tail logs with:"
echo "  tail -f ${LOG_DIR}/vbench_v2_gpu{0..3}.log"
