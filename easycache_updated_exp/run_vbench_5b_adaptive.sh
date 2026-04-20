#!/bin/bash
# Split VBench evaluation of 3 new modes across 3 GPUs, leaving GPU 3 idle.
#   GPU 0 → cog_ec_adaptive_hi0.10_lo0.075
#   GPU 1 → cog_ec_fixed_0.125
#   GPU 2 → cog_ec_adaptive_hi0.125_lo0.075

SCRIPT="/nfs/oagrawal/CogVideo/easycache_updated_exp/run_vbench_eval.py"
LOG_DIR="/nfs/oagrawal/CogVideo/easycache_updated_exp"
CONTAINER="hv_eval_wan"

# Kill any old session
tmux kill-session -t cogvideo_vbench_adaptive 2>/dev/null || true

tmux new-session -d -s cogvideo_vbench_adaptive

# GPU 0
tmux send-keys -t cogvideo_vbench_adaptive \
  "docker exec -e CUDA_VISIBLE_DEVICES=0 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_adaptive_hi0.10_lo0.075 \
   > ${LOG_DIR}/vbench_adaptive_gpu0.log 2>&1 &" C-m

# GPU 1
tmux send-keys -t cogvideo_vbench_adaptive \
  "docker exec -e CUDA_VISIBLE_DEVICES=1 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_fixed_0.125 \
   > ${LOG_DIR}/vbench_adaptive_gpu1.log 2>&1 &" C-m

# GPU 2
tmux send-keys -t cogvideo_vbench_adaptive \
  "docker exec -e CUDA_VISIBLE_DEVICES=2 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_adaptive_hi0.125_lo0.075 \
   > ${LOG_DIR}/vbench_adaptive_gpu2.log 2>&1 &" C-m

echo "VBench GPU jobs launched for the 3 adaptive/fixed modes in tmux session 'cogvideo_vbench_adaptive'."
echo "Tail logs with:"
echo "  tail -f ${LOG_DIR}/vbench_adaptive_gpu{0..2}.log"
