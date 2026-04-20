#!/bin/bash
# Split VBench evaluation of 5 modes across 4 GPUs:
#   GPU 0 → baseline
#   GPU 1 → cog_ec_fixed_0.025, cog_ec_fixed_0.05   (bundled: lightest two)
#   GPU 2 → cog_ec_fixed_0.075
#   GPU 3 → cog_ec_fixed_0.10

SCRIPT="/nfs/oagrawal/CogVideo/easycache_updated_exp/run_vbench_eval.py"
LOG_DIR="/nfs/oagrawal/CogVideo/easycache_updated_exp"
CONTAINER="hv_eval_wan"

# Kill any old session
tmux kill-session -t cogvideo_vbench 2>/dev/null || true

tmux new-session -d -s cogvideo_vbench

# GPU 0 — baseline
tmux send-keys -t cogvideo_vbench \
  "docker exec -e CUDA_VISIBLE_DEVICES=0 ${CONTAINER} python3 ${SCRIPT} \
   --modes baseline \
   > ${LOG_DIR}/vbench_gpu0.log 2>&1 &" C-m

# GPU 1 — fixed 0.025 + fixed 0.05 (bundle two light modes)
tmux send-keys -t cogvideo_vbench \
  "docker exec -e CUDA_VISIBLE_DEVICES=1 ${CONTAINER} python3 ${SCRIPT} \
   --modes 'cog_ec_fixed_0.025,cog_ec_fixed_0.05' \
   > ${LOG_DIR}/vbench_gpu1.log 2>&1 &" C-m

# GPU 2 — fixed 0.075
tmux send-keys -t cogvideo_vbench \
  "docker exec -e CUDA_VISIBLE_DEVICES=2 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_fixed_0.075 \
   > ${LOG_DIR}/vbench_gpu2.log 2>&1 &" C-m

# GPU 3 — fixed 0.10
tmux send-keys -t cogvideo_vbench \
  "docker exec -e CUDA_VISIBLE_DEVICES=3 ${CONTAINER} python3 ${SCRIPT} \
   --modes cog_ec_fixed_0.10 \
   > ${LOG_DIR}/vbench_gpu3.log 2>&1 &" C-m

echo "All 4 VBench GPU jobs launched in tmux session 'cogvideo_vbench'."
echo "Tail logs with:"
echo "  tail -f ${LOG_DIR}/vbench_gpu0.log"
echo "  tail -f ${LOG_DIR}/vbench_gpu1.log"
echo "  tail -f ${LOG_DIR}/vbench_gpu2.log"
echo "  tail -f ${LOG_DIR}/vbench_gpu3.log"
