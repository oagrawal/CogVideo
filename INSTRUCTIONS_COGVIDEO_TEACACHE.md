# CogVideo + TeaCache — Run Instructions

This document describes how to run **TeaCache** experiments on **CogVideo** (CogVideoX-2B, CogVideoX-5B, or CogVideoX1.5-5B) using the **diffusers** pipeline. It aligns with the [CogVideoX_T2V_Colab.ipynb](CogVideoX_T2V_Colab.ipynb) workflow but skips Colab-specific optimizations since you have good GPUs.

---

## TeaCache script location

The TeaCache script is at:

```
/nfs/oagrawal/CogVideo/
├── teacache_sample_video.py   # TeaCache script (diffusers CogVideoXPipeline)
├── vbench_eval/
│   ├── prompts_subset.json    # 33 VBench prompts (same as Wan/Mochi/HunyuanVideo)
│   └── batch_generate_cogvideo.py
├── CogVideoX_T2V_Colab.ipynb
└── INSTRUCTIONS_COGVIDEO_TEACACHE.md
```

The script uses **diffusers** (`CogVideoXPipeline`, `CogVideoXImageToVideoPipeline`) and loads models from HuggingFace, matching the Colab notebook.

---

## Difference from the Colab notebook (we skip Colab optimizations)

| Colab notebook (low-resource)        | Our setup (good GPUs)        |
|-------------------------------------|------------------------------|
| `torch.float16` (Turing GPU support) | `torch.bfloat16` (better quality) |
| Camenduru’s 5GB-sharded checkpoints  | Official `THUDM/...` checkpoints |
| `pipe.enable_sequential_cpu_offload()` | No CPU offload (full GPU)   |
| No bfloat16 (OOM on Turing)          | bfloat16 recommended          |

---

## Getting started — Run in containers (recommended)

### Step 1: Create the CogVideo container

From the host:

```bash
docker run -it --gpus all --name cogvideo \
  -v /nfs/oagrawal/CogVideo:/workspace/cogvideo \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash
```
docker start cogvideo
docker exec -it cogvideo bash

### Step 2: Install dependencies inside the container

```bash
cd /workspace/cogvideo

pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib   # for delta TEMNI plot
pip install hf_transfer  # optional: faster HF downloads
```

### Step 3: First step — Delta TEMNI plot (no TeaCache)

Run with `--rel_l1_thresh 0` to disable caching, record delta TEMNI at each diffusion step, and save the plot. This is the baseline run (same as Wan/Mochi/HunyuanVideo).

```bash
cd /workspace/cogvideo

python teacache_sample_video.py \
  --ckpts_path THUDM/CogVideoX1.5-5B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --rel_l1_thresh 0 \
  --output_path ./cogvideo_results \
  --seed 42
```

**Outputs** in `./cogvideo_results/`:
- Video: `teacache_CogVideoX1.5-5B_*_0.mp4`
- Delta TEMNI plot: `teacache_CogVideoX1.5-5B_*_0_delta_TEMNI_plot.png`
- Delta TEMNI values: `teacache_CogVideoX1.5-5B_*_0_delta_TEMNI.txt`

Models are loaded from HuggingFace on first run. Use the plot to guide threshold choices for TeaCache.

---


## TeaCache thresholds (with caching)

From the script’s help:

- `--rel_l1_thresh 0.1`: ~1.3× speedup
- `--rel_l1_thresh 0.2`: ~1.8× speedup (default)
- `--rel_l1_thresh 0.3`: ~2.1× speedup

---

## Models supported (matches `coefficients_dict` in script)

| Model                 | HF path                      |
|-----------------------|------------------------------|
| CogVideoX-2b          | `THUDM/CogVideoX-2b`         |
| CogVideoX-5b          | `THUDM/CogVideoX-5b`         |
| CogVideoX-5b-I2V      | `THUDM/CogVideoX-5b-I2V`     |
| CogVideoX1.5-5B       | `THUDM/CogVideoX1.5-5B`      |
| CogVideoX1.5-5B-I2V   | `THUDM/CogVideoX1.5-5B-I2V`  |

The script infers the mode from `ckpts_path.split("/")[-1]` for coefficient lookup.

---

## Example commands (T2V)

```bash
# CogVideoX1.5-5B
python teacache_sample_video.py \
  --ckpts_path THUDM/CogVideoX1.5-5B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --rel_l1_thresh 0.2 \
  --output_path ./cogvideo_results \
  --num_frames 81 \
  --height 768 --width 1360 \
  --fps 16 \
  --seed 42

# CogVideoX-2b (smaller)
python teacache_sample_video.py \
  --ckpts_path THUDM/CogVideoX-2b \
  --prompt "A panda strums a guitar in a bamboo forest." \
  --rel_l1_thresh 0.2 \
  --output_path ./cogvideo_results
```

---

## Image-to-video (I2V)

```bash
python teacache_sample_video.py \
  --ckpts_path THUDM/CogVideoX1.5-5B-I2V \
  --generate_type i2v \
  --image_path /path/to/image.png \
  --prompt "The scene comes to life with subtle motion." \
  --rel_l1_thresh 0.2 \
  --output_path ./cogvideo_results
```

---

## VBench evaluation pipeline (4 GPUs)

Same setup as Wan and Mochi: 33 VBench prompts, 4 modes (baseline + 3 TeaCache), VBench 16-dimension evaluation, fidelity metrics (PSNR/SSIM/LPIPS), and 3 output CSVs.

**Environments:** Video generation in **CogVideo** container (`cogvideo`). VBench and fidelity in **HunyuanVideo eval** container (`hunyuanvideo_eval_wan`).

### Modes and folder structure

Modes for VBench (matching TeaCache paper style):

| Mode | Description |
|------|-------------|
| `cogvideo_baseline` | No TeaCache (rel_l1_thresh 0) |
| `cogvideo_fixed_0.1` | Fixed threshold 0.1 |
| `cogvideo_fixed_0.3` | Fixed threshold 0.3 |
| `cogvideo_adaptive_0.1_17_0.3` | Adaptive: 0.1 then 0.3 (switch at step 17/50) |

```
CogVideo/vbench_eval/
├── prompts_subset.json
├── batch_generate_cogvideo.py
├── videos/
│   ├── cogvideo_baseline/    # {prompt}-{seed}.mp4
│   ├── cogvideo_fixed_0.1/
│   ├── cogvideo_fixed_0.3/
│   └── cogvideo_adaptive_0.1_17_0.3/
├── vbench_scores/
├── fidelity_metrics/
└── ...
```

### Step 1: Generate videos (CogVideo container, 4 GPUs)

Use tmux for long runs. Split 33 prompts across 4 GPUs: 8 + 8 + 8 + 9.

**Pre-download model once** (avoid 4× download):
```bash
docker exec -it cogvideo bash
cd /workspace/cogvideo
python3 -c "from diffusers import CogVideoXPipeline; CogVideoXPipeline.from_pretrained('THUDM/CogVideoX1.5-5B'); print('Model cached.')"
```

**GPU 0** (prompts 0–7):
```bash
tmux new -s cogvideo_gen0
docker exec -it cogvideo bash
cd /workspace/cogvideo
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_cogvideo.py \
  --output-dir vbench_eval/videos \
  --start-idx 0 --end-idx 8
```

if tmux session has already started: tmux attach -t cogvideo_gen0

**GPU 1** (prompts 8–15):
```bash
tmux new -s cogvideo_gen1
docker exec -it cogvideo bash
cd /workspace/cogvideo
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_cogvideo.py \
  --output-dir vbench_eval/videos \
  --start-idx 8 --end-idx 16
```

**GPU 2** (prompts 16–23):
```bash
tmux new -s cogvideo_gen2
docker exec -it cogvideo bash
cd /workspace/cogvideo
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/batch_generate_cogvideo.py \
  --output-dir vbench_eval/videos \
  --start-idx 16 --end-idx 24
```

**GPU 3** (prompts 24–32):
```bash
tmux new -s cogvideo_gen3
docker exec -it cogvideo bash
cd /workspace/cogvideo
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/batch_generate_cogvideo.py \
  --output-dir vbench_eval/videos \
  --start-idx 24 --end-idx 33
```

**Resume:** Re-run the same command; existing videos are skipped.

### Step 2: VBench evaluation (HunyuanVideo eval container, 4 GPUs by mode)

Requires the eval container that mounts `/nfs/oagrawal` (see Wan INSTRUCTIONS, Step 2.0).

**4-GPU split by mode (even, 1 mode per GPU):**

| GPU | Mode |
|-----|------|
| 0 | `cogvideo_baseline` |
| 1 | `cogvideo_fixed_0.1` |
| 2 | `cogvideo_fixed_0.3` |
| 3 | `cogvideo_adaptive_0.1_17_0.3` |

```bash
COGVIDEO_VBENCH=/nfs/oagrawal/CogVideo/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo

# GPU 0 (baseline)
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $COGVIDEO_VBENCH/videos \
  --save-dir $COGVIDEO_VBENCH/vbench_scores \
  --full-info $COGVIDEO_VBENCH/prompts_subset.json \
  --modes cogvideo_baseline

# GPU 1 (fixed 0.1)
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $COGVIDEO_VBENCH/videos \
  --save-dir $COGVIDEO_VBENCH/vbench_scores \
  --full-info $COGVIDEO_VBENCH/prompts_subset.json \
  --modes cogvideo_fixed_0.1

# GPU 2 (fixed 0.3)
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $COGVIDEO_VBENCH/videos \
  --save-dir $COGVIDEO_VBENCH/vbench_scores \
  --full-info $COGVIDEO_VBENCH/prompts_subset.json \
  --modes cogvideo_fixed_0.3

# GPU 3 (adaptive 0.1->0.3)
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $COGVIDEO_VBENCH/videos \
  --save-dir $COGVIDEO_VBENCH/vbench_scores \
  --full-info $COGVIDEO_VBENCH/prompts_subset.json \
  --modes cogvideo_adaptive_0.1_17_0.3
```

Or all modes on one GPU:
```bash
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $COGVIDEO_VBENCH/videos \
  --save-dir $COGVIDEO_VBENCH/vbench_scores \
  --full-info $COGVIDEO_VBENCH/prompts_subset.json \
  --modes cogvideo_baseline,cogvideo_fixed_0.1,cogvideo_fixed_0.3,cogvideo_adaptive_0.1_17_0.3
```

### Step 3: Fidelity metrics (HunyuanVideo eval container)

```bash
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py \
  --video-dir $COGVIDEO_VBENCH/videos \
  --baseline cogvideo_baseline \
  --modes cogvideo_fixed_0.1,cogvideo_fixed_0.3,cogvideo_adaptive_0.1_17_0.3 \
  --save-dir $COGVIDEO_VBENCH/fidelity_metrics
```

### Step 4: Compare results (3 CSVs)

```bash
cd $HV_ROOT
python3 vbench_eval/compare_results.py \
  --scores-dir $COGVIDEO_VBENCH/vbench_scores \
  --fidelity-dir $COGVIDEO_VBENCH/fidelity_metrics \
  --gen-log-dir $COGVIDEO_VBENCH/videos \
  --output-json $COGVIDEO_VBENCH/all_comparison_results.json \
  --modes cogvideo_baseline,cogvideo_fixed_0.1,cogvideo_fixed_0.3,cogvideo_adaptive_0.1_17_0.3
```

Outputs: `vbench_scores_table.csv`, `fidelity_table.csv`, `summary_table.csv` in `$COGVIDEO_VBENCH/`.

---

## Reference prompt (same as Wan / Mochi / HunyuanVideo)

```
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

---

## Quick reference

| Item   | Value                                      |
|--------|--------------------------------------------|
| Script | `CogVideo/teacache_sample_video.py`        |
| Backend| diffusers (same as Colab notebook)         |
| Models | HuggingFace `THUDM/CogVideoX*`             |
| GPU    | Use bfloat16; no sequential CPU offload   |
| VAE    | `pipe.vae.enable_slicing()` and `enable_tiling()` (optional on large GPUs) |
