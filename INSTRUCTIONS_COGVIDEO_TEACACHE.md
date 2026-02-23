# CogVideo + TeaCache — Run Instructions

This document describes how to run **TeaCache** experiments on **CogVideo** (CogVideoX-2B, CogVideoX-5B, or CogVideoX1.5-5B) using the **diffusers** pipeline. It aligns with the [CogVideoX_T2V_Colab.ipynb](CogVideoX_T2V_Colab.ipynb) workflow but skips Colab-specific optimizations since you have good GPUs.

---

## TeaCache script location

The TeaCache script is at:

```
/nfs/oagrawal/CogVideo/
├── teacache_sample_video.py   # <-- Your TeaCache script (uses diffusers CogVideoXPipeline)
├── CogVideoX_T2V_Colab.ipynb  # Official Colab notebook (reference)
├── INSTRUCTIONS_COGVIDEO_TEACACHE.md
└── ...
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

### Step 2: Install dependencies inside the container

```bash
cd /workspace/cogvideo

pip install --upgrade pip
pip install diffusers transformers accelerate
pip install hf_transfer  # optional: faster HF downloads
```

### Step 3: Run TeaCache (no model download step needed)

Models are loaded from HuggingFace on first run. Example for CogVideoX1.5-5B T2V:

```bash
cd /workspace/cogvideo

python teacache_sample_video.py \
  --ckpts_path THUDM/CogVideoX1.5-5B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --rel_l1_thresh 0.2 \
  --output_path ./cogvideo_results \
  --seed 42
```

---

## First step: Plot delta TEMNI (no TeaCache)

To record **delta TEMNI** over diffusion steps (baseline, no caching), use `--rel_l1_thresh 0`. This disables caching but still runs the full forward pass; the script logic will compute steps without skipping. For plotting, the existing script would need a small addition (similar to Wan/Mochi/HunyuanVideo) to append delta TEMNI values when `rel_l1_thresh == 0` and save a plot. Until that is added, you can run with `--rel_l1_thresh 0` to obtain a baseline video; plotting can be wired in later by mirroring the logic from `wan/Wan2.1/teacache_generate.py`’s `_plot_delta_temni`.

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
