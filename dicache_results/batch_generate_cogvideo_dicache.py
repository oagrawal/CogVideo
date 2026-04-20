"""
Batch video generation for CogVideoX1.5-5B + DiCache (33 VBench prompts).

Modes (baseline + fixed thresholds now; adaptive added after calibration):
  cog_dc_baseline              — no DiCache, original forward
  cog_dc_fixed_0.05            — fixed rel_l1_thresh=0.05
  cog_dc_fixed_0.10            — fixed rel_l1_thresh=0.10
  cog_dc_fixed_0.15            — fixed rel_l1_thresh=0.15
  cog_dc_fixed_0.20            — fixed rel_l1_thresh=0.20
  cog_dc_fixed_0.25            — fixed rel_l1_thresh=0.25
  cog_dc_fixed_0.30            — fixed rel_l1_thresh=0.30
  cog_dc_adaptive_hi0.60_lo0.10      — adaptive, low=0.10 steps 0-19, high=0.60 steps 20-49
  cog_dc_adaptive_hi0.50_lo0.10      — adaptive, low=0.10 steps 0-19, high=0.50 steps 20-49
  cog_dc_adaptive_hi0.70_lo0.10      — adaptive, low=0.10 steps 0-19, high=0.70 steps 20-49
  cog_dc_adaptive_hi0.60_lo0.10_late — adaptive, low=0.10 steps 0-19+45-49, high=0.60 steps 20-44

Output layout:
  dicache_results/
    videos/{mode_name}/{prompt_idx:03d}-seed0.mp4
    generation_log_gpu{N}_p{start:02d}-{end:02d}_steps50.json

Usage:
  # GPU 0 — prompts 0–17 (exclusive end)
  CUDA_VISIBLE_DEVICES=0 python3 batch_generate_cogvideo_dicache.py \\
      --start-idx 0 --end-idx 17

  # GPU 1 — prompts 17–33
  CUDA_VISIBLE_DEVICES=1 python3 batch_generate_cogvideo_dicache.py \\
      --start-idx 17 --end-idx 33
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Allow importing run_cogvideo_dicache from the parent CogVideo directory
COGVIDEO_ROOT = Path(__file__).resolve().parent.parent
if str(COGVIDEO_ROOT) not in sys.path:
    sys.path.insert(0, str(COGVIDEO_ROOT))

from run_cogvideo_dicache import dicache_forward, run_generation  # noqa: E402


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

def _mode_config(mode_name: str) -> Dict:
    """
    Map a mode name to its DiCache configuration dict.

    Keys:
      type         — "baseline" | "fixed" (| "adaptive" added later)
      rel_l1_thresh — threshold for fixed mode (ignored for baseline)
      probe_depth  — number of probe blocks (default 1)
      # adaptive keys added later:
      # thresh_low, thresh_high, stable_start, stable_end
    """
    if mode_name == "cog_dc_baseline":
        return {"type": "baseline"}
    if mode_name == "cog_dc_fixed_0.05":
        return {"type": "fixed", "rel_l1_thresh": 0.05, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.10":
        return {"type": "fixed", "rel_l1_thresh": 0.10, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.15":
        return {"type": "fixed", "rel_l1_thresh": 0.15, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.20":
        return {"type": "fixed", "rel_l1_thresh": 0.20, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.25":
        return {"type": "fixed", "rel_l1_thresh": 0.25, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.30":
        return {"type": "fixed", "rel_l1_thresh": 0.30, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.35":
        return {"type": "fixed", "rel_l1_thresh": 0.35, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.40":
        return {"type": "fixed", "rel_l1_thresh": 0.40, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.50":
        return {"type": "fixed", "rel_l1_thresh": 0.50, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.60":
        return {"type": "fixed", "rel_l1_thresh": 0.60, "probe_depth": 1}
    if mode_name == "cog_dc_fixed_0.70":
        return {"type": "fixed", "rel_l1_thresh": 0.70, "probe_depth": 1}
    # --- adaptive modes (stable_start=20, stable_end=50 unless noted) ---
    # low threshold for steps 0-19, high for steps 20-49 (no late-volatile band)
    if mode_name == "cog_dc_adaptive_hi0.60_lo0.10":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.10, "thresh_high": 0.60,
                "stable_start": 20, "stable_end": 50}
    if mode_name == "cog_dc_adaptive_hi0.50_lo0.10":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.10, "thresh_high": 0.50,
                "stable_start": 20, "stable_end": 50}
    if mode_name == "cog_dc_adaptive_hi0.70_lo0.10":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.10, "thresh_high": 0.70,
                "stable_start": 20, "stable_end": 50}
    # low threshold for steps 0-19 AND steps 45-49 (last 5 steps); high for steps 20-44
    if mode_name == "cog_dc_adaptive_hi0.60_lo0.10_late":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.10, "thresh_high": 0.60,
                "stable_start": 20, "stable_end": 45}
    
    if mode_name == "cog_dc_adaptive_hi0.60_lo0.20":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.20, "thresh_high": 0.60,
                "stable_start": 20, "stable_end": 50}
    if mode_name == "cog_dc_adaptive_hi0.50_lo0.20":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.20, "thresh_high": 0.50,
                "stable_start": 20, "stable_end": 50}
    if mode_name == "cog_dc_adaptive_hi0.70_lo0.20":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.20, "thresh_high": 0.70,
                "stable_start": 20, "stable_end": 50}
    if mode_name == "cog_dc_adaptive_hi0.60_lo0.20_late":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.20, "thresh_high": 0.60,
                "stable_start": 20, "stable_end": 45}

    # --- New Fidelity-Optimized Modes ---
    if mode_name == "cog_dc_adaptive_hi0.25_lo0.05_late":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.05, "thresh_high": 0.25,
                "stable_start": 20, "stable_end": 45}
    if mode_name == "cog_dc_adaptive_hi0.30_lo0.05_late":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.05, "thresh_high": 0.30,
                "stable_start": 20, "stable_end": 45}
    if mode_name == "cog_dc_adaptive_hi0.35_lo0.10_late":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.10, "thresh_high": 0.35,
                "stable_start": 20, "stable_end": 45}
    if mode_name == "cog_dc_adaptive_hi0.25_lo0.10_early":
        return {"type": "adaptive", "probe_depth": 1,
                "thresh_low": 0.10, "thresh_high": 0.25,
                "stable_start": 15, "stable_end": 50}
                
    raise ValueError(f"Unknown mode: {mode_name!r}")


DEFAULT_MODES = [
    "cog_dc_baseline",
    "cog_dc_fixed_0.05",
    "cog_dc_fixed_0.10",
    "cog_dc_fixed_0.15",
    "cog_dc_fixed_0.20",
    "cog_dc_fixed_0.25",
    "cog_dc_fixed_0.30",
]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CogVideoX1.5-5B + DiCache generation for VBench evaluation."
    )
    parser.add_argument("--model-id", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument(
        "--prompts-json", type=str,
        default=str(COGVIDEO_ROOT / "vbench_eval" / "prompts_subset.json"),
    )
    parser.add_argument("--start-idx", type=int, required=True,
                        help="Inclusive start index into prompts list.")
    parser.add_argument("--end-idx", type=int, required=True,
                        help="Exclusive end index into prompts list.")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "videos"),
        help="Root directory; mode subdirs created here.",
    )
    parser.add_argument(
        "--modes", type=str,
        default=",".join(DEFAULT_MODES),
        help="Comma-separated mode names.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    with open(args.prompts_json, "r") as f:
        all_prompts = json.load(f)
    prompts_slice = all_prompts[args.start_idx : args.end_idx]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    mode_names: List[str] = [m.strip() for m in args.modes.split(",") if m.strip()]

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu_id}  prompts [{args.start_idx},{args.end_idx})  "
          f"modes={mode_names}  model={args.model_id}  seed={args.seed}")

    # Load pipeline once; keep it alive for all (prompt, mode) pairs
    pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Save original forward once before any patching
    cls = pipe.transformer.__class__
    if not hasattr(cls, "_original_forward"):
        cls._original_forward = cls.forward

    log: Dict = {
        "gpu": gpu_id,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "model_id": args.model_id,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "runs": [],
    }

    for local_i, p in enumerate(prompts_slice):
        prompt_idx = args.start_idx + local_i
        prompt_text = p["prompt_en"] if isinstance(p, dict) and "prompt_en" in p else str(p)

        for mode_name in mode_names:
            cfg = _mode_config(mode_name)

            mode_dir = out_root / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)
            safe_text = prompt_text.replace("/", "-")
            out_path = mode_dir / f"{safe_text}-0.mp4"

            if out_path.exists():
                print(f"  SKIP (exists): {mode_name} / prompt {prompt_idx}", flush=True)
                continue

            is_baseline = cfg["type"] == "baseline"
            is_adaptive = cfg["type"] == "adaptive"

            # Set class-level forward once (instance reset below handles per-run state)
            if is_baseline:
                cls.forward = cls._original_forward
            else:
                cls.forward = dicache_forward

            # Reset all DiCache state on the INSTANCE for this (prompt, mode) pair.
            # Instance attributes shadow stale class attributes from prior runs.
            transformer = pipe.transformer
            transformer.cnt = 0
            transformer.probe_depth = int(cfg.get("probe_depth", 1))
            transformer.num_steps = args.steps
            transformer.rel_l1_thresh = float(cfg.get("rel_l1_thresh", 0.0))
            transformer.ret_ratio = 0.0
            transformer.accumulated_rel_l1_distance = 0.0
            transformer.resume_flag = False
            transformer.previous_probe_hs = None
            transformer.residual_cache_hs = None
            transformer.residual_cache_ehs = None
            transformer.probe_residual_cache = None
            transformer.residual_window_hs = []
            transformer.probe_residual_window = []
            transformer.calibrate = False
            # adaptive (harmless to set even for fixed/baseline modes)
            transformer.adaptive = is_adaptive
            transformer.thresh_low = float(cfg.get("thresh_low", 0.05))
            transformer.thresh_high = float(cfg.get("thresh_high", 0.20))
            transformer.stable_start = int(cfg.get("stable_start", 8))
            transformer.stable_end = int(cfg.get("stable_end", 40))

            g = torch.Generator("cuda").manual_seed(args.seed)

            t0 = time.time()
            video = pipe(
                prompt=prompt_text,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=g,
            ).frames[0]
            e2e = time.time() - t0

            export_to_video(video, str(out_path), fps=args.fps)

            log["runs"].append({
                "prompt_idx": prompt_idx,
                "mode": mode_name,
                "time_seconds": round(e2e, 2),
                "video_path": str(out_path),
                "rel_l1_thresh": float(cfg.get("rel_l1_thresh", 0.0)),
                "probe_depth": int(cfg.get("probe_depth", 1)),
            })
            print(
                f"DONE  {mode_name:28}  prompt {prompt_idx:03d}  "
                f"e2e {e2e:.1f}s  -> {out_path.name}",
                flush=True,
            )

    log_path = (
        out_root.parent
        / f"generation_log_gpu{gpu_id}_p{args.start_idx:02d}-{args.end_idx:02d}_steps{args.steps}.json"
    )
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote log: {log_path}")


if __name__ == "__main__":
    main()
