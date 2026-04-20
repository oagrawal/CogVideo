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

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from easycache_cogvideo import easycache_forward, easycache_baseline_forward


def _load_prompts(prompts_path: Path) -> List[Dict]:
    with open(prompts_path, "r") as f:
        data = json.load(f)
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generation for CogVideoX1.5-5B + EasyCache (updated settings).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="CogVideoX model id or local path.",
    )
    parser.add_argument(
        "--prompts-json",
        type=str,
        default="/workspace/cogvideo/easycache_updated_exp/prompts_subset.json",
        help="Path to prompts_subset.json.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        required=True,
        help="Inclusive start index into prompts list.",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        required=True,
        help="Exclusive end index into prompts list.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="CFG scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for all videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/cogvideo/easycache_updated_exp/videos",
        help="Root directory where mode subfolders and videos are saved.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,cog_ec_fixed_0.025,cog_ec_fixed_0.05,cog_ec_fixed_0.075,cog_ec_fixed_0.10",
        help="Comma-separated list of modes.",
    )
    return parser.parse_args()


def _mode_config(mode_name: str) -> Dict:
    """Map mode names to EasyCache config."""
    if mode_name == "baseline":
        return {"mode": "baseline"}
    elif mode_name.startswith("cog_ec_fixed_"):
        thresh = float(mode_name.split("_")[-1])
        return {"mode": "easycache", "thresh": thresh, "ret_steps": 5}
    elif mode_name.startswith("cog_ec_adapt_"):
        # Not requested for this run, but keeping parser open
        # format we might want: cog_ec_adapt_l0.035_h0.07_f10_l8
        pass
    raise ValueError(f"Unsupported mode name: {mode_name}")


def main() -> None:
    args = _parse_args()

    prompts_path = Path(args.prompts_json)
    prompts_all = _load_prompts(prompts_path)
    prompts_slice = prompts_all[args.start_idx : args.end_idx]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    transformer = pipe.transformer
    cls = transformer.__class__
    if not hasattr(cls, "_original_forward"):
        cls._original_forward = cls.forward

    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]

    log: Dict = {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "model_id": args.model_id,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "runs": [],
    }
    per_video_log_path = (
        out_root.parent
        / f"generation_log_gpu{log['gpu']}_p{args.start_idx:02d}-{args.end_idx:02d}_steps{args.steps}.jsonl"
    )

    for local_i, p in enumerate(prompts_slice):
        prompt_idx = args.start_idx + local_i
        prompt_text = p["prompt"] if isinstance(p, dict) and "prompt" in p else str(p)

        for mode_name in mode_names:
            cfg = _mode_config(mode_name)

            mode_dir = out_root / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)
            out_path = mode_dir / f"{prompt_idx:03d}-seed{args.seed}.mp4"

            if out_path.exists():
                print(f"Skipping {out_path}, already exists")
                continue

            transformer.cnt = 0
            transformer.num_steps = args.steps
            transformer.total_time = 0.0
            transformer.k = None
            transformer.previous_raw_input = None
            transformer.previous_output = None
            transformer.prev_prev_raw_input = None
            transformer.k_history = []
            transformer.pred_change_history = []
            transformer.cache = None
            transformer.accumulated_error = 0.0
            transformer.ret_steps = int(cfg.get("ret_steps", 5))
            transformer.thresh = float(cfg.get("thresh", 0.0)) if cfg["mode"] == "easycache" else 0.0
            transformer.thresh_low = float(cfg.get("thresh_low", 0.0)) if "thresh_low" in cfg else None
            transformer.thresh_high = float(cfg.get("thresh_high", 0.0)) if "thresh_high" in cfg else None
            transformer.first_steps = int(cfg.get("first_steps", 0))
            transformer.last_steps = int(cfg.get("last_steps", 0))

            if cfg["mode"] == "baseline":
                cls.forward = easycache_baseline_forward
            else:
                cls.forward = easycache_forward

            g = torch.Generator(device=device).manual_seed(args.seed)

            t0 = time.time()
            video = pipe(
                prompt=prompt_text,
                height=768,
                width=1360,
                num_frames=81,
                use_dynamic_cfg=True,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=g,
                output_type="np",
            ).frames[0]
            e2e = time.time() - t0
            dit = float(transformer.total_time)

            export_to_video(video, str(out_path), fps=16)

            log["runs"].append(
                {
                    "prompt_idx": prompt_idx,
                    "mode": mode_name,
                    "time_seconds": e2e,
                    "dit_time_seconds": dit,
                    "ret_steps": int(transformer.ret_steps),
                    "thresh": float(cfg.get("thresh", 0.0)),
                    "video_path": str(out_path),
                }
            )
            # Write one record immediately so timing survives interrupted runs.
            with open(per_video_log_path, "a") as f:
                f.write(json.dumps(log["runs"][-1]) + "\n")
                f.flush()
            print(
                f"DONE {mode_name} prompt {prompt_idx} e2e: {e2e:.2f}s dit: {dit:.2f}s",
                flush=True,
            )

    log_path = out_root.parent / f"generation_log_gpu{log['gpu']}_p{args.start_idx:02d}-{args.end_idx:02d}_steps{args.steps}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print("Wrote log:", log_path)
    print("Wrote incremental per-video log:", per_video_log_path)


if __name__ == "__main__":
    main()
