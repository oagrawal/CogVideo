import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from easycache_cogvideo import easycache_forward, easycache_baseline_forward


def _load_prompts(prompts_path: Path) -> List[Dict]:
    with open(prompts_path, "r") as f:
        data = json.load(f)
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generation for CogVideoX + EasyCache (baseline + fixed thresholds).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="CogVideoX model id or local path.",
    )
    parser.add_argument(
        "--prompts-json",
        type=str,
        default="/nfs/oagrawal/HunyuanVideo/vbench_eval/prompts_subset.json",
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
        help="Number of diffusion steps (num_inference_steps).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed for all videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/nfs/oagrawal/CogVideo/vbench_eval_easycache/videos",
        help="Root directory where mode subfolders and videos are saved.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="cog_ec_baseline,cog_ec_fixed_0.035,cog_ec_fixed_0.05,cog_ec_fixed_0.08",
        help=(
            "Comma-separated list of modes. "
            "Supported names: cog_ec_baseline, cog_ec_fixed_0.035, cog_ec_fixed_0.05, cog_ec_fixed_0.08, "
            "cog_ec_fixed_0.10, cog_ec_fixed_0.12, cog_ec_fixed_0.15, "
            "cog_ec_adapt_lo035_hi080, cog_ec_adapt_lo050_hi080, cog_ec_adapt_lo050_hi100, "
            "cog_ec_adapt_f8l8_lo035_hi010, cog_ec_adapt_f8l8_lo050_hi010, cog_ec_adapt_f8l8_lo050_hi012"
        ),
    )
    return parser.parse_args()


def _mode_config(mode_name: str) -> Dict:
    """
    Map mode names to EasyCache configuration.
    """
    if mode_name == "cog_ec_baseline":
        return {"mode": "baseline"}
    if mode_name == "cog_ec_fixed_0.035":
        return {"mode": "easycache", "thresh": 0.035, "ret_steps": 1}
    if mode_name == "cog_ec_fixed_0.05":
        return {"mode": "easycache", "thresh": 0.05, "ret_steps": 1}
    if mode_name == "cog_ec_fixed_0.08":
        return {"mode": "easycache", "thresh": 0.08, "ret_steps": 1}
    if mode_name == "cog_ec_fixed_0.10":
        return {"mode": "easycache", "thresh": 0.10, "ret_steps": 1}
    if mode_name == "cog_ec_fixed_0.12":
        return {"mode": "easycache", "thresh": 0.12, "ret_steps": 1}
    if mode_name == "cog_ec_fixed_0.15":
        return {"mode": "easycache", "thresh": 0.15, "ret_steps": 1}
    if mode_name == "cog_ec_adapt_lo035_hi080":
        return {
            "mode": "adaptive",
            "ret_steps": 3,
            "thresh_low": 0.035,
            "thresh_high": 0.08,
            "first_steps": 10,
            "last_steps": 5,
        }
    if mode_name == "cog_ec_adapt_lo050_hi080":
        return {
            "mode": "adaptive",
            "ret_steps": 2,
            "thresh_low": 0.05,
            "thresh_high": 0.08,
            "first_steps": 8,
            "last_steps": 5,
        }
    if mode_name == "cog_ec_adapt_lo050_hi100":
        return {
            "mode": "adaptive",
            "ret_steps": 1,
            "thresh_low": 0.05,
            "thresh_high": 0.10,
            "first_steps": 8,
            "last_steps": 5,
        }
    if mode_name == "cog_ec_adapt_f8l8_lo035_hi010":
        return {
            "mode": "adaptive",
            "ret_steps": 1,
            "thresh_low": 0.035,
            "thresh_high": 0.10,
            "first_steps": 8,
            "last_steps": 8,
        }
    if mode_name == "cog_ec_adapt_f8l8_lo050_hi010":
        return {
            "mode": "adaptive",
            "ret_steps": 1,
            "thresh_low": 0.05,
            "thresh_high": 0.10,
            "first_steps": 8,
            "last_steps": 8,
        }
    if mode_name == "cog_ec_adapt_f8l8_lo050_hi012":
        return {
            "mode": "adaptive",
            "ret_steps": 1,
            "thresh_low": 0.05,
            "thresh_high": 0.12,
            "first_steps": 8,
            "last_steps": 8,
        }
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
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

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

    for local_i, p in enumerate(prompts_slice):
        prompt_idx = args.start_idx + local_i
        prompt_text = p["prompt"] if isinstance(p, dict) and "prompt" in p else str(p)

        for mode_name in mode_names:
            cfg = _mode_config(mode_name)

            mode_dir = out_root / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)
            out_path = mode_dir / f"{prompt_idx:03d}-seed{args.seed}.mp4"

            if out_path.exists():
                continue

            # Reset EasyCache / baseline state for each (prompt, mode) run.
            # IMPORTANT: reset on the transformer INSTANCE, not the class.
            # After the first run, self.X += 1 inside the forward creates instance
            # attributes that shadow cls.X, so cls.X = ... has no effect on subsequent runs.
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
            transformer.ret_steps = int(cfg.get("ret_steps", 1))
            transformer.thresh = float(cfg.get("thresh", 0.05)) if cfg["mode"] == "easycache" else 0.0
            # Adaptive params (None when not adaptive; easycache_forward checks with getattr)
            transformer.thresh_low = float(cfg["thresh_low"]) if "thresh_low" in cfg else None
            transformer.thresh_high = float(cfg["thresh_high"]) if "thresh_high" in cfg else None
            transformer.first_steps = int(cfg.get("first_steps", 0))
            transformer.last_steps = int(cfg.get("last_steps", 0))

            if cfg["mode"] == "baseline":
                cls.forward = easycache_baseline_forward
            else:
                # both "easycache" and "adaptive" use easycache_forward;
                # adaptive is distinguished by thresh_low/thresh_high being set
                cls.forward = easycache_forward

            g = torch.Generator(device=device).manual_seed(args.seed)

            t0 = time.time()
            video = pipe(
                prompt=prompt_text,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=g,
                output_type="np",
            ).frames[0]
            e2e = time.time() - t0
            # total_time was reset to 0.0 above so this is the per-run DiT time
            dit = float(transformer.total_time)

            export_to_video(video, str(out_path), fps=8)

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
            print(
                "DONE",
                mode_name,
                "prompt",
                prompt_idx,
                "e2e",
                round(e2e, 2),
                "dit",
                round(dit, 2),
                flush=True,
            )

    log_path = out_root.parent / f"generation_log_gpu{log['gpu']}_p{args.start_idx:02d}-{args.end_idx:02d}_steps{args.steps}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print("Wrote log:", log_path)


if __name__ == "__main__":
    main()

