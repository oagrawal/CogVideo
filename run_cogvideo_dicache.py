"""
DiCache implementation for CogVideoX (CogVideoX1.5-5B / CogVideoX-5b).

DiCache uses shallow "probe" layers at every step to measure how much the model's
internal representations are changing. It accumulates the relative L1 change of the
probe output. When the accumulation stays below a threshold, the expensive deep layers
are skipped and the output is approximated using a trajectory-aligned extrapolation of
previously cached residuals. When the accumulation exceeds the threshold, a full forward
pass runs (optionally resuming from the already-computed probe state) and the cache is
refreshed.

Adaptive variant: use thresh_low in volatile regions (early/late steps) and thresh_high
in the stable middle region, controlled by --stable-start / --stable-end.

Usage (single-prompt calibration):
    python3 run_cogvideo_dicache.py --calibrate --seed 0 \\
        --calibrate-save-path ./dicache_results/calibration/cog_dc_probe_curve_seed0

Usage (single-prompt, fixed threshold):
    python3 run_cogvideo_dicache.py \\
        --prompt "A cat running on grass." --rel-l1-thresh 0.10 \\
        --output-dir ./dicache_results/videos/test

Usage (baseline, no caching):
    python3 run_cogvideo_dicache.py --baseline \\
        --output-dir ./dicache_results/videos/baseline_test
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusers import CogVideoXPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    export_to_video,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils import logging as diffusers_logging

logger = diffusers_logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# DiCache forward (patched onto CogVideoXTransformer3DModel)
# ---------------------------------------------------------------------------

def dicache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    ofs: Optional[Union[int, float, torch.LongTensor]] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    # ---- LoRA scale (unchanged from original) ----
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_frames, channels, height, width = hidden_states.shape

    # ---- 1. Time embedding ----
    timesteps = timestep
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = self.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    # ---- 2. Patch embedding ----
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    # ---- DiCache: determine skip threshold ----
    if getattr(self, "adaptive", False):
        stable_start = getattr(self, "stable_start", 8)
        stable_end = getattr(self, "stable_end", 40)
        if self.cnt < stable_start or self.cnt >= stable_end:
            thresh = float(self.thresh_low)
        else:
            thresh = float(self.thresh_high)
    else:
        thresh = float(self.rel_l1_thresh)

    # ---- DiCache: probe + skip decision ----
    skip_forward = False
    probe_ran = False

    ori_hidden_states = hidden_states
    ori_encoder_hidden_states = encoder_hidden_states

    calibrate_mode = getattr(self, "calibrate", False)

    # Probe only when previous probe state is available (warmup guard)
    if self.previous_probe_hs is not None:
        probe_ran = True
        test_hs = hidden_states.clone()
        test_ehs = encoder_hidden_states.clone()

        # Run probe blocks (shallow prefix of transformer_blocks)
        probe_blocks = self.transformer_blocks[: self.probe_depth]
        for block in probe_blocks:
            test_hs, test_ehs = block(
                hidden_states=test_hs,
                encoder_hidden_states=test_ehs,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        # Relative L1 probe change (hidden_states only — primary signal)
        delta_y = (
            (test_hs - self.previous_probe_hs).abs().mean()
            / self.previous_probe_hs.abs().mean()
        )
        self.accumulated_rel_l1_distance += delta_y.item()

        # Record calibration curve if requested
        if calibrate_mode and hasattr(self, "calibration_delta_y"):
            self.calibration_delta_y.append(float(delta_y.item()))

        if not calibrate_mode and self.accumulated_rel_l1_distance < thresh:
            skip_forward = True
            self.resume_flag = False
        else:
            self.accumulated_rel_l1_distance = 0.0
            self.resume_flag = True

    # ---- DiCache: skip branch (trajectory-aligned approximation) ----
    if skip_forward:
        ori_hidden_states = hidden_states.clone()

        if len(self.residual_window_hs) >= 2:
            # Probe-trajectory gamma: how much the probe residual changed vs last step
            current_probe_res = test_hs - hidden_states
            denom = (
                self.probe_residual_window[-1] - self.probe_residual_window[-2]
            ).abs().mean()
            if denom > 1e-8:
                gamma = (
                    (current_probe_res - self.probe_residual_window[-2]).abs().mean() / denom
                ).clip(1.0, 2.0)
            else:
                gamma = torch.tensor(1.0, device=hidden_states.device)
            hidden_states = (
                hidden_states
                + self.residual_window_hs[-2]
                + gamma * (self.residual_window_hs[-1] - self.residual_window_hs[-2])
            )
        else:
            hidden_states = hidden_states + self.residual_cache_hs

        # Text tokens: simple cached residual (they change slowly across steps)
        encoder_hidden_states = encoder_hidden_states + self.residual_cache_ehs

        # Update probe state for next step's signal computation
        self.previous_probe_hs = test_hs

    # ---- DiCache: full-run branch ----
    else:
        ori_hidden_states = hidden_states
        ori_encoder_hidden_states_for_residual = encoder_hidden_states

        if probe_ran and self.resume_flag:
            # Resume from already-computed probe output (skip first probe_depth blocks)
            hidden_states = test_hs
            encoder_hidden_states = test_ehs
            unpass_blocks = self.transformer_blocks[self.probe_depth :]
        else:
            unpass_blocks = self.transformer_blocks

        for ind, block in enumerate(unpass_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            # Capture probe-depth output for the next step's warmup/signal
            # ind is relative to unpass_blocks; when resuming, probe blocks are skipped,
            # so ind=-1 means we never hit probe_depth-1 again — use test_hs directly.
            if not (probe_ran and self.resume_flag):
                # Running from block 0: capture at probe_depth-1
                if ind == self.probe_depth - 1:
                    if probe_ran:
                        self.previous_probe_hs = test_hs
                    else:
                        self.previous_probe_hs = hidden_states.detach().clone()

        # If resumed from probe, probe output was already set on probe blocks
        if probe_ran and self.resume_flag:
            self.previous_probe_hs = test_hs

        # Compute and store residuals
        residual_hs = hidden_states - ori_hidden_states
        residual_ehs = encoder_hidden_states - ori_encoder_hidden_states_for_residual
        probe_res = self.previous_probe_hs - ori_hidden_states

        self.residual_cache_hs = residual_hs
        self.residual_cache_ehs = residual_ehs
        self.probe_residual_cache = probe_res

        # Sliding window update (keep last 2 entries)
        if len(self.residual_window_hs) < 2:
            self.residual_window_hs.append(residual_hs)
            self.probe_residual_window.append(probe_res)
        else:
            self.residual_window_hs[-2] = self.residual_window_hs[-1]
            self.residual_window_hs[-1] = residual_hs
            self.probe_residual_window[-2] = self.probe_residual_window[-1]
            self.probe_residual_window[-1] = probe_res

    # ---- 3. Final norm (model-variant dependent) ----
    if not self.config.use_rotary_positional_embeddings:
        # CogVideoX-2B
        hidden_states = self.norm_final(hidden_states)
    else:
        # CogVideoX-5B and CogVideoX1.5-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    # ---- 4. Final block ----
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # ---- 5. Unpatchify ----
    p = self.config.patch_size
    p_t = self.config.patch_size_t

    if p_t is None:
        output = hidden_states.reshape(
            batch_size, num_frames, height // p, width // p, -1, p, p
        )
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hidden_states.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    # ---- Increment counter; reset state at video boundary ----
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.resume_flag = False
        self.previous_probe_hs = None
        self.residual_cache_hs = None
        self.residual_cache_ehs = None
        self.probe_residual_cache = None
        self.residual_window_hs = []
        self.probe_residual_window = []

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Probe-curve plotting helper
# ---------------------------------------------------------------------------

def _save_probe_curve(delta_y_list: List[float], save_path_base: str) -> None:
    """Save probe curve as JSON and PNG."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path_base)), exist_ok=True)

    json_path = save_path_base + ".json"
    with open(json_path, "w") as f:
        json.dump({"delta_y": delta_y_list}, f, indent=2)
    print(f"Probe curve JSON saved to: {json_path}")

    png_path = save_path_base + ".png"
    plt.figure(figsize=(10, 5))
    x = range(len(delta_y_list))
    plt.plot(x, delta_y_list, "b-", linewidth=1.5, marker="o", markersize=3)
    plt.xlabel("Denoising step (cnt)")
    plt.ylabel("Probe rel-L1 delta_y")
    plt.title("DiCache probe curve (CogVideoX1.5-5B)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Probe curve PNG saved to: {png_path}")


# ---------------------------------------------------------------------------
# Single-video generation function (reusable by batch harness)
# ---------------------------------------------------------------------------

def run_generation(
    prompt: str,
    seed: int,
    rel_l1_thresh: float,
    save_file: str,
    ckpts_path: str = "THUDM/CogVideoX1.5-5B",
    num_inference_steps: int = 50,
    height: int = 768,
    width: int = 1360,
    num_frames: int = 81,
    guidance_scale: float = 6.0,
    fps: int = 16,
    probe_depth: int = 1,
    ret_ratio: float = 0.0,
    pipe=None,
    # calibration
    calibrate: bool = False,
    calibrate_save_path: Optional[str] = None,
    # baseline (no patching at all)
    baseline: bool = False,
    # adaptive threshold
    adaptive: bool = False,
    thresh_low: float = 0.05,
    thresh_high: float = 0.20,
    stable_start: int = 8,
    stable_end: int = 40,
) -> object:
    """
    Generate one video. Returns the pipeline object for reuse across prompts.

    State is always reset on the transformer INSTANCE so that subsequent calls
    with a different mode or prompt start clean (instance attributes shadow the
    class attributes set by earlier runs).
    """
    if pipe is None:
        pipe = CogVideoXPipeline.from_pretrained(ckpts_path, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    transformer = pipe.transformer
    cls = transformer.__class__

    # Save original forward once (never overwrite after first save)
    if not hasattr(cls, "_original_forward"):
        cls._original_forward = cls.forward

    if baseline:
        # Restore original forward, no DiCache state needed
        cls.forward = cls._original_forward
    else:
        cls.forward = dicache_forward

    # Reset all DiCache state on the INSTANCE (important: not the class,
    # so that instance attributes from prior runs don't linger as stale class attrs)
    transformer.cnt = 0
    transformer.probe_depth = probe_depth
    transformer.num_steps = num_inference_steps
    transformer.rel_l1_thresh = rel_l1_thresh
    transformer.ret_ratio = ret_ratio
    transformer.accumulated_rel_l1_distance = 0.0
    transformer.resume_flag = False
    transformer.previous_probe_hs = None
    transformer.residual_cache_hs = None
    transformer.residual_cache_ehs = None
    transformer.probe_residual_cache = None
    transformer.residual_window_hs = []
    transformer.probe_residual_window = []
    # adaptive
    transformer.adaptive = adaptive
    transformer.thresh_low = thresh_low
    transformer.thresh_high = thresh_high
    transformer.stable_start = stable_start
    transformer.stable_end = stable_end
    # calibration
    transformer.calibrate = calibrate
    if calibrate:
        transformer.calibration_delta_y = []

    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)

    video = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).frames[0]

    export_to_video(video, save_file, fps=fps)
    print(f"Saved: {save_file}")

    if calibrate and calibrate_save_path is not None:
        delta_y_list = getattr(transformer, "calibration_delta_y", [])
        _save_probe_curve(delta_y_list, calibrate_save_path)

    return pipe


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CogVideoX DiCache single-prompt generation / calibration."
    )
    parser.add_argument("--prompt", type=str,
        default="Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
                "intensely on a spotlighted stage.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpts-path", type=str, default="THUDM/CogVideoX1.5-5B")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="./dicache_results/videos/single_test")
    # DiCache core params
    parser.add_argument("--rel-l1-thresh", type=float, default=0.10,
        help="Accumulated relative-L1 threshold for skipping (fixed mode).")
    parser.add_argument("--probe-depth", type=int, default=1,
        help="Number of transformer blocks used as the shallow probe.")
    parser.add_argument("--ret-ratio", type=float, default=0.0,
        help="Fraction of initial steps that always run fully (deprecated; warmup handled by None guard).")
    # Modes
    parser.add_argument("--baseline", action="store_true",
        help="Baseline mode: no DiCache patching, full forward every step.")
    parser.add_argument("--calibrate", action="store_true",
        help="Calibration mode: full forward every step, record probe delta_y curve.")
    parser.add_argument("--calibrate-save-path", type=str,
        default="./dicache_results/calibration/cog_dc_probe_curve_seed0",
        help="Base path (no extension) to save calibration JSON and PNG.")
    # Adaptive threshold
    parser.add_argument("--adaptive", action="store_true",
        help="Use adaptive (low/high) thresholds instead of fixed rel_l1_thresh.")
    parser.add_argument("--thresh-low", type=float, default=0.05,
        help="Low threshold for volatile regions (adaptive mode).")
    parser.add_argument("--thresh-high", type=float, default=0.20,
        help="High threshold for stable middle region (adaptive mode).")
    parser.add_argument("--stable-start", type=int, default=8,
        help="First step of the stable middle region (0-indexed, inclusive).")
    parser.add_argument("--stable-end", type=int, default=40,
        help="First step of the late volatile region (0-indexed, exclusive).")
    return parser.parse_args()


def main():
    args = _parse_args()

    mode_tag = "baseline" if args.baseline else (
        "calibrate" if args.calibrate else (
            f"adaptive_hi{args.thresh_high}_lo{args.thresh_low}" if args.adaptive
            else f"fixed_{args.rel_l1_thresh}"
        )
    )
    save_file = os.path.join(args.output_dir, f"cog_dc_{mode_tag}_seed{args.seed}.mp4")

    t0 = time.time()
    run_generation(
        prompt=args.prompt,
        seed=args.seed,
        rel_l1_thresh=args.rel_l1_thresh,
        save_file=save_file,
        ckpts_path=args.ckpts_path,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        probe_depth=args.probe_depth,
        ret_ratio=args.ret_ratio,
        calibrate=args.calibrate,
        calibrate_save_path=args.calibrate_save_path if args.calibrate else None,
        baseline=args.baseline,
        adaptive=args.adaptive,
        thresh_low=args.thresh_low,
        thresh_high=args.thresh_high,
        stable_start=args.stable_start,
        stable_end=args.stable_end,
    )
    elapsed = time.time() - t0
    print(f"[{mode_tag}] Done in {elapsed:.1f}s  ->  {save_file}")


if __name__ == "__main__":
    main()
