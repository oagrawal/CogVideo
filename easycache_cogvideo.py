import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


@dataclass
class EasyCacheConfig:
    mode: str = "easycache"  # "baseline" | "easycache"
    thresh: float = 0.05
    ret_steps: int = 5


def _extract_sample_from_output(output, return_dict: bool):
    """
    CogVideoXTransformer3DModel returns either:
      - tuple(sample,) when return_dict == False
      - dict-like with key 'sample' when return_dict == True
    We normalize to a tensor here and then re-wrap on return.
    """
    if not return_dict:
        # expected (sample,)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
    # dict-like
    if isinstance(output, dict):
        return output.get("sample", next(iter(output.values())))
    # fallback
    return output


def _wrap_sample_for_output(sample: torch.Tensor, template_output, return_dict: bool):
    if not return_dict:
        return (sample,)
    if isinstance(template_output, dict):
        new_out = dict(template_output)
        # standard diffusers key
        if "sample" in new_out:
            new_out["sample"] = sample
        else:
            # best-effort: overwrite first tensor entry
            first_key = next(iter(new_out.keys()))
            new_out[first_key] = sample
        return new_out
    return sample


def easycache_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_kwargs: Dict[str, Any] | None = None,
    return_dict: bool = True,
):
    """
    Generic EasyCache wrapper around CogVideoXTransformer3DModel.forward.

    It:
    - Tracks raw input and output between steps.
    - Learns a per-step linear factor k such that:
          ||v_t - v_{t-1}|| ≈ k_t * ||x_t - x_{t-1}||
    - Uses k_t to predict the relative output change and accumulates it.
    - Skips the expensive model forward while accumulated predicted change < thresh,
      reusing a cached residual instead.

    This implementation is model-agnostic: it delegates the actual computation to
    `self._original_forward` so it does not depend on internal block structure.
    """
    torch.cuda.synchronize()
    start_time = time.time()

    # Ensure we have the original forward saved
    if not hasattr(self, "_original_forward"):
        raise RuntimeError("EasyCache forward called before _original_forward was set.")

    raw_input = hidden_states

    # Adaptive threshold: use low/high thresholds over step ranges if configured,
    # otherwise fall back to the single fixed self.thresh.
    if getattr(self, "thresh_low", None) is not None and getattr(self, "thresh_high", None) is not None:
        i = int(self.cnt)
        steps = int(self.num_steps)
        if i < int(getattr(self, "first_steps", 0)) or i >= steps - int(getattr(self, "last_steps", 0)):
            thresh = float(self.thresh_low)
        else:
            thresh = float(self.thresh_high)
    else:
        thresh = float(self.thresh)

    # Decide compute vs skip
    if self.cnt < self.ret_steps or self.cnt >= self.num_steps - 1:
        should_calc = True
        self.accumulated_error = 0.0
    else:
        if (
            self.previous_raw_input is not None
            and self.previous_output is not None
            and self.k is not None
        ):
            raw_input_change = (raw_input - self.previous_raw_input).abs().mean()
            output_norm = self.previous_output.abs().mean()
            # avoid div by zero
            if output_norm > 0:
                pred_change = (self.k * (raw_input_change / output_norm)).item()
            else:
                pred_change = float("inf")

            self.accumulated_error += pred_change

            if self.accumulated_error < thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_error = 0.0
        else:
            should_calc = True

    self.previous_raw_input = raw_input.detach()

    # Skip path: reuse cached residual
    if (not should_calc) and (self.cache is not None):
        result = raw_input + self.cache
        self.cnt += 1

        torch.cuda.synchronize()
        end_time = time.time()
        self.total_time += (end_time - start_time)

        # Re-wrap result to match original output structure
        # We don't have a template_output here in skip mode, so just mimic return_dict
        if return_dict:
            return {"sample": result}
        return (result,)

    # Full compute path
    original_output = self._original_forward(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=attention_kwargs,
        return_dict=return_dict,
    )
    result = _extract_sample_from_output(original_output, return_dict)

    # Update EasyCache state
    if self.previous_output is not None and self.prev_prev_raw_input is not None:
        output_change = (result - self.previous_output).abs().mean()
        input_change = (self.previous_raw_input - self.prev_prev_raw_input).abs().mean()
        if input_change > 0:
            self.k = output_change / input_change

    self.cache = result - raw_input
    self.prev_prev_raw_input = self.previous_raw_input
    self.previous_output = result.detach()

    self.cnt += 1

    torch.cuda.synchronize()
    end_time = time.time()
    self.total_time += (end_time - start_time)

    return _wrap_sample_for_output(result, original_output, return_dict)


def easycache_baseline_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_kwargs: Dict[str, Any] | None = None,
    return_dict: bool = True,
):
    """
    Baseline (no skipping) wrapper that:
    - Always calls the original forward.
    - Learns k_t over time and can be used to profile per-step change statistics
      to tune EasyCache thresholds.
    """
    torch.cuda.synchronize()
    start_time = time.time()

    if not hasattr(self, "_original_forward"):
        raise RuntimeError("EasyCache baseline forward called before _original_forward was set.")

    raw_input = hidden_states

    if (
        self.previous_raw_input is not None
        and self.previous_output is not None
        and self.k is not None
    ):
        raw_input_change = (raw_input - self.previous_raw_input).abs().mean()
        output_norm = self.previous_output.abs().mean()
        if output_norm > 0:
            pred_change = (self.k * (raw_input_change / output_norm)).item()
            self.pred_change_history.append(pred_change)

    self.previous_raw_input = raw_input.detach()

    original_output = self._original_forward(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=attention_kwargs,
        return_dict=return_dict,
    )
    result = _extract_sample_from_output(original_output, return_dict)

    if self.previous_output is not None and self.prev_prev_raw_input is not None:
        output_change = (result - self.previous_output).abs().mean()
        input_change = (self.previous_raw_input - self.prev_prev_raw_input).abs().mean()
        if input_change > 0:
            self.k = output_change / input_change
            self.k_history.append(self.k.item())

    self.prev_prev_raw_input = self.previous_raw_input
    self.previous_output = result.detach()

    self.cnt += 1

    torch.cuda.synchronize()
    end_time = time.time()
    self.total_time += (end_time - start_time)

    return _wrap_sample_for_output(result, original_output, return_dict)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="EasyCache wrapper for CogVideoX text-to-video generation.",
    )

    # Core CogVideoX args
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="THUDM/CogVideoX-2b",
        help="Model identifier for CogVideoXPipeline.from_pretrained.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output video frame height. Defaults to model config if None.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output video frame width. Defaults to model config if None.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames. Defaults to model config if None.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Denoising steps.",
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
        default=0,
        help="Random seed (<= 0 means random).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cogvideox_easycache.mp4",
        help="Output video path.",
    )

    # EasyCache args
    parser.add_argument(
        "--easycache-mode",
        type=str,
        default="easycache",
        choices=["baseline", "easycache"],
        help="baseline: no skipping, just profile; easycache: enable skipping.",
    )
    parser.add_argument(
        "--easycache-thresh",
        type=float,
        default=0.05,
        help="Accumulated predicted-change threshold for skipping.",
    )
    parser.add_argument(
        "--easycache-ret-steps",
        type=int,
        default=5,
        help="Number of initial steps that always compute.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = None
    if args.seed > 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    pipe = CogVideoXPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    # Save original transformer forward and patch EasyCache wrappers
    transformer = pipe.transformer
    transformer_cls = transformer.__class__

    if not hasattr(transformer_cls, "_original_forward"):
        transformer_cls._original_forward = transformer_cls.forward

    # Shared state on the class (mirroring Hunyuan / Wan patterns)
    transformer_cls.cnt = 0
    transformer_cls.num_steps = args.num_inference_steps
    transformer_cls.total_time = 0.0
    transformer_cls.k = None
    transformer_cls.previous_raw_input = None
    transformer_cls.previous_output = None
    transformer_cls.prev_prev_raw_input = None
    transformer_cls.k_history = []
    transformer_cls.pred_change_history = []
    transformer_cls.cache = None
    transformer_cls.accumulated_error = 0.0
    transformer_cls.ret_steps = int(args.easycache_ret_steps)
    transformer_cls.thresh = float(args.easycache_thresh)

    if args.easycache_mode == "easycache":
        transformer_cls.forward = easycache_forward
        mode_tag = f"easycache_thr{args.easycache_thresh}"
    else:
        transformer_cls.forward = easycache_baseline_forward
        mode_tag = "baseline_profile"

    # Generation
    start_time = time.time()
    video_out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt if args.negative_prompt else None,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
    ).frames[0]
    end_time = time.time()
    e2e_time = end_time - start_time

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_to_video(video_out, args.output, fps=8)

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{formatted_time}] CogVideoX EasyCache mode={mode_tag}, "
        f"steps={args.num_inference_steps}, "
        f"e2e_time={e2e_time:.2f}s, "
        f"transformer_time={transformer.total_time:.2f}s, "
        f"saved_to={args.output}"
    )


if __name__ == "__main__":
    main()

