import json
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from easycache_cogvideo import easycache_baseline_forward


def main():
    prompt = "A cat running on grass."
    neg_prompt = ""
    model_id = "THUDM/CogVideoX1.5-5B"
    steps = 50
    guidance = 6.0
    seed = 0
    height = 768
    width = 1360
    num_frames = 81
    fps = 16

    out_root = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "baseline_profile"
    run_dir = os.path.join(out_root, f"{tag}_{ts}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # enable memory optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    transformer = pipe.transformer
    cls = transformer.__class__
    if not hasattr(cls, "_original_forward"):
        cls._original_forward = cls.forward

    # reset + install baseline forward wrapper
    cls.forward = easycache_baseline_forward
    # Resetting state on INSTANCE to avoid side effects and shadow class variables if any
    transformer.cnt = 0
    transformer.num_steps = steps
    transformer.total_time = 0.0
    transformer.k = None
    transformer.previous_raw_input = None
    transformer.previous_output = None
    transformer.prev_prev_raw_input = None
    transformer.k_history = []
    transformer.pred_change_history = []
    transformer.cache = None

    start = time.time()
    # Use numpy output so export_to_video receives ndarray frames
    video = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt or None,
        height=height,
        width=width,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
        output_type="np",
    ).frames[0]
    e2e = time.time() - start

    export_to_video(video, os.path.join(run_dir, "video.mp4"), fps=fps)

    k_history = getattr(transformer, "k_history", [])
    pred_change_history = getattr(transformer, "pred_change_history", [])

    # Save logs
    with open(os.path.join(run_dir, "profiling.json"), "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "seed": seed,
                "steps": steps,
                "guidance_scale": guidance,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "fps": fps,
                "e2e_seconds": e2e,
                "transformer_seconds": float(getattr(transformer, "total_time", 0.0)),
                "k_history": k_history,
                "pred_change_history": pred_change_history,
            },
            f,
            indent=2,
        )

    # Plot k_t
    if k_history:
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(k_history)), k_history, marker="o", linewidth=1)
        plt.title(f"CogVideoX1.5-5B EasyCache baseline: k_t history")
        plt.xlabel("step index (late steps only if k starts later)")
        plt.ylabel("k_t")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "k_t_plot.png"), dpi=200)
        plt.close()

    # Plot pred_change
    if pred_change_history:
        plt.figure(figsize=(10, 4))
        plt.plot(
            range(len(pred_change_history)),
            pred_change_history,
            marker="s",
            linewidth=1,
        )
        plt.title(f"CogVideoX1.5-5B EasyCache baseline: pred_change history")
        plt.xlabel("step index (starts after k becomes available)")
        plt.ylabel("pred_change")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "pred_change_plot.png"), dpi=200)
        plt.close()

    print(f"Saved baseline profiling to: {run_dir}")
    print(
        f"e2e_seconds: {round(e2e, 2)} transformer_seconds: {round(float(transformer.total_time), 2)}"
    )

if __name__ == "__main__":
    main()
