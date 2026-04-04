import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from easycache_cogvideo import easycache_baseline_forward


def main():
    prompt = os.environ.get(
        "COG_PROMPT",
        "A panda, dressed in a small red jacket and tiny hat, plays a guitar in a bamboo forest.",
    )
    neg_prompt = os.environ.get("COG_NEG_PROMPT", "")
    model_id = os.environ.get("COG_MODEL", "THUDM/CogVideoX-2b")
    steps = int(os.environ.get("COG_STEPS", "50"))
    guidance = float(os.environ.get("COG_GUIDANCE", "6.0"))
    seed = int(os.environ.get("COG_SEED", "12345"))

    out_root = os.environ.get(
        "COG_PROF_DIR",
        "/nfs/oagrawal/CogVideo/vbench_eval_easycache/profiling",
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "baseline_profile"
    run_dir = os.path.join(out_root, f"{tag}_{ts}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    transformer = pipe.transformer
    cls = transformer.__class__
    if not hasattr(cls, "_original_forward"):
        cls._original_forward = cls.forward

    # reset + install baseline forward wrapper
    cls.forward = easycache_baseline_forward
    cls.cnt = 0
    cls.num_steps = steps
    cls.total_time = 0.0
    cls.k = None
    cls.previous_raw_input = None
    cls.previous_output = None
    cls.prev_prev_raw_input = None
    cls.k_history = []
    cls.pred_change_history = []

    start = time.time()
    # Use numpy output so export_to_video receives ndarray frames
    video = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt or None,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
        output_type="np",
    ).frames[0]
    e2e = time.time() - start

    export_to_video(video, os.path.join(run_dir, "video.mp4"), fps=8)

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
                "e2e_seconds": e2e,
                "transformer_seconds": float(getattr(transformer, "total_time", 0.0)),
                "k_history": cls.k_history,
                "pred_change_history": cls.pred_change_history,
            },
            f,
            indent=2,
        )

    # Plot k_t
    if cls.k_history:
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(cls.k_history)), cls.k_history, marker="o", linewidth=1)
        plt.title("CogVideoX EasyCache baseline: k_t history")
        plt.xlabel("step index (late steps only if k starts later)")
        plt.ylabel("k_t")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "k_t_plot.png"), dpi=200)
        plt.close()

    # Plot pred_change
    if cls.pred_change_history:
        plt.figure(figsize=(10, 4))
        plt.plot(
            range(len(cls.pred_change_history)),
            cls.pred_change_history,
            marker="s",
            linewidth=1,
        )
        plt.title("CogVideoX EasyCache baseline: pred_change history")
        plt.xlabel("step index (starts after k becomes available)")
        plt.ylabel("pred_change")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "pred_change_plot.png"), dpi=200)
        plt.close()

    print("Saved baseline profiling to:", run_dir)
    print(
        "e2e_seconds:",
        round(e2e, 2),
        "transformer_seconds:",
        round(float(transformer.total_time), 2),
    )


if __name__ == "__main__":
    main()

