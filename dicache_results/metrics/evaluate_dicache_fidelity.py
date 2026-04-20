import argparse
import json
import os
import tqdm
import torch
import numpy as np
import imageio
from pathlib import Path
from calculate_lpips import calculate_lpips
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim


def load_video(video_path, device="cuda"):
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for frame in reader:
        frame_tensor = torch.tensor(frame).to(device).permute(2, 0, 1)
        frames.append(frame_tensor)
    return torch.stack(frames)


def evaluate_mode(mode_name, gt_dir, gen_dir, output_file, device="cuda"):
    video_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(".mp4")])
    print(f"Evaluating {mode_name}: {len(video_files)} videos found.")

    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": []
    }

    for f in tqdm.tqdm(video_files):
        gen_path = os.path.join(gen_dir, f)
        gt_path = os.path.join(gt_dir, f)

        if not os.path.exists(gt_path):
            print(f"Warning: Baseline video {f} not found. Skipping.")
            continue

        gen_video = (load_video(gen_path, device="cpu") / 255.0).unsqueeze(0) # (1, T, C, H, W)
        gt_video = (load_video(gt_path, device="cpu") / 255.0).unsqueeze(0)   # (1, T, C, H, W)

        # Truncate to same length if needed (should be 81 frames)
        min_t = min(gen_video.shape[1], gt_video.shape[1])
        gen_video = gen_video[:, :min_t]
        gt_video = gt_video[:, :min_t]

        # Calculate metrics (per-video mean)
        # calculate_psnr etc expect (B, T, C, H, W)
        p_res = calculate_psnr(gt_video, gen_video)["value"]
        s_res = calculate_ssim(gt_video, gen_video)["value"]
        l_res = calculate_lpips(gt_video, gen_video, device=device)["value"]

        metrics["psnr"].append(float(np.mean(list(p_res.values()))))
        metrics["ssim"].append(float(np.mean(list(s_res.values()))))
        metrics["lpips"].append(float(np.mean(list(l_res.values()))))

    res = {
        "mode": mode_name,
        "baseline": "cog_dc_baseline",
        "num_videos": len(metrics["psnr"]),
        "psnr": {
            "mean": float(np.mean(metrics["psnr"])),
            "std": float(np.std(metrics["psnr"]))
        },
        "ssim": {
            "mean": float(np.mean(metrics["ssim"])),
            "std": float(np.std(metrics["ssim"]))
        },
        "lpips": {
            "mean": float(np.mean(metrics["lpips"])),
            "std": float(np.std(metrics["lpips"]))
        }
    }

    with open(output_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    evaluate_mode(args.mode, args.gt_dir, args.gen_dir, args.output)
