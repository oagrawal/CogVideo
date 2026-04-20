import argparse
import os
import json
import numpy as np
import imageio
import torch
import torchvision.transforms.functional as F
import tqdm
from calculate_lpips import calculate_lpips
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim


def load_video(video_path):
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []
    for frame in reader:
        frame_tensor = torch.tensor(frame).cuda().permute(2, 0, 1)
        frames.append(frame_tensor)
    video_tensor = torch.stack(frames)
    return video_tensor


def resize_video(video, target_height, target_width):
    resized_frames = []
    for frame in video:
        resized_frame = F.resize(frame, [target_height, target_width])
        resized_frames.append(resized_frame)
    return torch.stack(resized_frames)


def preprocess_eval_video(eval_video, generated_video_shape):
    T_gen, _, H_gen, W_gen = generated_video_shape
    T_eval, _, H_eval, W_eval = eval_video.shape

    if T_eval < T_gen:
        # Instead of error, we can truncate the gen video later, or truncate eval here if T_eval > T_gen
        pass

    if H_eval < H_gen or W_eval < W_gen:
        resize_height = max(H_gen, int(H_gen * (H_eval / W_eval)))
        resize_width = max(W_gen, int(W_gen * (W_eval / H_eval)))
        eval_video = resize_video(eval_video, resize_height, resize_width)
        T_eval, _, H_eval, W_eval = eval_video.shape

    start_h = (H_eval - H_gen) // 2
    start_w = (W_eval - W_gen) // 2
    min_t = min(T_gen, T_eval)
    cropped_video = eval_video[:min_t, :, start_h : start_h + H_gen, start_w : start_w + W_gen]

    return cropped_video


def main(args):
    device = "cuda"
    gt_video_dir = args.gt_video_dir
    generated_video_dir = args.generated_video_dir

    video_ids = []
    file_extension = "mp4"
    for f in os.listdir(generated_video_dir):
        if f.endswith(f".{file_extension}"):
            video_ids.append(f.replace(f".{file_extension}", ""))
    video_ids.sort()

    if not video_ids:
        raise ValueError("No videos found in the generated video dataset. Exiting.")

    print(f"Find {len(video_ids)} videos")
    batch_size = 1 # Process one by one for standard mean/std per video

    lpips_all = []
    psnr_all = []
    ssim_all = []

    for vid_id in tqdm.tqdm(video_ids):
        gen_path = os.path.join(generated_video_dir, f"{vid_id}.{file_extension}")
        gt_path = os.path.join(gt_video_dir, f"{vid_id}.{file_extension}")

        gen_video = load_video(gen_path)
        gt_video = load_video(gt_path)

        # Apply same preprocessing as in standard eval.py
        gt_video = preprocess_eval_video(gt_video, gen_video.shape)
        gen_video = gen_video[:gt_video.shape[0]] # Align lengths just in case

        # Normalize like standard eval.py
        gt_videos_tensor = (gt_video.unsqueeze(0) / 255.0).cpu()
        generated_videos_tensor = (gen_video.unsqueeze(0) / 255.0).cpu()

        # Metrics
        l_res = calculate_lpips(gt_videos_tensor, generated_videos_tensor, device=device)
        l_vals = list(l_res["value"].values())
        lpips_all.append(np.mean(l_vals))

        p_res = calculate_psnr(gt_videos_tensor, generated_videos_tensor)
        p_vals = list(p_res["value"].values())
        psnr_all.append(np.mean(p_vals))

        s_res = calculate_ssim(gt_videos_tensor, generated_videos_tensor)
        s_vals = list(s_res["value"].values())
        ssim_all.append(np.mean(s_vals))

    res = {
        "mode": args.mode_name if args.mode_name else os.path.basename(generated_video_dir),
        "baseline": os.path.basename(gt_video_dir),
        "num_videos": len(video_ids),
        "psnr": {
            "mean": float(np.mean(psnr_all)),
            "std": float(np.std(psnr_all))
        },
        "ssim": {
            "mean": float(np.mean(ssim_all)),
            "std": float(np.std(ssim_all))
        },
        "lpips": {
            "mean": float(np.mean(lpips_all)),
            "std": float(np.std(lpips_all))
        }
    }

    with open(args.output_json, "w") as f:
        json.dump(res, f, indent=2)

    print(f"Processed all videos. PSNR: {res['psnr']['mean']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_video_dir", type=str)
    parser.add_argument("--generated_video_dir", type=str)
    parser.add_argument("--output_json", type=str)
    parser.add_argument("--mode_name", type=str)

    args = parser.parse_args()
    main(args)
