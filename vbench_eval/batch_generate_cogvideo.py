#!/usr/bin/env python3
"""
Batch video generation for VBench evaluation using CogVideo + TeaCache (diffusers pipeline).

Generates videos for each prompt and each TeaCache mode:
  - cogvideo_baseline             — no TeaCache (rel_l1_thresh 0)
  - cogvideo_fixed_0.1            — fixed threshold 0.1
  - cogvideo_fixed_0.3            — fixed threshold 0.3
  - cogvideo_adaptive_0.1_17_0.3  — adaptive 0.1→0.3 (switch step 17/50)
  - cogvideo_adaptive_0.1_20_0.3  — adaptive 0.1→0.3 (switch step 20/50)

Saves videos in VBench naming: {prompt}-{seed}.mp4
Supports: resume (skips existing), --start-idx/--end-idx for GPU splitting.

Usage (from CogVideo repo root, inside cogvideo Docker container):

  # Single GPU (all 33 prompts x 4 modes)
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_cogvideo.py \
    --output-dir vbench_eval/videos

  # 4 GPUs (split prompts 0-8, 8-16, 16-24, 24-33)
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_cogvideo.py \
    --output-dir vbench_eval/videos --start-idx 0 --end-idx 8
  # ... repeat for GPUs 1,2,3 with end-idx 16,24,33
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

COGVIDEO_ROOT = str(Path(__file__).resolve().parent.parent)
if COGVIDEO_ROOT not in sys.path:
    sys.path.insert(0, COGVIDEO_ROOT)

MODES = [
    {"name": "cogvideo_baseline", "rel_l1_thresh": 0.0},
    {"name": "cogvideo_fixed_0.1", "rel_l1_thresh": 0.1},
    {"name": "cogvideo_fixed_0.3", "rel_l1_thresh": 0.3},
    # Adaptive schedule: thresh 0.1 then 0.3 (switch at step 17 of 50)
    {
        "name": "cogvideo_adaptive_0.1_17_0.3",
        "rel_l1_thresh": 0.3,
        "adaptive_schedule": {"switch_step": 17, "thresh1": 0.1, "thresh2": 0.3},
    },
    # Adaptive schedule: thresh 0.1 then 0.3 (switch at step 20 of 50)
    {
        "name": "cogvideo_adaptive_0.1_20_0.3",
        "rel_l1_thresh": 0.3,
        "adaptive_schedule": {"switch_step": 20, "thresh1": 0.1, "thresh2": 0.3},
    },
]


def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp, log_path)


def main():
    parser = argparse.ArgumentParser(description="CogVideo + TeaCache VBench batch video generation")
    parser.add_argument("--prompts-file", type=str,
                        default=os.path.join(COGVIDEO_ROOT, "vbench_eval", "prompts_subset.json"),
                        help="Path to VBench prompts JSON")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(COGVIDEO_ROOT, "vbench_eval", "videos"),
                        help="Base output directory (mode subdirs created here)")
    parser.add_argument("--ckpts-path", type=str, default="THUDM/CogVideoX1.5-5B",
                        help="HuggingFace model path")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        all_prompts = json.load(f)

    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    start_idx = args.start_idx
    prompts = all_prompts[start_idx:end_idx]

    seed = args.generation_seed
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)  # ensure output dir exists before any writes
    total_videos = len(prompts) * len(MODES)

    print("=" * 70)
    print("CogVideo + TeaCache VBench Batch Video Generation")
    print("=" * 70)
    print(f"Prompts file:  {args.prompts_file}")
    print(f"Prompt range:  [{start_idx}, {end_idx}) = {len(prompts)} prompts")
    print(f"Seed:          {seed}")
    print(f"Modes:         {[m['name'] for m in MODES]}")
    print(f"Total videos:  {total_videos}")
    print(f"Output dir:    {output_dir}")
    print(f"Model:         {args.ckpts_path}")
    print("  (Delta TEMNI plots disabled for batch; only .mp4 saved.)")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        for entry in prompts:
            prompt = entry["prompt_en"]
            for mode in MODES:
                fn = f"{prompt}-{seed}.mp4"
                path = os.path.join(output_dir, mode["name"], fn)
                st = "EXISTS" if os.path.exists(path) else "NEW"
                print(f"  [{st}] {mode['name']}/{fn}")
        existing = sum(1 for e in prompts for m in MODES
                      if os.path.exists(os.path.join(output_dir, m["name"], f"{e['prompt_en']}-{seed}.mp4")))
        print(f"\nAlready exist: {existing}, to generate: {total_videos - existing}")
        return

    try:
        from teacache_sample_video import run_generation
    except ImportError as e:
        print(f"ERROR: Failed to import teacache_sample_video.run_generation. "
              f"Run from CogVideo repo root: {e}")
        sys.exit(1)

    log_filename = f"generation_log_{start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    gen_log = load_generation_log(log_path)
    print(f"Log file: {log_path}\n")

    pipe = None
    completed = 0
    skipped = 0
    failed = 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(MODES):
            mode_name = mode["name"]
            video_filename = f"{prompt}-{seed}.mp4"
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, video_filename)
            run_num = prompt_idx * len(MODES) + mode_idx + 1
            run_key = f"{mode_name}|{prompt}|{seed}"

            if os.path.exists(video_path):
                print(f"[{run_num}/{total_videos}] SKIP (exists): {mode_name} | {prompt[:50]}...")
                skipped += 1
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)
                continue

            os.makedirs(video_dir, exist_ok=True)
            print(f"[{run_num}/{total_videos}] Generating: {mode_name} | {prompt[:50]}...")

            try:
                t0 = time.time()
                pipe = run_generation(
                    prompt=prompt,
                    seed=seed,
                    rel_l1_thresh=mode["rel_l1_thresh"],
                    save_file=video_path,
                    ckpts_path=args.ckpts_path,
                    num_inference_steps=args.num_inference_steps,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    fps=args.fps,
                    pipe=pipe,
                    skip_delta_plot=True,
                    adaptive_schedule=mode.get("adaptive_schedule"),
                )
                gen_time = time.time() - t0

                prompt_short = (prompt[:48] + "..") if len(prompt) > 50 else prompt
                print(f"  {mode_name:24} | {gen_time:7.1f}s | {prompt_short}")
                print(f"      -> {video_path}")

                completed += 1
                total_gen_time += gen_time
                gen_log["runs"].append({
                    "prompt": prompt,
                    "prompt_index": global_idx,
                    "seed": seed,
                    "mode": mode_name,
                    "time_seconds": round(gen_time, 1),
                    "video_path": video_path,
                    "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt,
                    "prompt_index": global_idx,
                    "seed": seed,
                    "mode": mode_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                save_generation_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print("BATCH GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Completed:     {completed}")
    print(f"  Skipped:       {skipped} (already existed)")
    print(f"  Failed:        {failed}")
    if completed:
        print(f"  Total time:    {total_gen_time:.1f}s  ({total_gen_time/3600:.1f}h)")
        print(f"  Avg per video: {total_gen_time/completed:.1f}s")
    print(f"  Log file:      {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
