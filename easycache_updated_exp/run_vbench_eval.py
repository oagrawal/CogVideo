#!/usr/bin/env python3
"""Run VBench evaluation on CogVideo updated EasyCache generated videos.

Modes evaluated:
  baseline, cog_ec_fixed_0.025, cog_ec_fixed_0.05, cog_ec_fixed_0.075, cog_ec_fixed_0.10

GPU split strategy: pass --modes to restrict to a subset of modes per GPU.
"""

# Compatibility patches (required for VBench + newer transformers/pytorch)
def _patch_torch_load():
    import torch
    import functools
    _original_load = torch.load
    @functools.wraps(_original_load)
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load

_patch_torch_load()


def _patch_transformers_modeling_utils():
    import transformers.modeling_utils as _mutils
    if not hasattr(_mutils, "apply_chunking_to_forward"):
        try:
            from transformers.pytorch_utils import apply_chunking_to_forward
            _mutils.apply_chunking_to_forward = apply_chunking_to_forward
        except ImportError:
            pass
    if not hasattr(_mutils, "prune_linear_layer"):
        try:
            from transformers.pytorch_utils import prune_linear_layer
            _mutils.prune_linear_layer = prune_linear_layer
        except ImportError:
            pass
    if not hasattr(_mutils, "find_pruneable_heads_and_indices"):
        try:
            from transformers.pytorch_utils import find_pruneable_heads_and_indices
            _mutils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        except ImportError:
            import torch
            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                heads = set(heads) - already_pruned_heads
                if not heads:
                    return set(), torch.arange(0, n_heads * head_size, device="cpu")
                mask = torch.ones(n_heads, head_size)
                for h in heads:
                    mask[h] = 0
                index = torch.where(mask.view(-1) == 1)[0]
                return heads, index
            _mutils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

_patch_transformers_modeling_utils()


import argparse
import json
import os
import sys
import torch
from vbench import VBench

DIMENSIONS = [
    "subject_consistency", "imaging_quality", "background_consistency",
    "motion_smoothness", "overall_consistency", "human_action",
    "multiple_objects", "spatial_relationship", "object_class", "color",
    "aesthetic_quality", "appearance_style", "temporal_flickering",
    "scene", "temporal_style", "dynamic_degree",
]
assert len(DIMENSIONS) == 16

ALL_MODES = [
    "baseline",
    "cog_ec_fixed_0.025",
    "cog_ec_fixed_0.05",
    "cog_ec_fixed_0.075",
    "cog_ec_fixed_0.10",
]


def _dimension_has_valid_result(save_path: str, dimension: str) -> bool:
    p = os.path.join(save_path, f"{dimension}_eval_results.json")
    if not os.path.isfile(p):
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        val = data.get(dimension)
        if not isinstance(val, list) or len(val) < 1:
            return False
        score = val[0]
        if score is None or (isinstance(score, float) and score != score):
            return False
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default="/nfs/oagrawal/CogVideo/easycache_updated_exp/videos")
    p.add_argument("--save-dir", default="/nfs/oagrawal/CogVideo/easycache_updated_exp/vbench_scores")
    p.add_argument("--full-info", default="/nfs/oagrawal/HunyuanVideo/vbench_eval/prompts_subset.json")
    p.add_argument("--modes", default="all",
                   help="Comma-separated list of modes, or 'all'")
    p.add_argument("--dimensions", default="all",
                   help="Comma-separated list of dimensions, or 'all'")
    args = p.parse_args()

    modes = ALL_MODES if args.modes == "all" else [m.strip() for m in args.modes.split(",")]
    dims = DIMENSIONS if args.dimensions == "all" else [d.strip() for d in args.dimensions.split(",")]

    print("=" * 70)
    print("CogVideo Updated EasyCache VBench Evaluation")
    print(f"Video dir : {args.video_dir}")
    print(f"Save dir  : {args.save_dir}")
    print(f"Modes     : {modes}")
    print(f"Dimensions: {len(dims)}")
    print("=" * 70)

    os.makedirs(args.save_dir, exist_ok=True)

    for mode in modes:
        video_path = os.path.join(args.video_dir, mode)
        if not os.path.exists(video_path):
            print(f"WARNING: {video_path} not found, skipping")
            continue

        save_path = os.path.join(args.save_dir, mode)
        os.makedirs(save_path, exist_ok=True)

        remaining = [d for d in dims if not _dimension_has_valid_result(save_path, d)]
        if not remaining:
            print(f"  {mode}: all {len(dims)} dimensions already done, skipping")
            continue

        print(f"\nEvaluating mode: {mode}  ({len(remaining)} dims remaining) ...")
        for i, dimension in enumerate(remaining, 1):
            print(f"  [{i}/{len(remaining)}] {dimension} ...", flush=True)
            full_info_path = os.path.join(save_path, f"{dimension}_full_info.json")
            try:
                vbench_eval = VBench(torch.device("cuda"), args.full_info, save_path)
                vbench_eval.evaluate(
                    videos_path=video_path,
                    name=dimension,
                    local=False,
                    read_frame=False,
                    dimension_list=[dimension],
                    mode="vbench_standard",
                    imaging_quality_preprocessing_mode="longer",
                )
            except Exception as e:
                print(f"    ERROR in {dimension}: {e}")

    # Summary verification
    print("\n" + "=" * 70)
    print("Verification: 16 dimensions per mode")
    print("=" * 70)
    all_ok = True
    for mode in modes:
        save_path = os.path.join(args.save_dir, mode)
        if not os.path.isdir(save_path):
            print(f"  {mode}: NO RESULTS DIR")
            all_ok = False
            continue
        missing = [d for d in DIMENSIONS if not _dimension_has_valid_result(save_path, d)]
        if missing:
            print(f"  {mode}: MISSING/INVALID dims: {missing}")
            all_ok = False
        else:
            print(f"  {mode}: OK (16/16 dimensions)")

    if all_ok:
        print("\nAll modes have all 16 dimensions evaluated.")
    else:
        print("\nWARNING: Some dimensions are missing. Re-run to retry failed dims.")
    print("\nCogVideo EasyCache VBench evaluation complete.")


if __name__ == "__main__":
    main()
