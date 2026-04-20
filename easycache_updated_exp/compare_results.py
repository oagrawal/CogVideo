#!/usr/bin/env python3
"""Aggregate EasyCache VBench + fidelity + timing results. Runs outside Docker."""

import argparse
import json
import os
import glob
import csv

QUALITY_LIST = ["subject consistency", "background consistency", "temporal flickering", "motion smoothness", "aesthetic quality", "imaging quality", "dynamic degree"]
SEMANTIC_LIST = ["object class", "multiple objects", "human action", "color", "spatial relationship", "scene", "appearance style", "temporal style", "overall consistency"]
QUALITY_WEIGHT, SEMANTIC_WEIGHT = 4, 1

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0}, "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0}, "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0}, "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0}, "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0}, "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0}, "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222}, "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364}, "overall consistency": {"Min": 0.0, "Max": 0.364},
}
DIM_WEIGHT = {d: 1 for d in QUALITY_LIST + SEMANTIC_LIST}
DIM_WEIGHT["dynamic degree"] = 0.5

ALL_MODES = [
    "baseline", 
    "cog_ec_fixed_0.025", "cog_ec_fixed_0.05", "cog_ec_fixed_0.075", "cog_ec_fixed_0.10", "cog_ec_fixed_0.125",
    "cog_ec_fixed_0.15", "cog_ec_fixed_0.20",
    "cog_ec_adaptive_hi0.10_lo0.075_f9_l6", "cog_ec_adaptive_hi0.125_lo0.075_f9_l6",
    "cog_ec_adaptive_hi0.10_lo0.075_f13_l8", "cog_ec_adaptive_hi0.125_lo0.075_f13_l8",
    "cog_ec_adaptive_hi0.15_lo0.075_f9_l6", "cog_ec_adaptive_hi0.15_lo0.075_f13_l8",
    "cog_ec_adaptive_hi0.20_lo0.075_f9_l6", "cog_ec_adaptive_hi0.20_lo0.075_f13_l8",
]
MODE_LABELS = {
    "baseline": "Baseline",
    "cog_ec_fixed_0.025": "EasyCache 0.025",
    "cog_ec_fixed_0.05": "EasyCache 0.05",
    "cog_ec_fixed_0.075": "EasyCache 0.075",
    "cog_ec_fixed_0.10": "EasyCache 0.10",
    "cog_ec_fixed_0.125": "EasyCache 0.125",
    "cog_ec_fixed_0.15": "EasyCache 0.15",
    "cog_ec_fixed_0.20": "EasyCache 0.20",
    "cog_ec_adaptive_hi0.10_lo0.075_f9_l6": "Adapt-v2 (0.10/0.075) f9/l6",
    "cog_ec_adaptive_hi0.125_lo0.075_f9_l6": "Adapt-v2 (0.125/0.075) f9/l6",
    "cog_ec_adaptive_hi0.10_lo0.075_f13_l8": "Adapt-v2 (0.10/0.075) f13/l8",
    "cog_ec_adaptive_hi0.125_lo0.075_f13_l8": "Adapt-v2 (0.125/0.075) f13/l8",
    "cog_ec_adaptive_hi0.15_lo0.075_f9_l6": "Adapt-v2 (0.15/0.075) f9/l6",
    "cog_ec_adaptive_hi0.15_lo0.075_f13_l8": "Adapt-v2 (0.15/0.075) f13/l8",
    "cog_ec_adaptive_hi0.20_lo0.075_f9_l6": "Adapt-v2 (0.20/0.075) f9/l6",
    "cog_ec_adaptive_hi0.20_lo0.075_f13_l8": "Adapt-v2 (0.20/0.075) f13/l8",
}

def load_vbench_scores(score_dir):
    r = {}
    if not os.path.exists(score_dir):
        return r
    for f in os.listdir(score_dir):
        if f.endswith("_eval_results.json"):
            with open(os.path.join(score_dir, f)) as fp:
                d = json.load(fp)
            for k, v in d.items():
                r[k] = v[0] if isinstance(v, list) else v
    return r

def compute_aggregate(raw):
    scaled = {}
    for k, v in raw.items():
        dim = k.replace("_", " ")
        if dim in NORMALIZE_DIC:
            n = NORMALIZE_DIC[dim]
            s = (float(v) - n["Min"]) / (n["Max"] - n["Min"])
            scaled[dim] = s * DIM_WEIGHT.get(dim, 1)
    q = [scaled[d] for d in QUALITY_LIST if d in scaled]
    s = [scaled[d] for d in SEMANTIC_LIST if d in scaled]
    q_wsum = sum(DIM_WEIGHT.get(d, 1) for d in QUALITY_LIST if d in scaled)
    s_wsum = sum(DIM_WEIGHT.get(d, 1) for d in SEMANTIC_LIST if d in scaled)
    qs = sum(q) / q_wsum if q else None
    ss = sum(s) / s_wsum if s else None
    total = (qs * QUALITY_WEIGHT + ss * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT) if qs and ss else None
    return {"quality_score": qs, "semantic_score": ss, "total_score": total}

def load_timing(log_dir):
    m = {}
    # Handles both 'batch_gpu*.log' and 'batch_adaptive_gpu*.log'
    for p in glob.glob(os.path.join(log_dir, "batch*_gpu*.log")):
        with open(p) as f:
            for line in f:
                if line.startswith("DONE "):
                    parts = line.split()
                    mode = parts[1]
                    if "e2e:" in parts:
                        idx = parts.index("e2e:")
                        time_str = parts[idx + 1]
                        time_val = float(time_str.replace("s", ""))
                        m.setdefault(mode, []).append(time_val)
    return {mode: {"avg_time": sum(t)/len(t), "num_videos": len(t)} for mode, t in m.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", default="/nfs/oagrawal/CogVideo/easycache_updated_exp/vbench_scores")
    parser.add_argument("--gen-log-dir", default="/nfs/oagrawal/CogVideo/easycache_updated_exp")
    parser.add_argument("--output-csv", default="/nfs/oagrawal/CogVideo/easycache_updated_exp/vbench_scores_table.csv")
    args = parser.parse_args()

    timing = load_timing(args.gen_log_dir)
    baseline_time = timing.get("baseline", {}).get("avg_time")

    print("=" * 80)
    print("CogVideoX1.5-5B EasyCache Evaluation Results (VBench & Latency)")
    print("=" * 80)
    print("%-20s | %8s | %8s | %8s | %8s | %8s" % ("Mode", "Latency", "Speedup", "VBench", "Quality", "Semantic"))
    print("-" * 80)

    rows = []
    for mode in ALL_MODES:
        raw = load_vbench_scores(os.path.join(args.scores_dir, mode))
        agg = compute_aggregate(raw)
        t = timing.get(mode, {})
        speedup = baseline_time / t["avg_time"] if baseline_time and t else None
        
        row = {
            "mode": MODE_LABELS.get(mode, mode),
            "latency": "%.1fs" % t["avg_time"] if t else "-",
            "speedup": "%.2fx" % speedup if speedup else "-",
            "vbench": "%.4f" % (agg["total_score"] * 100) if agg["total_score"] else "-",
            "quality": "%.4f" % (agg["quality_score"] * 100) if agg["quality_score"] else "-",
            "semantic": "%.4f" % (agg["semantic_score"] * 100) if agg["semantic_score"] else "-",
        }
        rows.append(row)
        print("%-20s | %8s | %8s | %8s | %8s | %8s" % (
            row["mode"], row["latency"], row["speedup"], 
            row["vbench"], row["quality"], row["semantic"]))

    with open(args.output_csv, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["mode", "latency", "speedup", "vbench", "quality", "semantic"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV to {args.output_csv}")

if __name__ == "__main__":
    main()
