import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Data - using speedup values provided by user for CogVideo
speedups = {
    "cog_dc_fixed_0.05": 1.24,
    "cog_dc_fixed_0.10": 1.54,
    "cog_dc_fixed_0.15": 1.77,
    "cog_dc_fixed_0.20": 1.97,
    "cog_dc_fixed_0.25": 2.19,
    "cog_dc_fixed_0.30": 2.43,
    "cog_dc_fixed_0.40": 2.89,
    "cog_dc_fixed_0.50": 3.71,
    "cog_dc_fixed_0.60": 4.04,
    "cog_dc_fixed_0.70": 4.19,
    "cog_dc_fixed_0.35": 2.58,
    "cog_dc_adaptive_hi0.50_lo0.10": 2.16,
    "cog_dc_adaptive_hi0.50_lo0.20": 2.35,
    "cog_dc_adaptive_hi0.60_lo0.10": 2.17,
    "cog_dc_adaptive_hi0.60_lo0.10_late": 2.01,
    "cog_dc_adaptive_hi0.60_lo0.20": 2.38,
    "cog_dc_adaptive_hi0.60_lo0.20_late": 2.28,
    "cog_dc_adaptive_hi0.70_lo0.10": 2.19,
    "cog_dc_adaptive_hi0.70_lo0.20": 2.39,
    "cog_dc_adaptive_hi0.25_lo0.05_late": 1.74,
    "cog_dc_adaptive_hi0.30_lo0.05_late": 1.74,
    "cog_dc_adaptive_hi0.35_lo0.10_late": 1.93,
    "cog_dc_adaptive_hi0.25_lo0.10_early": 2.09,
}

# Scan fidelity metrics
fidelity_dir = "/nfs/oagrawal/CogVideo/dicache_results/fidelity_metrics"
data = []

if not os.path.exists(fidelity_dir):
    print(f"Error: {fidelity_dir} not found")
    exit(1)

for filename in os.listdir(fidelity_dir):
    if filename.endswith(".json"):
        with open(os.path.join(fidelity_dir, filename), "r") as f:
            try:
                js = json.load(f)
                mode = js["mode"]
                if mode in speedups:
                    data.append({
                        "mode": mode,
                        "speedup": speedups[mode],
                        "psnr": js["psnr"]["mean"],
                        "ssim": js["ssim"]["mean"],
                        "lpips": js["lpips"]["mean"],
                        "is_adaptive": "adaptive" in mode
                    })
            except Exception as e:
                print(f"Warning: Failed to parse {filename}: {e}")

if not data:
    print("Error: No data points found to plot.")
    exit(1)

df = pd.DataFrame(data)
print(f"Plotting {len(df)} points.")

# Colors
colors = {True: "#FF6F61", False: "#6B5B95"} # Adaptive vs Fixed
labels = {True: "Adaptive Modes", False: "Fixed Modes"}

metrics = ["psnr", "ssim", "lpips"]
metric_titles = {"psnr": "PSNR (dB) - Higher is Better", 
                 "ssim": "SSIM - Higher is Better", 
                 "lpips": "LPIPS - Lower is Better (Inverted Axis)"}

for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Sort for plotting line
    fixed_group = df[~df["is_adaptive"]].sort_values("speedup")
    
    # Plot connecting line for Fixed Modes
    plt.plot(fixed_group["speedup"], fixed_group[metric], linestyle="-", color="#6B5B95", alpha=0.5, label="Fixed Threshold Curve", zorder=1)
    
    # Scatter points
    for is_adapt, group in df.groupby("is_adaptive"):
        plt.scatter(group["speedup"], group[metric], 
                    c=colors[is_adapt], label=labels[is_adapt], s=120, edgecolors="black", alpha=0.8, zorder=3)
    
    plt.xlabel("Speedup (x)", fontsize=12)
    plt.ylabel(metric_titles[metric], fontsize=12)
    plt.title(f"CogVideo DiCache Pareto: {metric.upper()} vs Speedup (Top-Right is Better)", fontsize=14, pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left" if metric == "lpips" else "upper right")
    
    if metric == "lpips":
        plt.gca().invert_yaxis() # Lower is better, so move lower values to top
    
    output_path = f"/nfs/oagrawal/CogVideo/dicache_results/pareto_frontier_cog_{metric}_v2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
