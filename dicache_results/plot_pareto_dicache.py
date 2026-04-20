import pandas as pd
import matplotlib.pyplot as plt
import os

# Data provided by the user
data = [
    {"mode": "cog_dc_baseline", "latency": 1031, "vbench": 0.797976},
    {"mode": "cog_dc_fixed_0.05", "latency": 828, "vbench": 0.798914},
    {"mode": "cog_dc_fixed_0.10", "latency": 669, "vbench": 0.818691},
    {"mode": "cog_dc_fixed_0.15", "latency": 583, "vbench": 0.799257},
    {"mode": "cog_dc_fixed_0.20", "latency": 522, "vbench": 0.818878},
    {"mode": "cog_dc_fixed_0.25", "latency": 471, "vbench": 0.819763},
    {"mode": "cog_dc_fixed_0.30", "latency": 425, "vbench": 0.820780},
    {"mode": "cog_dc_fixed_0.40", "latency": 357, "vbench": 0.807067},
    {"mode": "cog_dc_fixed_0.50", "latency": 278, "vbench": 0.771306},
    {"mode": "cog_dc_fixed_0.60", "latency": 255, "vbench": 0.772201},
    {"mode": "cog_dc_fixed_0.70", "latency": 246, "vbench": 0.773524},
    {"mode": "cog_dc_adaptive_hi0.50_lo0.10", "latency": 477, "vbench": 0.763994},
    {"mode": "cog_dc_adaptive_hi0.60_lo0.10", "latency": 475, "vbench": 0.815023},
    {"mode": "cog_dc_adaptive_hi0.70_lo0.10", "latency": 470, "vbench": 0.797934},
    {"mode": "cog_dc_adaptive_hi0.60_lo0.10_late", "latency": 514, "vbench": 0.821098},
    {"mode": "cog_dc_adaptive_hi0.60_lo0.20", "latency": 433, "vbench": 0.7952},
    {"mode": "cog_dc_adaptive_hi0.50_lo0.20", "latency": 439, "vbench": 0.7952},
    {"mode": "cog_dc_adaptive_hi0.60_lo0.20_late", "latency": 452, "vbench": 0.82},
    {"mode": "cog_dc_adaptive_hi0.70_lo0.20", "latency": 432, "vbench": 0.8174},
]

df = pd.DataFrame(data)

# Separate baseline/fixed from adaptive
df['is_adaptive'] = df['mode'].str.contains('adaptive')

# Define colors
colors = {True: '#FF6F61', False: '#6B5B95'} # Adaptive: Coral/Red-ish, Fixed/Baseline: Purple/Blue-ish
labels = {True: 'Adaptive Modes', False: 'Baseline & Fixed Modes'}

plt.figure(figsize=(12, 7))

# Plot the connecting line for Baseline & Fixed Modes
fixed_group = df[~df['is_adaptive']].sort_values('latency')
plt.plot(fixed_group['latency'], fixed_group['vbench'], 
         linestyle='-', color='#6B5B95', alpha=0.5, label='Fixed Threshold Curve', zorder=1)

# Scatter points
for is_adapt, group in df.groupby('is_adaptive'):
    plt.scatter(group['latency'], group['vbench'], 
                c=colors[is_adapt], label=labels[is_adapt], s=120, edgecolors='black', alpha=0.8, zorder=3)

plt.xlabel('Latency (seconds)', fontsize=12)
plt.ylabel('VBench Score (Aggregated)', fontsize=12)
plt.title('CogVideoX1.5-5B DiCache Pareto Frontier: Quality vs. Latency', fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left')

plt.xlabel('Latency (seconds)', fontsize=12)

# Save the plot
output_path = "/nfs/oagrawal/CogVideo/dicache_results/pareto_frontier_dicache.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
