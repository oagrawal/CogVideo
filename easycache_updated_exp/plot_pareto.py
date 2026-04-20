import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
csv_path = "/nfs/oagrawal/CogVideo/easycache_updated_exp/vbench_scores_table.csv"
df = pd.read_csv(csv_path)

# Convert columns to numeric (removing 's' from latency and 'x' from speedup if they exist, 
# though based on the table output they might be strings in the CSV if I didn't clean them in the writer)
# Let's check the CSV format in a moment if needed, but usually DictWriter writes the string values I provided.

def clean_latency(x):
    if isinstance(x, str):
        return float(x.replace('s', ''))
    return float(x)

df['latency_num'] = df['latency'].apply(clean_latency)
df['vbench_num'] = pd.to_numeric(df['vbench'])

# Separate baseline/fixed from adaptive
df['is_adaptive'] = df['mode'].str.contains('Adapt')

# Define colors
colors = {True: '#FF6F61', False: '#6B5B95'} # Adaptive: Coral/Red-ish, Fixed/Baseline: Purple/Blue-ish
labels = {True: 'Adaptive Modes', False: 'Baseline & Fixed Modes'}

plt.figure(figsize=(10, 6))

# Plot the connecting line for Baseline & Fixed Modes
fixed_group = df[~df['is_adaptive']].sort_values('latency_num')
plt.plot(fixed_group['latency_num'], fixed_group['vbench_num'], 
         linestyle='-', color='#6B5B95', alpha=0.5, label='Fixed Threshold Curve')

# Scatter points
for is_adapt, group in df.groupby('is_adaptive'):
    plt.scatter(group['latency_num'], group['vbench_num'], 
                c=colors[is_adapt], label=labels[is_adapt], s=100, edgecolors='black', alpha=0.8, zorder=3)

plt.xlabel('Latency (seconds)')
plt.ylabel('VBench Score (Aggregated)')
plt.title('CogVideoX1.5-5B EasyCache Pareto Frontier')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save the plot
output_path = "/nfs/oagrawal/CogVideo/easycache_updated_exp/pareto_frontier.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
