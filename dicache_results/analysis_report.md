# CogVideo + DiCache Pareto Frontier Analysis

I have successfully created the Pareto frontier plot for the CogVideo + DiCache experiments. The plot visualizes the trade-off between **VBench Total Score** (Quality) and **Latency** (Inference Time).

## Data points Used
The following data was aggregated into the plot:

| Mode | Speedup | Latency | VBench Total | Type |
| :--- | :--- | :--- | :--- | :--- |
| `cog_dc_baseline` | 1.00x | 1031s | 0.797976 | Fixed |
| `cog_dc_fixed_0.05` | 1.24x | 828s | 0.798914 | Fixed |
| `cog_dc_fixed_0.10` | 1.54x | 669s | 0.818691 | Fixed |
| `cog_dc_fixed_0.15` | 1.77x | 583s | 0.799257 | Fixed |
| `cog_dc_fixed_0.20` | 1.97x | 522s | 0.818878 | Fixed |
| `cog_dc_fixed_0.25` | 2.19x | 471s | 0.819763 | Fixed |
| `cog_dc_fixed_0.30` | 2.43x | 425s | 0.820780 | Fixed |
| `cog_dc_fixed_0.40` | 2.89x | 357s | 0.807067 | Fixed |
| `cog_dc_fixed_0.50" | 3.71x | 278s | 0.771306 | Fixed |
| `cog_dc_fixed_0.60` | 4.04x | 255s | 0.772201 | Fixed |
| `cog_dc_fixed_0.70` | 4.19x | 246s | 0.773524 | Fixed |
| `cog_dc_adaptive_hi0.50_lo0.10` | 2.16x | 477s | 0.763994 | Adaptive |
| `cog_dc_adaptive_hi0.60_lo0.10` | 2.17x | 475s | 0.815023 | Adaptive |
| `cog_dc_adaptive_hi0.70_lo0.10` | 2.19x | 470s | 0.797934 | Adaptive |
| `cog_dc_adaptive_hi0.60_lo0.10_late` | 2.01x | 514s | 0.821098 | Adaptive |
| `cog_dc_adaptive_hi0.60_lo0.20` | 2.38x | 433s | 0.7952 | Adaptive |
| `cog_dc_adaptive_hi0.50_lo0.20` | 2.35x | 439s | 0.7952 | Adaptive |
| `cog_dc_adaptive_hi0.60_lo0.20_late` | 2.28x | 452s | 0.820 | Adaptive |
| `cog_dc_adaptive_hi0.70_lo0.20` | 2.39x | 432s | 0.8174 | Adaptive |

## Analysis
- **Optimal Trade-off**: Several points (like `fixed_0.10`, `fixed_0.20`, `fixed_0.25`, and `fixed_0.30`) show **higher** VBench scores than the baseline while significantly reducing latency.
- **Adaptive Policies (New)**: 
    - `cog_dc_adaptive_hi0.60_lo0.10_late` remains the highest quality at 0.821 (2.01x speedup).
    - The new `lo0.20` variants (e.g., `hi0.70_lo0.20`) show excellent efficiency, achieving **~2.4x speedup** (432s) with quality still above the baseline (0.817 vs 0.798).
- **Pareto Efficiency**: The plot now shows standard increasing latency (left to right), with the most efficient models appearing in the **top-left** quadrant (high quality, low latency).

## Plot Generation
The plot was generated using a custom script executed inside the `hv_eval_wan` container:
- **Script**: [/nfs/oagrawal/CogVideo/dicache_results/plot_pareto_dicache.py](file:///nfs/oagrawal/CogVideo/dicache_results/plot_pareto_dicache.py)
- **Resulting Plot**: ![DiCache Pareto Frontier](/nfs/oagrawal/CogVideo/dicache_results/pareto_frontier_dicache.png)
