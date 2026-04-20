import json
import os
import matplotlib.pyplot as plt

def main():
    json_path = "/nfs/oagrawal/CogVideo/easycache_updated_exp/baseline_profile_20260408_134530_seed0/profiling.json"
    run_dir = os.path.dirname(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    pred_change = data["pred_change_history"]
    
    # In easycache_baseline_forward:
    # pred_change[0] is recorded at cnt=2
    # pred_change[i] is recorded at cnt=i+2
    # Total steps = 50 (0 to 49)
    # pred_change list has 48 items (cnt=2 to cnt=49)
    
    # We want to plot steps 5 to 48 (eligible steps)
    # Step 5 corresponds to index 5 - 2 = 3
    # Step 48 corresponds to index 48 - 2 = 46
    
    eligible_steps_indices = range(5, 49) # 5..48
    eligible_pred_change = pred_change[3:47] 
    
    plt.figure(figsize=(10, 5))
    plt.plot(eligible_steps_indices, eligible_pred_change, marker='s', color='#2ecc71', linewidth=1.5, markersize=4)
    
    plt.title("CogVideoX1.5-5B: Prediction Change (Cache-Eligible Steps 5-48)")
    plt.xlabel("Denoising Step")
    plt.ylabel("pred_change")
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line for the 0.05 threshold for reference
    plt.axhline(y=0.05, color='#e74c3c', linestyle='--', alpha=0.6, label="Fixed Thresh 0.05")
    
    # Add a horizontal line for the valley mean (~0.035)
    plt.axhline(y=0.035, color='#3498db', linestyle=':', alpha=0.6, label="Valley Mean (~0.035)")
    
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join("/nfs/oagrawal/CogVideo/easycache_updated_exp/", "pred_change_eligible_steps.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    
    print(f"Specialized plot saved to: {save_path}")

if __name__ == "__main__":
    main()
