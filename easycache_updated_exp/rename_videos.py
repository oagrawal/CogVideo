import json
import os
import shutil

PROMPTS_FILE = "/nfs/oagrawal/CogVideo/easycache_updated_exp/prompts_subset.json"
VIDEOS_DIR = "/nfs/oagrawal/CogVideo/easycache_updated_exp/videos"
SEED = 0

def main():
    with open(PROMPTS_FILE, "r") as f:
        prompts = json.load(f)

    for mode in os.listdir(VIDEOS_DIR):
        mode_dir = os.path.join(VIDEOS_DIR, mode)
        if not os.path.isdir(mode_dir):
            continue

        for i, p in enumerate(prompts):
            prompt_text = p["prompt_en"] if "prompt_en" in p else (p["prompt"] if isinstance(p, dict) and "prompt" in p else str(p))
            
            old_name = f"{i:03d}-seed{SEED}.mp4"
            new_name = f"{prompt_text}-{SEED}.mp4"
            
            old_path = os.path.join(mode_dir, old_name)
            new_path = os.path.join(mode_dir, new_name)
            
            if os.path.exists(old_path):
                shutil.move(old_path, new_path)
                print(f"Renamed {old_path} -> {new_path}")

if __name__ == "__main__":
    main()
