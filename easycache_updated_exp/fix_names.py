import json
import os
import shutil
import re

PROMPTS_FILE = "/nfs/oagrawal/CogVideo/easycache_updated_exp/prompts_subset.json"
VIDEOS_DIR = "/nfs/oagrawal/CogVideo/easycache_updated_exp/videos"

with open(PROMPTS_FILE, "r") as f:
    prompts = json.load(f)

for mode in os.listdir(VIDEOS_DIR):
    mode_dir = os.path.join(VIDEOS_DIR, mode)
    if not os.path.isdir(mode_dir):
        continue

    files = os.listdir(mode_dir)
    print(f"Checking {mode_dir} with {len(files)} files...")
    
    for i, p in enumerate(prompts):
        original_prompt = p.get("prompt_en", "")
        # VBench expects EXACTLY prompt_en + "-0.mp4"
        expected_name = f"{original_prompt}-0.mp4"
        
        # In case it is already correct
        if expected_name in files:
            continue
            
        # Try finding by removing punctuation
        stripped_prompt = re.sub(r'[^\w\s]', '', original_prompt)
        # also match potential existing names with no punctuation
        target_name_1 = f"{stripped_prompt}-0.mp4"
        target_name_2 = f"{i:03d}-seed0.mp4"
        
        found_file = None
        for f in files:
            # check the various ways the file might be named
            if f == target_name_1 or f == target_name_2:
                found_file = f
                break
            # Or if prompt string is in the filename anywhere
            # e.g., if punctuation was just removed
            stripped_f = re.sub(r'[^\w\s\-\.]', '', f)
            if stripped_f.replace('-0.mp4', '') == stripped_prompt:
                found_file = f
                break
                
        if found_file and found_file != expected_name:
            src = os.path.join(mode_dir, found_file)
            dst = os.path.join(mode_dir, expected_name)
            shutil.move(src, dst)
            print(f"Renamed: {found_file}\n      -> {expected_name}")
