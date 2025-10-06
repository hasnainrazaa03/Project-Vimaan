import json
from tqdm import tqdm
import os
from utils import get_next_version_path, find_latest_version_path

def clean_dataset(input_file, output_file):
    clean_data = []
    original_count = 0
    issues_found = 0
    
    print(f"Starting to clean {input_file}...")
    
    with open(input_file, 'r') as f:
        for line in tqdm(f.readlines(), desc="Processing entries"):
            original_count += 1
            try:
                entry = json.loads(line)
                is_valid = all(str(v).lower() in entry['text'].lower() for v in entry['slots'].values())
                
                if is_valid:
                    clean_data.append(entry)
                else:
                    issues_found += 1
                    
            except (json.JSONDecodeError, KeyError):
                issues_found += 1
                print(f"Skipping malformed or incomplete line: {line.strip()}")

    print(f"\nWriting {len(clean_data)} valid entries to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in clean_data:
            f.write(json.dumps(entry) + '\n')
            
    print("\n--- Cleaning Complete ---")
    print(f"Total entries processed: {original_count}")
    print(f"Entries removed due to issues: {issues_found}")
    print(f"Final count of clean entries: {len(clean_data)}")
    print(f"Clean dataset saved to '{output_file}'")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    
    INPUT_DIR = os.path.join(script_dir, "datasets", "02_augmented_pegasus")
    OUTPUT_DIR = os.path.join(script_dir, "datasets", "04_clean_pegasus")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    BASE_INPUT_FILENAME = os.path.join(INPUT_DIR, "aviation_cmds_augmented_pegasus.jsonl")
    BASE_OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "aviation_cmds_clean_pegasus.jsonl")
    
    print(f"Searching for the latest version in '{INPUT_DIR}'...")
    latest_input_file = find_latest_version_path(BASE_INPUT_FILENAME)
    
    if latest_input_file:
        next_output_file = get_next_version_path(BASE_OUTPUT_FILENAME)
        print(f"Found input: {os.path.basename(latest_input_file)}")
        print(f"Clean output will be saved to: {os.path.basename(next_output_file)}")
        
        clean_dataset(latest_input_file, next_output_file)
    else:
        print(f"Error: Could not find any version of the Pegasus dataset in '{INPUT_DIR}' to clean.")