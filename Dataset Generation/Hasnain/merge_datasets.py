import json
from tqdm import tqdm
import os
import random
from utils import get_next_version_path, find_latest_version_path

def merge_datasets(file1, file2, output_file):

    unique_texts = set()
    merged_data = []
    
    files_to_process = [f for f in [file1, file2] if f is not None]
    
    print("Starting dataset merge...")
    
    for filename in files_to_process:
        print(f"Processing {filename}...")
        try:
            with open(filename, 'r') as f:
                for line in tqdm(f.readlines(), desc=f"Reading {os.path.basename(filename)}"):
                    try:
                        entry = json.loads(line)
                        text = entry.get("text", "")
                        
                        if text and text not in unique_texts:
                            unique_texts.add(text)
                            merged_data.append(entry)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line in {filename}")
        except FileNotFoundError:
            print(f"Warning: Input file not found: {filename}. Skipping.")

    random.shuffle(merged_data)
    
    print(f"\nWriting {len(merged_data)} unique entries to {output_file}...")
    with open(output_file, 'w') as f:
        for entry in merged_data:
            f.write(json.dumps(entry) + '\n')
            
    print("Merge complete!")
    print(f"Final merged dataset saved to '{output_file}'")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    
    PEGASUS_DIR = os.path.join(script_dir, "datasets", "04_clean_pegasus")
    FLAN_T5_DIR = os.path.join(script_dir, "datasets", "03_augmented_flan_t5")
    OUTPUT_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    BASE_PEGASUS_FILE = os.path.join(PEGASUS_DIR, "aviation_cmds_clean_pegasus.jsonl")
    BASE_FLAN_T5_FILE = os.path.join(FLAN_T5_DIR, "aviation_cmds_augmented_flan_t5.jsonl")
    BASE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aviation_cmds_final_training_set.jsonl")
    
    print("Searching for latest input files...")
    latest_pegasus_file = find_latest_version_path(BASE_PEGASUS_FILE)
    latest_flan_t5_file = find_latest_version_path(BASE_FLAN_T5_FILE)
    
    if not latest_pegasus_file and not latest_flan_t5_file:
        print("Error: No versioned input files found to merge.")
    else:
        print(f"Found Pegasus file: {os.path.basename(latest_pegasus_file) if latest_pegasus_file else 'None'}")
        print(f"Found FLAN-T5 file: {os.path.basename(latest_flan_t5_file) if latest_flan_t5_file else 'None'}")
        
        next_output_file = get_next_version_path(BASE_OUTPUT_FILE)
        print(f"Merged output will be saved to: {os.path.basename(next_output_file)}")
        merge_datasets(latest_pegasus_file, latest_flan_t5_file, next_output_file)