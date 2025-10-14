import json
from tqdm import tqdm
import os
from utils import find_latest_version_path

try:
    from num2words import num2words
except ImportError:
    print("Installing 'num2words' library...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "num2words"])
    from num2words import num2words


def verify_dataset(filename):
    print(f"Verifying dataset: {filename}...")
    
    total_entries = 0
    issues_found = 0
    
    with open(filename, "r") as f:
        for i, line in enumerate(tqdm(f.readlines(), desc="Verifying entries")):
            total_entries += 1
            try:
                entry = json.loads(line)
                
                if not all(key in entry for key in ["text", "intent", "slots"]):
                    print(f"\nL{i+1}: Missing key. Entry: {entry}")
                    issues_found += 1
                    continue
                
                text_lower = entry['text'].lower()
                for slot_value in entry['slots'].values():
                    value_str = str(slot_value).lower()
                    
                    is_present = False
                    if value_str.isdigit():
                        value_word = num2words(value_str)
                        if value_str in text_lower or value_word in text_lower:
                            is_present = True
                    else:
                        if value_str in text_lower:
                            is_present = True
                    
                    if not is_present:
                        print(f"\nL{i+1}: Slot value '{slot_value}' not found as digit or word in text: '{entry['text']}'")
                        issues_found += 1

            except json.JSONDecodeError:
                print(f"\nL{i+1}: Invalid JSON format.")
                issues_found += 1

    print("\n--- Verification Complete ---")
    print(f"Total entries checked: {total_entries}")
    if issues_found == 0:
        print("No issues found. The dataset looks great!")
    else:
        print(f"Found {issues_found} potential issues. Please review the lines printed above.")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    
    print(f"Selection: \n 1. Pegasus \n 2. Flan T5 \n 3. Merged")
    selection = 3
    print("Selection: ", selection)
    if(selection == 1):
        INPUT_DIR = os.path.join(script_dir, "datasets", "02_augmented_pegasus")
        BASE_FILENAME_TO_VERIFY = os.path.join(INPUT_DIR, "aviation_cmds_augmented_pegasus.jsonl")
    elif(selection == 2):
        INPUT_DIR = os.path.join(script_dir, "datasets", "03_augmented_flan_t5")
        BASE_FILENAME_TO_VERIFY = os.path.join(INPUT_DIR, "aviation_cmds_augmented_flan_t5.jsonl")
    else:
        INPUT_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
        BASE_FILENAME_TO_VERIFY = os.path.join(INPUT_DIR, "aviation_cmds_final_training_set.jsonl")
    
    print(f"Searching for the latest version in '{INPUT_DIR}'...")
    latest_file = find_latest_version_path(BASE_FILENAME_TO_VERIFY)
    
    if latest_file and os.path.exists(latest_file):
        verify_dataset(filename=latest_file)
    else:
        print(f"Error: Could not find any version of the final dataset in '{INPUT_DIR}' to check.")