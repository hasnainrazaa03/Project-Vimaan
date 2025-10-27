import json
import os
from utils import find_latest_version_path

def debug_training_labels(dataset_path):
    
    print(f"Loading dataset {dataset_path}...")
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    print("\nFinding examples with multi-word slots...")
    
    for item in data:
        for slot_name, slot_value in item.get('slots', {}).items():
            value_str = str(slot_value).lower()
            words_in_value = value_str.split()
            
            if len(words_in_value) > 1:
                print(f"\n{'='*80}")
                print(f"Text: {item['text']}")
                print(f"Intent: {item['intent']}")
                print(f"Slot: {slot_name} = {slot_value}")
                print(f"Value words: {words_in_value}")
                
                text_lower = item['text'].lower()
                if value_str in text_lower:
                    print(f"✓ Found in text: '{value_str}'")
                else:
                    print(f"✗ NOT found in text!")
                    for word in words_in_value:
                        if word in text_lower:
                            print(f"  - Found word: '{word}'")
                        else:
                            print(f"  - MISSING word: '{word}'")
                
                break
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    BASE_FILENAME = os.path.join(DATA_DIR, "aviation_cmds_final_training_set.jsonl")
    
    latest_dataset = find_latest_version_path(BASE_FILENAME)
    if latest_dataset:
        debug_training_labels(latest_dataset)
