import json
import os
from normalization import normalize_slot_value
from utils import find_latest_version_path

def debug_labels(dataset_path):
    
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    print(f"Total examples: {len(data)}\n")
    
    numeric_issues = []
    
    for idx, item in enumerate(data[:1000]): 
        text = item['text'].lower()
        
        for slot_name, slot_value in item.get('slots', {}).items():
            value_str = str(slot_value).lower()
            
            if len(value_str) > 1 and value_str.isdigit():
                if value_str in text:
    
                    words = text.split()
                    
                    for word_idx, word in enumerate(words):
                        if value_str in word or value_str == word:
                            numeric_issues.append({
                                'text': item['text'],
                                'slot': slot_name,
                                'value': slot_value,
                                'word_containing_value': word,
                                'word_index': word_idx
                            })
                            break
    
    print("Multi-digit numeric slots found:")
    for issue in numeric_issues[:10]:
        print(f"  Text: {issue['text']}")
        print(f"  Slot: {issue['slot']} = {issue['value']}")
        print(f"  Word: '{issue['word_containing_value']}' @ index {issue['word_index']}\n")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    BASE_FILENAME = os.path.join(DATA_DIR, "aviation_cmds_final_training_set.jsonl")
    
    latest_dataset = find_latest_version_path(BASE_FILENAME)
    if latest_dataset:
        debug_labels(latest_dataset)
