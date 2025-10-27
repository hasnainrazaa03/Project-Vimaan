import json
import os
from tqdm import tqdm
from utils import find_latest_version_path, get_next_version_path
from core import normalize_dataset


def add_word_form_variants(dataset_path):
    print(f"Loading dataset: {dataset_path}")
    
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in tqdm(f.readlines(), desc="Loading dataset")]
    
    new_examples = []
    numeric_slots = ['altitude', 'degrees', 'flight_level', 'frequency', 'com_port']
    
    count_added = 0
    
    print("\nGenerating word-form variants...")
    for item in tqdm(data, desc="Processing examples"):
        new_examples.append(item)
        
        text = item['text']
        slots = item.get('slots', {})
        
        if not slots:
            continue
        
        modified_text = text
        modified = False
        
        for slot_name, slot_value in slots.items():
            if slot_name in numeric_slots:
                value_str = str(slot_value).strip()
                
                word_forms = {
                    '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
                    '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '0': 'zero',
                }
                
                try:
                    if value_str in word_forms:
                        word_form = word_forms[value_str]
                        modified_text = modified_text.replace(value_str, word_form)
                        modified = True
                except:
                    pass
        
        if modified and modified_text != text:
            new_examples.append({
                "text": modified_text,
                "intent": item.get('intent'),
                "slots": slots
            })
            count_added += 1
    
    print(f"\n Generated {count_added} word-form variants")
    print(f"Total examples before normalization: {len(new_examples)}")
    
    print("\nNormalizing all slot values...")
    new_examples = normalize_dataset(new_examples)
    
    output_path = get_next_version_path(dataset_path)
    print(f"\nSaving {len(new_examples)} total examples to {output_path}")
    
    with open(output_path, 'w') as f:
        for example in tqdm(new_examples, desc="Writing to file"):
            f.write(json.dumps(example) + '\n')
    
    print(f"\n" + "="*70)
    print(f"AUGMENTATION COMPLETE!")
    print(f"="*70)
    print(f"Added {count_added} word-form examples")
    print(f"Dataset expanded from {len(data)} → {len(new_examples)} examples")
    print(f"All slots normalized with improved normalization.py")
    print(f"Output file: {os.path.basename(output_path)}")
    print(f"Full path: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    BASE_FILENAME = os.path.join(DATA_DIR, "aviation_cmds_final_training_set.jsonl")
    
    latest_dataset = find_latest_version_path(BASE_FILENAME)
    if latest_dataset:
        print(f"Found latest dataset: {os.path.basename(latest_dataset)}")
        print(f"Full path: {latest_dataset}")
        add_word_form_variants(latest_dataset)
    else:
        print("Dataset not found!")
        print(f"Looking in: {DATA_DIR}")