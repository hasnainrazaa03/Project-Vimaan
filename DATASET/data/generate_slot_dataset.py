import sys 
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(current_dir)

sys.path.append(ml_dir)
sys.path.append(os.path.join(ml_dir, "config"))

import json
import random
from utils import get_next_version_path
from schema_config import SCHEMA

# --- CONFIGURATION ---
NUM_EXAMPLES_PER_INTENT = 2500  
OUTPUT_SUBDIR = os.path.join(ml_dir, "datasets", "01_base")

AVIATION_PREFIXES = [
    "check", "verify", "confirm", "set", "request", "select", "tune"
]

AVIATION_SUFFIXES = [
    "now", "please", "checked"
]

# --- DYNAMIC VALUE GENERATORS ---
def generate_dynamic_value(tag):
    if tag == "<DYNAMIC>": # COM Radio
        base = random.randint(118, 136)
        decimal = random.choice([0, 25, 50, 75, 5, 10, 15, 20, 30, 40, 60, 80, 90])
        return f"{base}.{decimal:02d}"
    
    elif tag == "<DYNAMIC_NAV>": # NAV Radio
        base = random.randint(108, 117)
        decimal = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 80, 90])
        return f"{base}.{decimal:02d}"
    
    elif tag == "<DYNAMIC_ADF>": # ADF
        return str(random.randint(190, 1750))
    
    elif tag == "<DYNAMIC_SQUAWK>": # Transponder
        return "".join([str(random.randint(0, 7)) for _ in range(4)])
    
    elif tag == "<DYNAMIC_BARO>": # Barometer
        if random.random() > 0.5:
            val = 28.0 + (random.random() * 3.0)
            return f"{val:.2f}"
        else:
            return str(random.randint(950, 1050))
    return "0"

# --- MAIN GENERATION LOGIC ---
def generate_dataset(schema, num_examples):
    dataset = []
    print(f"Starting generation for {len(schema)} intents...")
    
    for intent, details in schema.items():
        for _ in range(num_examples):
            template = random.choice(details.get("templates", [""]))
            filled_template = template
            slots_data = {}
            
            for slot_name, slot_details in details.get("slots", {}).items():
                allowed_values = slot_details["values"]
                
                if len(allowed_values) == 1 and allowed_values[0].startswith("<DYNAMIC"):
                    slot_value = generate_dynamic_value(allowed_values[0])
                else:
                    slot_value = random.choice(allowed_values)
                
                slots_data[slot_name] = slot_value
                
                word_to_speak = slot_value
                if 'synonyms' in slot_details:
                    if random.random() > 0.5:
                        syn_list = slot_details['synonyms'].get(slot_value, [])
                        if syn_list:
                            word_to_speak = random.choice(syn_list)
                            
                placeholder = "{" + slot_name + "}"
                filled_template = filled_template.replace(placeholder, word_to_speak)

            text = filled_template.strip()
            
            if random.random() < 0.25:
                if random.random() > 0.3: 
                    prefix = random.choice(AVIATION_PREFIXES)
                    
                    first_word = text.split()[0].lower()
                    if prefix not in first_word: 
                        text = f"{prefix} {text}"
                else:
                    suffix = random.choice(AVIATION_SUFFIXES)
                    if not text.lower().endswith(suffix):
                        text = f"{text} {suffix}"

            text = " ".join(text.split())
            
            dataset.append({
                "text": text,
                "intent": intent,
                "slots": slots_data
            })
            
    return dataset

if __name__ == "__main__":
    os.makedirs(OUTPUT_SUBDIR, exist_ok=True)
    
    generated_data = generate_dataset(SCHEMA, num_examples=NUM_EXAMPLES_PER_INTENT)
    random.shuffle(generated_data)
    
    base_filename = os.path.join(OUTPUT_SUBDIR, "aviation_cmds.jsonl")
    final_path = get_next_version_path(base_filename)
    
    print(f"\nWriting {len(generated_data)} examples to file...")
    with open(final_path, "w") as f:
        for entry in generated_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"SUCCESS: Dataset saved to: {final_path}")