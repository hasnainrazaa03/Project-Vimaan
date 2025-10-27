import os
import json
import random
from utils import get_next_version_path
from schema_config import SCHEMA

#PHRASE VARIATIONS (PREFIXES AND SUFFIXES)
PREFIXES = ["", "please", "could you", "request", "confirm", "go ahead and"]
SUFFIXES = ["", "now", "immediately", "for me", "if you would"]

#GENERATION LOGIC
def generate_dataset(schema, num_examples_per_intent=1000):
    dataset = []
    
    for intent, details in schema.items():
        print(f"Generating examples for intent: {intent}...")
        
        for i in range(num_examples_per_intent):
            #random template for the command
            template = random.choice(details.get("templates", [""]))
            
            #slot fill in the template with random values
            filled_template = template
            slots_data = {}
            for slot_name, slot_details in details.get("slots", {}).items():
                slot_value = ""
                #COM frequency generation
                if slot_details["values"] == ["<DYNAMIC>"]:
                    if slot_name == "frequency":
                        slot_value = f"{random.randint(118, 136)}.{random.randint(0, 99):02d}"
                    # Add more dynamic types here if needed in the future
                else:
                    slot_value = random.choice(slot_details["values"])
                
                slots_data[slot_name] = slot_value
                
                word_to_use_in_text = slot_value
                if 'synonyms' in slot_details and random.random() > 0.5:
                    synonyms = slot_details['synonyms'].get(slot_value, [])
                    if synonyms:
                        word_to_use_in_text = random.choice(synonyms)

                #replace placeholder in the template string
                placeholder = "{" + slot_name + "}"
                filled_template = filled_template.replace(placeholder, word_to_use_in_text)

            #natural language variations
            text = f"{random.choice(PREFIXES)} {filled_template} {random.choice(SUFFIXES)}".strip()
            text = " ".join(text.split())

            #final structured data entry
            dataset_entry = {
                "text": text,
                "intent": intent,
                "slots": slots_data
            }
            dataset.append(dataset_entry)
            
    return dataset

if __name__ == "__main__":
    generated_data = generate_dataset(SCHEMA, num_examples_per_intent=2500)
    random.shuffle(generated_data)
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "01_base")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_output_filename = os.path.join(OUTPUT_DIR, "aviation_cmds.jsonl")
    output_filename = get_next_version_path(base_output_filename)
    print(f"Output will be saved to new version: {output_filename}")
    with open(output_filename, "w") as f:
        for entry in generated_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nSuccessfully generated {len(generated_data)} examples.")
    print(f"Dataset saved to '{output_filename}'")