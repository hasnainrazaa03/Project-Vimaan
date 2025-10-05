import json
import random

#COMMAND SCHEMA WITH INTENTS AND SLOTS
SCHEMA = {
    "set_autopilot_heading": {
        "templates": [
            "set heading {degrees}",
            "change heading to {degrees}",
            "turn to {degrees} degrees",
            "fly heading {degrees}"
        ],
        "slots": {
            "degrees": {
                "type": "numerical",
                "values": [str(i) for i in range(0, 361, 1)] #Headings from 0 to 360
            }
        }
    },
    "set_autopilot_altitude": {
        "templates": [
            "set altitude {altitude}",
            "climb to {altitude} feet",
            "descend to {altitude} feet",
            "maintain flight level {altitude}"
        ],
        "slots": {
            "altitude": {
                "type": "numerical",
                "values": [str(i) for i in range(100, 40001, 100)] #Altitudes from 100 to 40000
            }
        }
    },
    "set_com_frequency": {
        "templates": [
            "set com {com_port} to {frequency}",
            "tune com {com_port} {frequency}",
            "frequency {frequency} on com {com_port}"
        ],
        "slots": {
            "com_port": {
                "type": "categorical",
                "values": ["1", "2"]
            },
            "frequency": {
                "type": "numerical",
                #Generate sample COM frequencies
                "values": [f"{random.randint(118, 136)}.{random.randint(0, 99):02d}" for _ in range(100)]
            }
        }
    },
    "toggle_landing_gear": {
        "templates": [
            "gear {state}",
            "{state} landing gear",
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["up", "down"]
            }
        }
    }
}

#PHRASE VARIATIONS (PREFIXES AND SUFFIXES)
PREFIXES = ["", "please", "could you", "request", "confirm"]
SUFFIXES = ["", "now", "immediately", "for me"]

#GENERATION LOGIC
def generate_dataset(schema, num_examples_per_intent=200):
    dataset = []
    
    for intent, details in schema.items():
        print(f"Generating examples for intent: {intent}...")
        
        for i in range(num_examples_per_intent):
            #random template for the command
            template = random.choice(details["templates"])
            
            #slot fill in the template with random values
            filled_template = template
            slots_data = {}
            for slot_name, slot_details in details["slots"].items():
                slot_value = random.choice(slot_details["values"])
                slots_data[slot_name] = slot_value
                
                #replace placeholder in the template string
                placeholder = "{" + slot_name + "}"
                filled_template = filled_template.replace(placeholder, slot_value)

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
    generated_data = generate_dataset(SCHEMA, num_examples_per_intent=1500)
    random.shuffle(generated_data)
    
    output_filename = "aviation_cmds.jsonl"
    with open(output_filename, "w") as f:
        for entry in generated_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nSuccessfully generated {len(generated_data)} examples.")
    print(f"Dataset saved to '{output_filename}'")