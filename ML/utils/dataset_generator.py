"""
For categorical types (boolean_mode, binary_state, discrete_value):
  - Generates ALL exhaustive combinations of synonyms
  
For numeric types:
  - Cycles through unique values to ensure even distribution
  - Each value paired with different templates for variety
"""
import json
import random
from itertools import product, cycle
from template_filler import fill_template


def get_intent_type(intent_data):
    return intent_data.get("type", "unknown")


def get_all_slot_value_combinations(intent_data):
    slot_values_dict = {}
    
    for slot_name, slot_def in intent_data.get("slots", {}).items():
        if slot_def.get("type") == "categorical":
            slot_values_dict[slot_name] = slot_def.get("values", [])
    
    if not slot_values_dict:
        return [{}]
    
    slot_names = list(slot_values_dict.keys())
    slot_value_lists = [slot_values_dict[name] for name in slot_names]
    
    combinations = list(product(*slot_value_lists))
    
    result = []
    for combo in combinations:
        result.append({slot_names[i]: combo[i] for i in range(len(slot_names))})
    
    return result


def get_all_templates_for_combination(intent_data, combo, max_variations=None):
    templates = intent_data.get("templates", [])
    variations = []
    seen_texts = set()
    
    for template in templates:
        attempts = 10 
        for _ in range(attempts):
            text, slots_data = fill_template(template, intent_data, combo)
            
            if text not in seen_texts:
                seen_texts.add(text)
                variations.append((text, slots_data))
                
                if max_variations and len(variations) >= max_variations:
                    return variations
    
    return variations


def generate_examples_categorical(intent_key, intent_data):
    dataset = []
    intent_code = intent_data["intent"]
    
    combinations = get_all_slot_value_combinations(intent_data)
    
    print(f"  Found {len(combinations)} slot value combinations")
    
    total_examples = 0
    
    for combo in combinations:
        variations = get_all_templates_for_combination(intent_data, combo)
        
        for text, slots_data in variations:
            example = {
                "text": text,
                "intent": intent_code,
                "slots": slots_data
            }
            dataset.append(example)
            total_examples += 1
    
    print(f"  Generated {total_examples} exhaustive variations")
    
    return dataset


def generate_examples_numeric(intent_key, intent_data, num_examples=1000):
    dataset = []
    intent_code = intent_data["intent"]
    templates = intent_data.get("templates", [])
    
    numeric_slot = None
    slot_min = 0
    slot_max = 100
    
    for slot_name, slot_def in intent_data.get("slots", {}).items():
        if slot_def.get("type") == "numeric":
            numeric_slot = slot_name
            slot_min = slot_def.get("min", 0)
            slot_max = slot_def.get("max", 100)
            break
    
    if not numeric_slot:
        print(f"  Generating {num_examples} random examples (no numeric slot found)")
        for _ in range(num_examples):
            template = random.choice(templates)
            text, slots_data = fill_template(template, intent_data)
            if text:
                example = {
                    "text": text,
                    "intent": intent_code,
                    "slots": slots_data
                }
                dataset.append(example)
        return dataset
    
    value_range = slot_max - slot_min + 1
    
    print(f"  Generating {num_examples} examples with cycled values ({slot_min}-{slot_max})")
    
    all_values = list(range(slot_min, slot_max + 1))
    values_cycle = cycle(all_values)
    
    for i in range(num_examples):
        current_value = str(next(values_cycle))
        
        template = random.choice(templates)
        
        specific_values = {numeric_slot: current_value}
        text, slots_data = fill_template(template, intent_data, specific_values)
        
        if text:
            example = {
                "text": text,
                "intent": intent_code,
                "slots": slots_data
            }
            dataset.append(example)
    
    cycles = num_examples // value_range
    remainder = num_examples % value_range
    print(f"    Value distribution: {cycles} full cycles + {remainder} remainder")
    print(f"    Each value used ~{cycles}-{cycles+1} times across templates")
    
    return dataset


def generate_examples(schema, num_examples_per_intent_numeric=1000):
    dataset = []
    
    intents_by_type = {}
    for intent_key, intent_data in schema.items():
        intent_type = get_intent_type(intent_data)
        if intent_type not in intents_by_type:
            intents_by_type[intent_type] = []
        intents_by_type[intent_type].append((intent_key, intent_data))
    
    type_order = ["boolean_mode", "binary_state", "discrete_value", "numeric_value"]
    
    print(f"\n[INFO] Generating dataset for {len(schema)} intents...")
    
    for intent_type in type_order:
        if intent_type not in intents_by_type:
            continue
        
        print(f"\n[{intent_type.upper()}]")
        intents = intents_by_type[intent_type]
        
        for intent_key, intent_data in intents:
            intent_code = intent_data["intent"]
            print(f"  {intent_code}...")
            
            if intent_type in ["boolean_mode", "binary_state", "discrete_value"]:
                examples = generate_examples_categorical(intent_key, intent_data)
            
            elif intent_type == "numeric_value":
                examples = generate_examples_numeric(intent_key, intent_data, num_examples_per_intent_numeric)
            
            else:
                print(f"Unknown intent type: {intent_type}")
                examples = []
            
            dataset.extend(examples)
    
    return dataset


def calculate_stats(dataset):
    intent_counts = {}
    slot_counts = {}
    text_lengths = []
    intent_type_counts = {}
    
    for example in dataset:
        intent = example.get("intent")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        for slot_name in example.get("slots", {}).keys():
            slot_counts[slot_name] = slot_counts.get(slot_name, 0) + 1
        
        text_lengths.append(len(example.get("text", "")))
    
    stats = {
        "total_examples": len(dataset),
        "total_intents": len(intent_counts),
        "intent_distribution": intent_counts,
        "slot_coverage": slot_counts,
        "text_length": {
            "min": min(text_lengths) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
            "avg": sum(text_lengths) / len(text_lengths) if text_lengths else 0
        }
    }
    
    return stats


def write_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
