import sys
import os
import json
import random
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_dir = os.path.dirname(current_dir)

sys.path.append(ml_dir)
sys.path.append(os.path.join(ml_dir, "config"))
sys.path.append(os.path.join(ml_dir, "utils"))

from schema_config import SCHEMA, PREFIX_SUFFIX_CONFIG
from schema_validator import validate_schema, get_all_intents, get_all_slots, print_schema_info
from dataset_generator import generate_examples, calculate_stats, write_jsonl
from prefix_suffix_applier import create_prefix_suffix_variants
from dataset_validator import validate_dataset, print_validation_report
from utils import get_next_version_path

NUM_EXAMPLES_PER_INTENT = 2500
OUTPUT_DIR = os.path.join(ml_dir, "datasets", "01_base")


def get_intent_type(intent_data):
    return intent_data.get("type", "unknown")


def group_dataset_by_type_and_intent(schema, dataset):
    grouped = {
        "boolean_mode": {},
        "binary_state": {},
        "discrete_value": {},
        "numeric_value": {}
    }
    
    # Add intent names to groups
    for intent_key, intent_data in schema.items():
        intent_type = get_intent_type(intent_data)
        intent_code = intent_data["intent"]
        if intent_type in grouped:
            grouped[intent_type][intent_code] = []
    
    # Assign examples to groups
    for example in dataset:
        intent = example.get("intent")
        # Find which group this intent belongs to
        for intent_type, intents_dict in grouped.items():
            if intent in intents_dict:
                grouped[intent_type][intent].append(example)
                break
    
    ordered_dataset = []
    for intent_type in ["boolean_mode", "binary_state", "discrete_value", "numeric_value"]:
        for intent_code, examples in grouped[intent_type].items():
            ordered_dataset.extend(examples)
    
    return ordered_dataset, grouped


def write_grouped_jsonl(schema, dataset, filepath):
    ordered_dataset, grouped = group_dataset_by_type_and_intent(schema, dataset)
    
    with open(filepath, 'w') as f:
        header_comment = {
            "__comment": "=== VIMAAN BASE DATASET - GROUPED BY TYPE AND INTENT ===",
        }
        f.write(json.dumps(header_comment) + "\n")
        f.write("\n\n") 
        
        for idx, intent_type in enumerate(["boolean_mode", "binary_state", "discrete_value", "numeric_value"]):
            if not grouped[intent_type]:
                continue
            
            type_comment = {
                "__comment": f"--- {intent_type.upper()} ---",
                "type": intent_type,
                "total_intents": len(grouped[intent_type]),
                "total_examples": sum(len(ex) for ex in grouped[intent_type].values())
            }
            f.write(json.dumps(type_comment) + "\n")
            
            for intent_code, examples in grouped[intent_type].items():
                intent_comment = {
                    "__comment": intent_code,
                    "intent": intent_code,
                    "total_examples": len(examples)
                }
                f.write(json.dumps(intent_comment) + "\n")
                
                for example in examples:
                    f.write(json.dumps(example) + "\n")
            
            if idx < 3:
                f.write("\n\n")



def main():
    print("\n" + "="*70)
    print("VIMAAN BASE DATASET GENERATION - STAGE 1")
    print("="*70)
    
    print("\n[STEP 1] Validating schema...")
    is_valid, errors = validate_schema(SCHEMA)
    if not is_valid:
        print("Schema validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("Schema validation passed")
    print_schema_info(SCHEMA)
    
    print("\n[STEP 2] Generating base dataset...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = generate_examples(SCHEMA, NUM_EXAMPLES_PER_INTENT)
    print(f"Generated {len(dataset)} examples")
    
    print("\n[STEP 3] Calculating statistics...")
    stats = calculate_stats(dataset)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Total intents: {stats['total_intents']}")
    
    print("\n[STEP 4] Writing grouped base dataset...")
    base_path = os.path.join(OUTPUT_DIR, "aviation_cmds.jsonl")
    base_final_path = get_next_version_path(base_path)
    write_grouped_jsonl(SCHEMA, dataset, base_final_path)
    print(f"Base dataset saved: {os.path.basename(base_final_path)}")
    print(f"Examples: {stats['total_examples']}")
    
    print("\n[STEP 5] Creating 25% prefix/suffix augmented dataset...")
    augmented_examples = create_prefix_suffix_variants(dataset, PREFIX_SUFFIX_CONFIG, augmentation_rate=0.25)
    
    combined_dataset = dataset + augmented_examples
    
    print(f"Created {len(augmented_examples)} augmented examples (25% per intent)")
    print(f"Total dataset: {len(combined_dataset)} examples (125% of base)")
    
    print("\n[STEP 6] Writing grouped prefix/suffix dataset...")
    prefix_path = os.path.join(OUTPUT_DIR, "aviation_cmds_prefix_suffix.jsonl")
    prefix_final_path = get_next_version_path(prefix_path)
    write_grouped_jsonl(SCHEMA, combined_dataset, prefix_final_path)
    print(f"Prefix/suffix dataset saved: {os.path.basename(prefix_final_path)}")
    print(f"Examples: {len(combined_dataset)}")
    
    print("\n[STEP 7] Validating datasets...")
    valid_intents = get_all_intents(SCHEMA)
    valid_slots = list(get_all_slots(SCHEMA).keys())
    
    print("\nBase dataset validation:")
    base_report = validate_dataset(dataset, valid_intents, valid_slots)
    print_validation_report(base_report)
    
    print("Prefix/suffix dataset validation:")
    prefix_report = validate_dataset(combined_dataset, valid_intents, valid_slots)
    print_validation_report(prefix_report)
    
    print("\n[STEP 8] Calculating augmentation statistics...")
    augmented_stats = calculate_stats(augmented_examples)
    combined_stats = calculate_stats(combined_dataset)
    
    print(f"Augmented examples breakdown:")
    for intent, count in sorted(augmented_stats['intent_distribution'].items()):
        original_count = stats['intent_distribution'].get(intent, 0)
        percentage = (count / original_count * 100) if original_count > 0 else 0
        print(f"  {intent}: +{count} examples ({percentage:.1f}% of original)")
    
    print("\n[STEP 9] Saving metadata...")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "base_dataset": {
            "filename": os.path.basename(base_final_path),
            "path": base_final_path,
            "total_examples": len(dataset),
            "stats": stats,
            "validation": base_report
        },
        "prefix_suffix_dataset": {
            "filename": os.path.basename(prefix_final_path),
            "path": prefix_final_path,
            "total_examples": len(combined_dataset),
            "augmentation": {
                "method": "25% per intent prefix/suffix",
                "original_examples": len(dataset),
                "augmented_examples": len(augmented_examples),
                "total_percentage": f"{(len(combined_dataset) / len(dataset) * 100):.1f}%"
            },
            "augmented_stats": augmented_stats,
            "combined_stats": combined_stats,
            "validation": prefix_report
        },
        "schema_info": {
            "total_intents": len(SCHEMA),
            "intents": valid_intents,
            "total_slots": len(valid_slots),
            "slots": valid_slots
        }
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "generation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_path}")
    
    print("\n" + "="*70)
    print("STAGE 1 COMPLETE - BASE DATASETS GENERATED")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  1. {os.path.basename(base_final_path)}")
    print(f"     - {stats['total_examples']} examples")
    print(f"     - Grouped by type and intent")
    print(f"\n  2. {os.path.basename(prefix_final_path)}")
    print(f"     - {len(combined_dataset)} examples (100% + 25% augmentation)")
    print(f"     - Grouped by type and intent")
    print(f"     - Contains 25% of each intent with prefix/suffix")
    print(f"\n  3. generation_metadata.json")
    print(f"     - Complete statistics and validation reports")
    print()


if __name__ == "__main__":
    main()
