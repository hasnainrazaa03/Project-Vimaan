import json
import os
from collections import defaultdict, Counter
from utils import find_latest_version_path

def analyze_dataset(dataset_path):
    """Analyze the dataset and print a comprehensive summary."""
    
    print("=" * 80)
    print("DATASET ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    total_examples = len(data)
    print(f"Total examples: {total_examples}\n")
    
    # Intent analysis
    print("-" * 80)
    print("INTENT DISTRIBUTION")
    print("-" * 80)
    
    intent_counts = Counter(item['intent'] for item in data)
    intent_sorted = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
    
    for intent, count in intent_sorted:
        percentage = (count / total_examples) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"{intent:40s} | {count:6d} ({percentage:5.1f}%) {bar}")
    
    # Slot analysis
    print("\n" + "-" * 80)
    print("SLOT DISTRIBUTION")
    print("-" * 80)
    
    slot_counts = defaultdict(int)
    slot_examples = defaultdict(list)
    
    for item in data:
        for slot_name, slot_value in item.get('slots', {}).items():
            slot_counts[slot_name] += 1
            # Store up to 3 examples for each slot
            if len(slot_examples[slot_name]) < 3:
                slot_examples[slot_name].append(str(slot_value))
    
    slot_sorted = sorted(slot_counts.items(), key=lambda x: x[1], reverse=True)
    
    for slot_name, count in slot_sorted:
        percentage = (count / total_examples) * 100
        examples = ", ".join(slot_examples[slot_name])
        print(f"\n{slot_name}:")
        print(f"  Occurrences: {count} ({percentage:.1f}%)")
        print(f"  Example values: {examples}")
    
    # Intent-Slot combinations
    print("\n" + "-" * 80)
    print("INTENT-SLOT COMBINATIONS")
    print("-" * 80)
    
    intent_slot_combinations = defaultdict(lambda: defaultdict(int))
    
    for item in data:
        intent = item['intent']
        slots = set(item.get('slots', {}).keys())
        for slot_name in slots:
            intent_slot_combinations[intent][slot_name] += 1
    
    for intent in sorted(intent_slot_combinations.keys()):
        print(f"\n{intent}:")
        for slot_name in sorted(intent_slot_combinations[intent].keys()):
            count = intent_slot_combinations[intent][slot_name]
            print(f"  - {slot_name}: {count}")
    
    # Command text analysis
    print("\n" + "-" * 80)
    print("TEXT STATISTICS")
    print("-" * 80)
    
    text_lengths = [len(item['text'].split()) for item in data]
    avg_length = sum(text_lengths) / len(text_lengths)
    min_length = min(text_lengths)
    max_length = max(text_lengths)
    
    print(f"Average command length: {avg_length:.1f} words")
    print(f"Min length: {min_length} words")
    print(f"Max length: {max_length} words")
    
    # Slots per example
    print("\n" + "-" * 80)
    print("SLOTS PER EXAMPLE DISTRIBUTION")
    print("-" * 80)
    
    slots_per_example = Counter(len(item.get('slots', {})) for item in data)
    for num_slots in sorted(slots_per_example.keys()):
        count = slots_per_example[num_slots]
        percentage = (count / total_examples) * 100
        print(f"  {num_slots} slots: {count:6d} examples ({percentage:5.1f}%)")
    
    # Sample examples
    print("\n" + "-" * 80)
    print("SAMPLE EXAMPLES")
    print("-" * 80)
    
    for i in range(min(5, len(data))):
        item = data[i]
        print(f"\nExample {i+1}:")
        print(f"  Text: {item['text']}")
        print(f"  Intent: {item['intent']}")
        if item.get('slots'):
            print(f"  Slots: {json.dumps(item['slots'], indent=4)}")
        else:
            print(f"  Slots: (none)")
    
    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    DATA_DIR = os.path.join(script_dir, "datasets", "05_final_merged")
    BASE_FILENAME = os.path.join(DATA_DIR, "aviation_cmds_final_training_set.jsonl")
    
    print(f"Searching for the latest dataset in '{DATA_DIR}'...")
    latest_dataset = find_latest_version_path(BASE_FILENAME)
    
    if latest_dataset and os.path.exists(latest_dataset):
        print(f"Found: {os.path.basename(latest_dataset)}\n")
        analyze_dataset(latest_dataset)
    else:
        print(f"Error: Dataset not found in '{DATA_DIR}'.")
