import json
import os
import re

from tqdm import tqdm
from utils import find_latest_version_path, get_next_version_path
from vimaan_nlu import normalize_dataset

_DIGIT_WORD_FORMS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def add_word_form_variants(dataset_path):
    print(f"Loading dataset: {dataset_path}")

    with open(dataset_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in tqdm(f.readlines(), desc="Loading dataset")]

    new_examples = []
    # Only single-digit slots are safe to spell out, and ONLY with a
    # word-boundary replace: a bare `str.replace("1","one")` mangled the "1" in
    # a "118.75" frequency ("one18.75"). frequency/com_port are excluded (a
    # bare digit appears inside them), leaving single-digit degrees.
    safe_numeric_slots = {"degrees", "altitude", "flight_level"}

    count_added = 0

    print("\nGenerating word-form variants...")
    for item in tqdm(data, desc="Processing examples"):
        new_examples.append(item)

        text = item["text"]
        slots = item.get("slots", {})

        if not slots:
            continue

        modified_text = text
        modified = False

        for slot_name, slot_value in slots.items():
            if slot_name in safe_numeric_slots:
                value_str = str(slot_value).strip()
                if value_str in _DIGIT_WORD_FORMS:
                    modified_text = re.sub(
                        r"\b" + re.escape(value_str) + r"\b",
                        _DIGIT_WORD_FORMS[value_str],
                        modified_text,
                    )
                    modified = True

        if modified and modified_text != text:
            new_examples.append(
                {"text": modified_text, "intent": item.get("intent"), "slots": slots}
            )
            count_added += 1

    print(f"\n Generated {count_added} word-form variants")
    print(f"Total examples before normalization: {len(new_examples)}")

    print("\nNormalizing all slot values...")
    new_examples = normalize_dataset(new_examples)

    output_path = get_next_version_path(dataset_path)
    print(f"\nSaving {len(new_examples)} total examples to {output_path}")

    with open(output_path, "w") as f:
        for example in tqdm(new_examples, desc="Writing to file"):
            f.write(json.dumps(example) + "\n")

    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE!")
    print("=" * 70)
    print(f"Added {count_added} word-form examples")
    print(f"Dataset expanded from {len(data)} → {len(new_examples)} examples")
    print("All slots normalized with improved normalization.py")
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
