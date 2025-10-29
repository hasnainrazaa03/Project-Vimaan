import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from Dataset.input.config import SCHEMA


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
OUTPUT_FILE = Path("./Dataset/output/base_cmds.jsonl")

PREFIX_PHRASES = ["", "please", "hey", "could you"]
SUFFIX_PHRASES = ["", "for me"]
TEMPLATE_PATTERNS = [
    "{prefix} {command} {suffix}",
    "{command} {suffix}",
    "{prefix} {command}",
    "{command}",
]

# ---------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------
def generate_dataset(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    random.seed(42)
    records: List[Dict[str, Any]] = []

    for intent_name, intent_data in schema.items():
        command_templates = intent_data["command_templates"]
        slot_definitions = intent_data["slots"]

        for slot_name, slot_info in slot_definitions.items():
            for canonical_value, synonyms in slot_info["values"].items():
                for synonym in synonyms:
                    combinations = itertools.product(
                        PREFIX_PHRASES, SUFFIX_PHRASES, TEMPLATE_PATTERNS, command_templates
                    )

                    for prefix, suffix, template, base_cmd in combinations:
                        cmd_text = base_cmd.format(**{slot_name: synonym})
                        text = template.format(prefix=prefix, command=cmd_text, suffix=suffix)
                        text = " ".join(text.split())

                        records.append(
                            {
                                "text": text,
                                "intent": intent_name,
                                "slots": {slot_name: canonical_value},
                            }
                        )
    return records


# ---------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------
def deduplicate_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []
    for r in records:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique


# ---------------------------------------------------------------------
# Save to JSONL
# ---------------------------------------------------------------------
def save_dataset(dataset: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved dataset → {output_path.resolve()}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    print("🚀 Generating base dataset...")
    base_data = generate_dataset(SCHEMA)
    unique_data = deduplicate_dataset(base_data)
    print(f"✅ Base dataset: {len(unique_data)} unique commands")

    save_dataset(unique_data, OUTPUT_FILE)
    print(f"✅ Final dataset size (with paraphrases): {len(unique_data)}")


if __name__ == "__main__":
    main()
