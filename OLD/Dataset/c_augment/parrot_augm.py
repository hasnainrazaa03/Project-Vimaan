import json
import torch
from tqdm import tqdm
from pathlib import Path
from parrot import Parrot

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
INPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\e_output\\base_cmds.jsonl")
OUTPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\e_output\\parrot_cmds.jsonl")
NUM_PARAPHRASES = 1


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_base():
    """Load base dataset from JSONL."""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_dataset(dataset, output_path):
    """Save dataset as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Saved → {len(dataset)} entries → {output_path.resolve()}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("🚀 Loading Parrot model...")
    parrot = Parrot(
        model_tag="prithivida/parrot_paraphraser_on_T5",
        use_gpu=torch.cuda.is_available()
    )

    dataset = load_base()
    augmented = []

    print(f"✨ Starting Parrot paraphrasing for {len(dataset)} base commands...\n")

    for entry in tqdm(dataset, desc="🔁 Paraphrasing", unit="cmd"):
        try:
            results = parrot.augment(
                input_phrase=entry["text"],
                use_gpu=torch.cuda.is_available(),
                do_diverse=False
            )

            if results:
                for paraphrase, _ in results[:NUM_PARAPHRASES]:
                    augmented.append({
                        "text": paraphrase,
                        "intent": entry["intent"],
                        "slots": entry["slots"]
                    })
        except Exception as e:
            tqdm.write(f"⚠️ Paraphrasing failed for '{entry['text']}' → {e}")

    all_data = dataset + augmented
    unique = list({d["text"]: d for d in all_data}.values())

    save_dataset(unique, OUTPUT_FILE)

# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
