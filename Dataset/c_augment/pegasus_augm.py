import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_NAME = "tuner007/pegasus_paraphrase"
INPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\e_output\\base_cmds.jsonl")
OUTPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\e_output\\pegasus_cmds.jsonl")


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"❌ Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------
# Pegasus Augmentation
# ------------------------------------------------------------
def generate_paraphrases(model, tokenizer, text, num_return_sequences=3, num_beams=10):
    batch = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        max_length=60,
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        translated = model.generate(
            **batch,
            max_length=60,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1.5,
        )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
def main():
    print("🚀 Loading Pegasus model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    dataset = load_dataset(INPUT_FILE)
    augmented = []

    print(f"✅ Loaded {len(dataset)} entries from {INPUT_FILE.name}")
    print("⚙️  Starting Pegasus paraphrasing...\n")

    for entry in tqdm(dataset, desc="✨ Generating paraphrases"):
        text = entry["text"]
        try:
            paraphrases = generate_paraphrases(model, tokenizer, text)
            for p in paraphrases:
                augmented.append({
                    "text": p.strip(),
                    "intent": entry["intent"],
                    "slots": entry.get("slots", {})
                })
        except Exception as e:
            print(f"⚠️ Failed for '{text}': {e}")

    # Combine + remove duplicates
    all_data = dataset + augmented
    unique_data = {d["text"]: d for d in all_data}.values()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in unique_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Pegasus augmentation complete → {len(unique_data)} total entries")
    print(f"📁 Saved to: {OUTPUT_FILE.resolve()}")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
