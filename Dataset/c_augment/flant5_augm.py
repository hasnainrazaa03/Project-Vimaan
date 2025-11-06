import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Input and output paths
INPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\output\\base_cmds.jsonl")
OUTPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\output\\flant5_cmds.jsonl")

def load_base():
    """Load the base dataset"""
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    print("🚀 Loading FLAN-T5 Paraphraser (google/flan-t5-large)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)

    dataset = load_base()
    augmented = []

    for entry in tqdm(dataset, desc="🔁 Generating FLAN-T5 Paraphrases"):
        text = entry["text"]
        prompt = f"Paraphrase the following aviation command: '{text}'"

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_return_sequences=3,
                num_beams=5,
                temperature=0.9,
                do_sample=True
            )

            for out in outputs:
                paraphrase = tokenizer.decode(out, skip_special_tokens=True)
                if paraphrase.strip() and paraphrase.lower() != text.lower():
                    augmented.append({
                        "text": paraphrase.strip(),
                        "intent": entry["intent"],
                        "slots": entry["slots"]
                    })

        except Exception as e:
            print(f"⚠️ Failed for '{text}' → {e}")

    # Merge base + augmented and remove duplicates
    all_data = dataset + augmented
    unique = {d["text"]: d for d in all_data}.values()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in unique:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ FLAN-T5 augmentation complete → {len(unique)} entries → {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
