import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import unicodedata

INPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\e_output\\base_cmds.jsonl")
OUTPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\e_output\\flant5_cmds.jsonl")

def load_base():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
    return s.lower()

def main(batch_size=24):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    model.eval()

    dataset = load_base()
    augmented = []

    prompts = [
        (
            entry,
            f"Paraphrase the following aviation command: {entry['text']}"
        )
        for entry in dataset
    ]

    for i in tqdm(range(0, len(prompts), batch_size), desc="⚡ Generating batched paraphrases"):
        batch = prompts[i:i + batch_size]
        texts = [p[1] for p in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_return_sequences=1,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, out_text in enumerate(decoded):
            entry = batch[j % len(batch)][0]  # in case of repetition
            norm_p = normalize_text(out_text)
            norm_t = normalize_text(entry["text"])
            if norm_p and norm_p != norm_t:
                augmented.append({
                    "text": norm_p,
                    "intent": entry["intent"],
                    "slots": entry["slots"]
                })

    all_data = dataset + augmented
    unique = {}
    for d in all_data:
        key = normalize_text(d["text"])
        if key not in unique:
            d["text"] = key
            unique[key] = d

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in unique.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ FLAN-T5 augmentation complete → {len(unique)} entries → {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()