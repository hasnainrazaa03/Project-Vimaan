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
    print("🚀 Loading FLAN-T5 Paraphraser (google/flan-t5-base)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

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

# import json
# from pathlib import Path
# from tqdm import tqdm
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# MODEL_NAME = "google/flan-t5-base"
# INPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\output\\base_cmds.jsonl")
# OUTPUT_FILE = Path(r"D:\\Project-Vimaan\\Dataset\\output\\flant5_cmds.jsonl")

# BATCH_SIZE = 4
# MAX_NEW_TOKENS = 32
# NUM_RETURN = 3
# TOP_P = 0.92
# TOP_K = 50
# TEMPERATURE = 0.9

# def load_base():
#     with open(INPUT_FILE, "r", encoding="utf-8") as f:
#         for line in f:
#             yield json.loads(line)

# def batched_iterable(iterable, n):
#     batch = []
#     for item in iterable:
#         batch.append(item)
#         if len(batch) >= n:
#             yield batch
#             batch = []
#     if batch:
#         yield batch

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}. Loading model {MODEL_NAME} ...")

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     map_args = {}
#     if device == "cuda":
#         map_args = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True)
#     model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, **map_args).to(device)
#     model.eval()

#     gen_kwargs = dict(
#         max_new_tokens=MAX_NEW_TOKENS,
#         do_sample=True,
#         top_p=TOP_P,
#         top_k=TOP_K,
#         temperature=TEMPERATURE,
#         num_return_sequences=NUM_RETURN,
#         num_beams=1,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     dataset_iter = load_base()
#     seen_texts = set()
#     OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

#     with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
#         warm_prompts = ["Paraphrase the following aviation command: 'fly heading 180'"] * min(2, BATCH_SIZE)
#         with torch.inference_mode():
#             inputs = tokenizer(warm_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
#             model.generate(**inputs, **gen_kwargs)

#         for batch in tqdm(batched_iterable(dataset_iter, BATCH_SIZE), desc="Generating"):
#             prompts = [f"Paraphrase the following aviation command: '{entry['text']}'" for entry in batch]

#             inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

#             with torch.inference_mode():
#                 outputs = model.generate(**inputs, **gen_kwargs)

#             decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#             for i, entry in enumerate(batch):
#                 base_text = entry["text"]
#                 start = i * NUM_RETURN
#                 end = start + NUM_RETURN
#                 for paraphrase in decoded[start:end]:
#                     if not paraphrase:
#                         continue
#                     p = paraphrase.strip()
#                     if not p:
#                         continue
#                     if p.lower() == base_text.lower():
#                         continue
#                     if p.lower() in seen_texts:
#                         continue
#                     seen_texts.add(p.lower())
#                     out_obj = {"text": p, "intent": entry.get("intent"), "slots": entry.get("slots")}
#                     fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

#         with open(INPUT_FILE, "r", encoding="utf-8") as fin:
#             for line in fin:
#                 entry = json.loads(line)
#                 t = entry["text"].strip()
#                 if t.lower() in seen_texts:
#                     continue
#                 fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
#                 seen_texts.add(t.lower())

#     print(f"Done. Wrote augmented results to {OUTPUT_FILE.resolve()} (unique entries: {len(seen_texts)})")

# if __name__ == "__main__":
#     main()
