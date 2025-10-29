import json
from pathlib import Path
from statistics import mean
from typing import List, Dict
import random
from tabulate import tabulate

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer, util


class PES:
    """
    Project Vimaan Paraphrase Evaluation System (PES)
    Combines Semantic, Fluency, and Diversity evaluations.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = self._load_dataset()

        print(f"✅ Loaded dataset with {len(self.data)} entries.")

        # Lazy initialization
        self._semantic_model = None
        self._fluency_model = None
        self._fluency_tokenizer = None

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    def _load_dataset(self) -> List[Dict[str, str]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # ------------------------------------------------------------------
    # Semantic Similarity Evaluation
    # ------------------------------------------------------------------
    def evaluate_semantic(self, reference_texts: List[str]) -> float:
        """
        Compute average semantic similarity using Sentence-BERT.
        """
        if not self._semantic_model:
            self._semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)

        scores = []
        for entry in tqdm(self.data, desc="🔍 Evaluating Semantic Similarity"):
            paraphrase = entry["text"]
            ref = random.choice(reference_texts)
            emb1 = self._semantic_model.encode(paraphrase, convert_to_tensor=True)
            emb2 = self._semantic_model.encode(ref, convert_to_tensor=True)
            score = util.cos_sim(emb1, emb2).item()
            scores.append(score)

        return round(mean(scores), 4)

    # ------------------------------------------------------------------
    # Fluency Evaluation (GPT-2 Perplexity)
    # ------------------------------------------------------------------
    def evaluate_fluency(self) -> float:
        """
        Compute average fluency using perplexity (GPT-2).
        """
        if not self._fluency_model:
            self._fluency_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._fluency_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

        perplexities = []
        for entry in tqdm(self.data, desc="💬 Evaluating Fluency"):
            text = entry["text"]
            encodings = self._fluency_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._fluency_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)

        avg_ppl = mean(perplexities)
        normalized_score = max(0.0, min(1.0, 100 / (avg_ppl + 1)))  # normalize 0–1
        return round(normalized_score, 4)

    # ------------------------------------------------------------------
    # Lexical Diversity Evaluation
    # ------------------------------------------------------------------
    def evaluate_diversity(self) -> float:
        """
        Compute lexical diversity = unique tokens / total tokens.
        """
        all_tokens = []
        for entry in self.data:
            all_tokens.extend(entry["text"].split())

        diversity = len(set(all_tokens)) / len(all_tokens)
        return round(diversity, 4)

    # ------------------------------------------------------------------
    # Run Benchmark
    # ------------------------------------------------------------------
    def run_benchmark(self, reference_texts: List[str]) -> Dict[str, float]:
        """
        Run all metrics and return combined results.
        """
        print("\n🚀 Running PES Benchmark...")
        semantic = self.evaluate_semantic(reference_texts)
        fluency = self.evaluate_fluency()
        diversity = self.evaluate_diversity()

        results = {
            "Semantic_Similarity": semantic,
            "Fluency": fluency,
            "Lexical_Diversity": diversity,
        }

        print("\n📊 PES Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")

        return results


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":

    BASE_PATH = Path(r"D:\Project-Vimaan\Dataset\output\base_cmds.jsonl")
    AUG_PATH = Path(r"D:\Project-Vimaan\Dataset\output\parrot_cmds.jsonl")
    REFERENCE_PATH = BASE_PATH  # reference remains the same

    if not BASE_PATH.exists() or not AUG_PATH.exists():
        raise FileNotFoundError("❌ Base or augmented dataset not found!")

    # Load reference texts
    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        refs = [json.loads(line)["text"] for line in f]

    # Evaluate Base dataset
    print("\n🏁 Evaluating BASE dataset...")
    base_pes = PES(BASE_PATH)
    base_results = base_pes.run_benchmark(refs)

    # Evaluate Augmented dataset
    print("\n⚙️ Evaluating AUGMENTED dataset...")
    aug_pes = PES(AUG_PATH)
    aug_results = aug_pes.run_benchmark(refs)

    # Compare Results
    print("\n📊 COMPARISON RESULTS:")
    rows = []
    for k in base_results.keys():
        base = base_results[k]
        aug = aug_results[k]
        diff = round(aug - base, 4)
        symbol = "✅" if diff > 0 else ("⚠️" if diff == 0 else "❌")
        rows.append([k, base, aug, diff, symbol])

    print(tabulate(rows, headers=["Metric", "Base", "Augmented", "Change", "Status"], tablefmt="fancy_grid"))

    # Save results
    out_path = Path(r"D:\Project-Vimaan\Dataset\evaluation\pes_benchmark.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base": base_results, "augmented": aug_results}, f, indent=2)

    print(f"\n✅ Comparison complete → {out_path.resolve()}")