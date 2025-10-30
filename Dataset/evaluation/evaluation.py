import json
from pathlib import Path
from statistics import mean
from typing import List, Dict
import random
from tabulate import tabulate

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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

        print(f"✅ Loaded dataset from {dataset_path.name} with {len(self.data)} entries.")

        self._semantic_model = None
        self._fluency_model = None
        self._fluency_tokenizer = None

    def _load_dataset(self) -> List[Dict[str, str]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # ------------------------------------------------------------------
    # Semantic Similarity Evaluation
    # ------------------------------------------------------------------
    def evaluate_semantic(self, reference_texts: List[str]) -> float:
        if not self._semantic_model:
            self._semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)

        scores = []
        for entry in tqdm(self.data, desc=f"🔍 Semantic: {self.dataset_path.stem}"):
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
        if not self._fluency_model:
            self._fluency_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._fluency_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

        perplexities = []
        for entry in tqdm(self.data, desc=f"💬 Fluency: {self.dataset_path.stem}"):
            text = entry["text"]
            encodings = self._fluency_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self._fluency_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)

        avg_ppl = mean(perplexities)
        normalized_score = max(0.0, min(1.0, 100 / (avg_ppl + 1)))
        return round(normalized_score, 4)

    # ------------------------------------------------------------------
    # Lexical Diversity
    # ------------------------------------------------------------------
    def evaluate_diversity(self) -> float:
        all_tokens = []
        for entry in self.data:
            all_tokens.extend(entry["text"].split())
        diversity = len(set(all_tokens)) / len(all_tokens)
        return round(diversity, 4)

    # ------------------------------------------------------------------
    # Run Benchmark
    # ------------------------------------------------------------------
    def run_benchmark(self, reference_texts: List[str]) -> Dict[str, float]:
        print(f"\n🚀 Running PES Benchmark for {self.dataset_path.stem} ...")
        results = {
            "Semantic_Similarity": self.evaluate_semantic(reference_texts),
            "Fluency": self.evaluate_fluency(),
            "Lexical_Diversity": self.evaluate_diversity(),
        }
        return results


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dataset paths
    BASE_PATH = Path(r"D:\\Project-Vimaan\\Dataset\\output\\base_cmds.jsonl")
    PARROT_PATH = Path(r"D:\\Project-Vimaan\\Dataset\\output\\parrot_cmds.jsonl")
    PEGASUS_PATH = Path(r"D:\\Project-Vimaan\Dataset\\output\\pegasus_cmds.jsonl")
    FLANT5_PATH = Path(r"D:\\Project-Vimaan\Dataset\\output\\flant5_cmds.jsonl")

    REFERENCE_PATH = BASE_PATH

    datasets = {
        "Base": BASE_PATH,
        "Parrot": PARROT_PATH,
        "Pegasus": PEGASUS_PATH,
        "FLAN-T5": FLANT5_PATH,
    }

    # Load reference texts
    with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
        refs = [json.loads(line)["text"] for line in f]

    # Run PES for each dataset
    results_summary = {}
    for name, path in datasets.items():
        if not path.exists():
            print(f"⚠️ Skipping {name}: file not found ({path})")
            continue
        pes = PES(path)
        results_summary[name] = pes.run_benchmark(refs)

    # ------------------------------------------------------------------
    # Print Comparison Table
    # ------------------------------------------------------------------
    print("\n📊 OVERALL COMPARISON RESULTS:")
    metrics = ["Semantic_Similarity", "Fluency", "Lexical_Diversity"]
    table = []
    headers = ["Metric"] + list(results_summary.keys())

    for metric in metrics:
        row = [metric]
        for ds_name in results_summary.keys():
            row.append(results_summary[ds_name].get(metric, "—"))
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    # ------------------------------------------------------------------
    # Save JSON Summary
    # ------------------------------------------------------------------
    # eval_dir = Path(r"D:\Project-Vimaan\Dataset\evaluation")
    # eval_dir.mkdir(parents=True, exist_ok=True)
    # json_path = eval_dir / "pes_benchmark_all.json"
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(results_summary, f, indent=2)
    # print(f"✅ JSON report saved → {json_path.resolve()}")

    # ------------------------------------------------------------------
    # Generate HTML Report
    # ------------------------------------------------------------------

    html = """
    <html>
    <head>
        <title>Project Vimaan - PES Benchmark Report</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 40px; }
            h1 { color: #222; }
            table { border-collapse: collapse; width: 80%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
            th { background-color: #333; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>🚀 Project Vimaan — PES Benchmark Report</h1>
        <table>
            <tr>
                <th>Metric</th>""" + "".join(f"<th>{name}</th>" for name in results_summary.keys()) + "</tr>"

    for metric in metrics:
        html += f"<tr><td>{metric}</td>"
        for name in results_summary.keys():
            val = results_summary[name].get(metric, "—")
            html += f"<td>{val}</td>"
        html += "</tr>"

    html += """
        </table>
        <p style='margin-top:20px;'>✅ Generated automatically by Project Vimaan PES Evaluator.</p>
    </body>
    </html>
    """

    html_path = Path(r"D:\\Project-Vimaan\\Dataset\\evaluation") / "pes_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📝 HTML report generated → {html_path.resolve()}")
