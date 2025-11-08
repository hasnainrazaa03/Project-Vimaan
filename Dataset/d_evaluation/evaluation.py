# import json
# from pathlib import Path
# from statistics import mean
# from typing import List, Dict
# import random
# from tabulate import tabulate

# import torch
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util


# class PES:
#     """
#     Project Vimaan Paraphrase Evaluation System (PES)
#     Combines Semantic, Fluency, and Diversity evaluations.
#     """

#     def __init__(self, dataset_path: Path):
#         self.dataset_path = dataset_path
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.data = self._load_dataset()

#         print(f"✅ Loaded dataset from {dataset_path.name} with {len(self.data)} entries.")

#         self._semantic_model = None
#         self._fluency_model = None
#         self._fluency_tokenizer = None

#     def _load_dataset(self) -> List[Dict[str, str]]:
#         with open(self.dataset_path, "r", encoding="utf-8") as f:
#             return [json.loads(line) for line in f]

#     # ------------------------------------------------------------------
#     # Semantic Similarity Evaluation
#     # ------------------------------------------------------------------
#     def evaluate_semantic(self, reference_texts: List[str]) -> float:
#         if not self._semantic_model:
#             self._semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)

#         scores = []
#         for entry in tqdm(self.data, desc=f"🔍 Semantic: {self.dataset_path.stem}"):
#             paraphrase = entry["text"]
#             ref = random.choice(reference_texts)
#             emb1 = self._semantic_model.encode(paraphrase, convert_to_tensor=True)
#             emb2 = self._semantic_model.encode(ref, convert_to_tensor=True)
#             score = util.cos_sim(emb1, emb2).item()
#             scores.append(score)
#         return round(mean(scores), 4)

#     # ------------------------------------------------------------------
#     # Fluency Evaluation (GPT-2 Perplexity)
#     # ------------------------------------------------------------------
#     def evaluate_fluency(self) -> float:
#         if not self._fluency_model:
#             self._fluency_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#             self._fluency_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

#         perplexities = []
#         for entry in tqdm(self.data, desc=f"💬 Fluency: {self.dataset_path.stem}"):
#             text = entry["text"]
#             encodings = self._fluency_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
#             with torch.no_grad():
#                 outputs = self._fluency_model(**encodings, labels=encodings["input_ids"])
#             loss = outputs.loss
#             perplexity = torch.exp(loss).item()
#             perplexities.append(perplexity)

#         avg_ppl = mean(perplexities)
#         normalized_score = max(0.0, min(1.0, 100 / (avg_ppl + 1)))
#         return round(normalized_score, 4)

#     # ------------------------------------------------------------------
#     # Lexical Diversity
#     # ------------------------------------------------------------------
#     def evaluate_diversity(self) -> float:
#         all_tokens = []
#         for entry in self.data:
#             all_tokens.extend(entry["text"].split())
#         diversity = len(set(all_tokens)) / len(all_tokens)
#         return round(diversity, 4)

#     # ------------------------------------------------------------------
#     # Run Benchmark
#     # ------------------------------------------------------------------
#     def run_benchmark(self, reference_texts: List[str]) -> Dict[str, float]:
#         print(f"\n🚀 Running PES Benchmark for {self.dataset_path.stem} ...")
#         results = {
#             "Semantic_Similarity": self.evaluate_semantic(reference_texts),
#             "Fluency": self.evaluate_fluency(),
#             "Lexical_Diversity": self.evaluate_diversity(),
#         }
#         return results


# # ----------------------------------------------------------------------
# # Entry Point
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # Dataset paths
#     BASE_PATH = Path(r"D:\\Project-Vimaan\\Dataset\\output\\base_cmds.jsonl")
#     PARROT_PATH = Path(r"D:\\Project-Vimaan\\Dataset\\output\\parrot_cmds.jsonl")
#     PEGASUS_PATH = Path(r"D:\\Project-Vimaan\Dataset\\output\\pegasus_cmds.jsonl")
#     FLANT5_PATH = Path(r"D:\\Project-Vimaan\Dataset\\output\\flant5_cmds.jsonl")

#     REFERENCE_PATH = BASE_PATH

#     datasets = {
#         "Base": BASE_PATH,
#         "Parrot": PARROT_PATH,
#         "Pegasus": PEGASUS_PATH,
#         "FLAN-T5": FLANT5_PATH,
#     }

#     # Load reference texts
#     with open(REFERENCE_PATH, "r", encoding="utf-8") as f:
#         refs = [json.loads(line)["text"] for line in f]

#     # Run PES for each dataset
#     results_summary = {}
#     for name, path in datasets.items():
#         if not path.exists():
#             print(f"⚠️ Skipping {name}: file not found ({path})")
#             continue
#         pes = PES(path)
#         results_summary[name] = pes.run_benchmark(refs)

#     # ------------------------------------------------------------------
#     # Print Comparison Table
#     # ------------------------------------------------------------------
#     print("\n📊 OVERALL COMPARISON RESULTS:")
#     metrics = ["Semantic_Similarity", "Fluency", "Lexical_Diversity"]
#     table = []
#     headers = ["Metric"] + list(results_summary.keys())

#     for metric in metrics:
#         row = [metric]
#         for ds_name in results_summary.keys():
#             row.append(results_summary[ds_name].get(metric, "—"))
#         table.append(row)

#     print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

#     # ------------------------------------------------------------------
#     # Generate HTML Report
#     # ------------------------------------------------------------------

#     html = """
#     <html>
#     <head>
#         <title>Project Vimaan - PES Benchmark Report</title>
#         <style>
#             body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 40px; }
#             h1 { color: #222; }
#             table { border-collapse: collapse; width: 80%; margin-top: 20px; }
#             th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
#             th { background-color: #333; color: white; }
#             tr:nth-child(even) { background-color: #f2f2f2; }
#         </style>
#     </head>
#     <body>
#         <h1>🚀 Project Vimaan — PES Benchmark Report</h1>
#         <table>
#             <tr>
#                 <th>Metric</th>""" + "".join(f"<th>{name}</th>" for name in results_summary.keys()) + "</tr>"

#     for metric in metrics:
#         html += f"<tr><td>{metric}</td>"
#         for name in results_summary.keys():
#             val = results_summary[name].get(metric, "—")
#             html += f"<td>{val}</td>"
#         html += "</tr>"

#     html += """
#         </table>
#         <p style='margin-top:20px;'>✅ Generated automatically by Project Vimaan PES Evaluator.</p>
#     </body>
#     </html>
#     """

#     html_path = Path(r"D:\\Project-Vimaan\\Dataset\\evaluation") / "pes_report.html"
#     with open(html_path, "w", encoding="utf-8") as f:
#         f.write(html)

#     print(f"📝 HTML report generated → {html_path.resolve()}")

import json
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any
import random

import torch
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# PES with plotting/reporting
# ------------------------------
class PES:
    """
    Project Vimaan Paraphrase Evaluation System (PES)
    Combines Semantic, Fluency, and Diversity evaluations.
    Now returns per-example scores so we can visualize distributions.
    """

    def __init__(self, dataset_path: Path, device: str = None):
        self.dataset_path = dataset_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    # returns: list of cosine similarity scores (one per entry)
    # ------------------------------------------------------------------
    def evaluate_semantic(self, reference_texts: List[str]) -> List[float]:
        if not self._semantic_model:
            self._semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(self.device)

        scores = []
        for entry in tqdm(self.data, desc=f"🔍 Semantic: {self.dataset_path.stem}"):
            paraphrase = entry["text"]
            ref = random.choice(reference_texts)
            emb1 = self._semantic_model.encode(paraphrase, convert_to_tensor=True)
            emb2 = self._semantic_model.encode(ref, convert_to_tensor=True)
            score = util.cos_sim(emb1, emb2).item()
            # cosine similarity in [-1,1] — clamp to [0,1] for visualization by (score+1)/2
            scores.append(float((score + 1.0) / 2.0))
        return [round(s, 6) for s in scores]

    # ------------------------------------------------------------------
    # Fluency Evaluation (GPT-2 Perplexity)
    # returns: list of normalized fluency scores in [0,1] (higher = more fluent)
    # ------------------------------------------------------------------
    def evaluate_fluency(self) -> List[float]:
        if not self._fluency_model:
            self._fluency_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._fluency_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

        per_scores = []
        for entry in tqdm(self.data, desc=f"💬 Fluency: {self.dataset_path.stem}"):
            text = entry["text"]
            encodings = self._fluency_tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self._fluency_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            # convert to perplexity; then to a normalized fluency score
            try:
                ppl = float(torch.exp(loss).item())
            except OverflowError:
                ppl = float('inf')
            # map perplexity to [0,1] using a simple transform: score = 100 / (ppl + 1), then clamp to [0,1]
            raw = 100.0 / (ppl + 1.0) if np.isfinite(ppl) else 0.0
            norm = max(0.0, min(1.0, raw))
            per_scores.append(norm)
        return [round(s, 6) for s in per_scores]

    # ------------------------------------------------------------------
    # Lexical Diversity per-entry (type-token ratio for each paraphrase)
    # returns: list of TTRs per entry AND dataset-level overall diversity
    # ------------------------------------------------------------------
    def evaluate_diversity_per_entry(self) -> List[float]:
        per = []
        for entry in self.data:
            tokens = entry["text"].split()
            if len(tokens) == 0:
                per.append(0.0)
            else:
                per.append(len(set(tokens)) / len(tokens))
        return [round(s, 6) for s in per]

    def evaluate_overall_diversity(self) -> float:
        all_tokens = []
        for entry in self.data:
            all_tokens.extend(entry["text"].split())
        if len(all_tokens) == 0:
            return 0.0
        diversity = len(set(all_tokens)) / len(all_tokens)
        return round(diversity, 6)

    # ------------------------------------------------------------------
    # Run Benchmark -> returns both summary stats and per-entry details
    # ------------------------------------------------------------------
    def run_benchmark(self, reference_texts: List[str]) -> Dict[str, Any]:
        print(f"\n🚀 Running PES Benchmark for {self.dataset_path.stem} ...")

        semantic_scores = self.evaluate_semantic(reference_texts)
        fluency_scores = self.evaluate_fluency()
        diversity_per = self.evaluate_diversity_per_entry()
        overall_diversity = self.evaluate_overall_diversity()

        results = {
            "Semantic_Similarity": round(mean(semantic_scores) if semantic_scores else 0.0, 6),
            "Fluency": round(mean(fluency_scores) if fluency_scores else 0.0, 6),
            "Lexical_Diversity": overall_diversity,
            # detailed lists for plotting
            "details": {
                "semantic_scores": semantic_scores,
                "fluency_scores": fluency_scores,
                "diversity_per_entry": diversity_per,
            },
        }
        return results

# ------------------------------
# Plotting helpers
# ------------------------------
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    return d

def plot_histogram(data: List[float], title: str, out_path: Path, bins: int = 30):
    plt.figure(figsize=(6, 4))
    if _HAS_SEABORN:
        sns.histplot(data, kde=True, bins=bins)
    else:
        plt.hist(data, bins=bins, alpha=0.8)
    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_boxplot(groups: Dict[str, List[float]], title: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    labels = list(groups.keys())
    data = [groups[k] for k in labels]
    if _HAS_SEABORN:
        sns.boxplot(data=data)
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=30)
    else:
        plt.boxplot(data, labels=labels)
        plt.xticks(rotation=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter(x: List[float], y: List[float], title: str, out_path: Path, labels: List[str] = None):
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel("Diversity (TTR)")
    plt.ylabel("Semantic Similarity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_radar(averages: Dict[str, float], model_name: str, out_path: Path):
    # averages: {"Semantic_Similarity": val, "Fluency": val, "Lexical_Diversity": val}
    categories = list(averages.keys())
    values = [float(averages[c]) for c in categories]
    # radar requires closed loop
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories) + 1, endpoint=True)

    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title(model_name)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def generate_plots(results_summary: Dict[str, Dict[str, Any]], out_dir: Path) -> Dict[str, str]:
    """
    results_summary: {model_name: {Semantic_Similarity, Fluency, Lexical_Diversity, details: {...}}}
    Saves images into out_dir and returns mapping of image names for embedding in HTML.
    """
    ensure_dir(out_dir)
    images = {}

    # --- per-model histograms ---
    for name, res in results_summary.items():
        details = res.get("details", {})
        sem = details.get("semantic_scores", [])
        flu = details.get("fluency_scores", [])
        div = details.get("diversity_per_entry", [])

        if sem:
            p = out_dir / f"{name}_semantic_hist.png"
            plot_histogram(sem, f"{name} — Semantic Distribution", p)
            images[f"{name}_semantic_hist"] = str(p.name)
        if flu:
            p = out_dir / f"{name}_fluency_hist.png"
            plot_histogram(flu, f"{name} — Fluency Distribution", p)
            images[f"{name}_fluency_hist"] = str(p.name)
        if div:
            p = out_dir / f"{name}_diversity_hist.png"
            plot_histogram(div, f"{name} — Diversity (TTR) Distribution", p)
            images[f"{name}_diversity_hist"] = str(p.name)

        # radar summary per model
        avg_map = {
            "Semantic_Similarity": float(res.get("Semantic_Similarity", 0.0)),
            "Fluency": float(res.get("Fluency", 0.0)),
            "Lexical_Diversity": float(res.get("Lexical_Diversity", 0.0)),
        }
        p = out_dir / f"{name}_radar.png"
        plot_radar(avg_map, name, p)
        images[f"{name}_radar"] = str(p.name)

    # --- Boxplots comparing models (semantic / fluency / diversity) ---
    # gather groups
    semantic_groups = {name: res["details"]["semantic_scores"] for name, res in results_summary.items() if res.get("details","").get("semantic_scores")}
    fluency_groups = {name: res["details"]["fluency_scores"] for name, res in results_summary.items() if res.get("details","").get("fluency_scores")}
    diversity_groups = {name: res["details"]["diversity_per_entry"] for name, res in results_summary.items() if res.get("details","").get("diversity_per_entry")}

    if semantic_groups:
        p = out_dir / "compare_semantic_box.png"
        plot_boxplot(semantic_groups, "Semantic Comparison Across Models", p)
        images["compare_semantic_box"] = str(p.name)

    if fluency_groups:
        p = out_dir / "compare_fluency_box.png"
        plot_boxplot(fluency_groups, "Fluency Comparison Across Models", p)
        images["compare_fluency_box"] = str(p.name)

    if diversity_groups:
        p = out_dir / "compare_diversity_box.png"
        plot_boxplot(diversity_groups, "Diversity Comparison Across Models", p)
        images["compare_diversity_box"] = str(p.name)

    # --- Semantic vs Diversity scatter for each model ---
    for name, res in results_summary.items():
        sem = res["details"].get("semantic_scores", [])
        div = res["details"].get("diversity_per_entry", [])
        if sem and div and len(sem) == len(div):
            p = out_dir / f"{name}_sem_vs_div.png"
            plot_scatter(div, sem, f"{name} — Semantic vs Diversity", p)
            images[f"{name}_sem_vs_div"] = str(p.name)

    return images

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == "__main__":
    # Dataset paths
    BASE_PATH = Path(r"D:\\Project-Vimaan\Dataset\\output\\base_cmds.jsonl")
    PARROT_PATH = Path(r"D:\\Project-Vimaan\Dataset\\output\\parrot_cmds.jsonl")
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
    # Generate plots & HTML Report
    # ------------------------------------------------------------------
    report_dir = Path(r"D:\\Project-Vimaan\\Dataset\\evaluation")
    ensure_dir(report_dir)

    images = generate_plots(results_summary, report_dir)

    # Build HTML with embedded images
    html_lines = [
        "<html>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<title>Project Vimaan - PES Benchmark Report</title>",
        "<style>",
        "body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 24px; }",
        "h1 { color: #222; }",
        ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }",
        ".card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.08);} ",
        "img { max-width: 100%; height: auto; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 12px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }",
        "th { background-color: #333; color: white; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>🚀 Project Vimaan — PES Benchmark Report</h1>",
        "<p>Generated automatically with per-example distributions and model comparisons.</p>",
        "<h2>Summary Table</h2>",
        "<table>",
        "<tr><th>Metric</th>" + "".join(f"<th>{name}</th>" for name in results_summary.keys()) + "</tr>"
    ]

    for metric in metrics:
        html_lines.append("<tr>")
        html_lines.append(f"<td>{metric}</td>")
        for name in results_summary.keys():
            val = results_summary[name].get(metric, "—")
            html_lines.append(f"<td>{val}</td>")
        html_lines.append("</tr>")
    html_lines.append("</table>")

    # Images grid
    html_lines.append("<h2>Visualizations</h2>")
    html_lines.append("<div class='grid'>")
    # insert each image file found
    for key, filename in images.items():
        html_lines.append("<div class='card'>")
        html_lines.append(f"<h3>{key.replace('_', ' ').title()}</h3>")
        # use relative path from report_dir
        html_lines.append(f"<img src='{filename}' alt='{key}'/>")
        html_lines.append("</div>")
    html_lines.append("</div>")

    html_lines.append("<p style='margin-top:16px;'>✅ Generated automatically by Project Vimaan PES Evaluator.</p>")
    html_lines.append("</body></html>")

    html_path = report_dir / "pes_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"📝 HTML report generated → {html_path.resolve()}")
    print(f"📁 Images saved to → {report_dir.resolve()}")
