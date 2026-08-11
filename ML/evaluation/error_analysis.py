"""Error analysis for a trained NLU model (Phase 6F).

Runs the model over a sample of the dataset and reports where it struggles:
intent accuracy, the most common intent confusions, the confidence
distribution, the lowest-confidence utterances, and slot mismatches. Saves a
JSON report under ``ML/evaluation/results/error_analysis/`` and prints a
summary.

A committable, lint-clean stand-in for the "error analysis notebook" (repo
gitignores ``*.ipynb``). Needs torch + transformers + a trained model, so it is
not run in CI.

Usage::

    python ML/evaluation/error_analysis.py [version] [--limit N] [--dataset PATH]
    python ML/evaluation/error_analysis.py 10 --limit 1000
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, ".."))

from utils import find_latest_version_path, get_latest_model_path  # noqa: E402
from vimaan_nlu import normalize_slot_value, predict  # noqa: E402
from vimaan_nlu.model_loader import ModelLoader  # noqa: E402


def _norm_slots(slots):
    return {k: normalize_slot_value(v) for k, v in (slots or {}).items()}


def main():
    parser = argparse.ArgumentParser(description="Error analysis for a trained NLU model.")
    parser.add_argument("version", nargs="?", default=None, help="model version, e.g. 10")
    parser.add_argument("--limit", type=int, default=500, help="max utterances to sample")
    parser.add_argument("--dataset", default=None, help="path to a .jsonl dataset")
    parser.add_argument("--conf-threshold", type=float, default=0.55)
    args = parser.parse_args()

    models_dir = os.path.join(script_dir, "..", "models", "vimaan_nlu_model_best")
    if args.version:
        model_path = os.path.join(models_dir, f"v{args.version}")
    else:
        model_path = get_latest_model_path()
    if not model_path or not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        raise SystemExit(2)

    dataset = args.dataset or find_latest_version_path(
        os.path.join(
            script_dir,
            "..",
            "datasets",
            "05_final_merged",
            "aviation_cmds_final_training_set.jsonl",
        )
    )
    if not (dataset and os.path.exists(dataset)):
        print(f"ERROR: dataset not found: {dataset}")
        raise SystemExit(2)

    print(f"Model:   {model_path}")
    print(f"Dataset: {dataset}")
    loader = ModelLoader()
    loader.load_all(model_path)

    with open(dataset) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(42)
    rng.shuffle(rows)
    rows = rows[: args.limit]
    print(f"Sampled {len(rows)} utterances (seed=42)\n")

    correct = 0
    confusion = Counter()  # (gold_intent, pred_intent) -> count
    low_conf = []  # (conf, text, gold, pred)
    slot_mismatches = []  # dicts
    confidences = []

    for row in rows:
        text = row["text"]
        gold_intent = row["intent"]
        gold_slots = _norm_slots(row.get("slots", {}))
        r = predict(
            text,
            loader.model,
            loader.tokenizer,
            loader.device,
            loader.intent_map_rev,
            loader.slot_map_rev,
        )
        pred_intent = r["intent"]
        conf = float(r["confidence"])
        pred_slots = _norm_slots(r["slots"])
        confidences.append(conf)

        if pred_intent == gold_intent:
            correct += 1
            if gold_slots != pred_slots:
                slot_mismatches.append(
                    {"text": text, "intent": gold_intent, "gold": gold_slots, "pred": pred_slots}
                )
        else:
            confusion[(gold_intent, pred_intent)] += 1

        if conf < args.conf_threshold:
            low_conf.append((round(conf, 3), text, gold_intent, pred_intent))

    n = len(rows) or 1
    buckets = Counter()
    for c in confidences:
        lo = min(int(c * 10), 9) * 10  # clamp so conf==1.0 lands in 90-100%, not 100-110%
        buckets[f"{lo}-{lo + 10}%"] += 1

    report = {
        "model_path": model_path,
        "dataset": dataset,
        "sample_size": len(rows),
        "intent_accuracy": round(correct / n, 4),
        "mean_confidence": round(sum(confidences) / n, 4),
        "top_confusions": [
            {"gold": g, "pred": p, "count": c} for (g, p), c in confusion.most_common(15)
        ],
        "confidence_buckets": dict(sorted(buckets.items())),
        "lowest_confidence": [
            {"conf": c, "text": t, "gold": g, "pred": p} for c, t, g, p in sorted(low_conf)[:25]
        ],
        "slot_mismatch_count": len(slot_mismatches),
        "slot_mismatch_examples": slot_mismatches[:25],
    }

    out_dir = os.path.join(script_dir, "results", "error_analysis")
    os.makedirs(out_dir, exist_ok=True)
    version = os.path.basename(model_path)
    out_path = os.path.join(out_dir, f"error_analysis_{version}.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(
        f"Intent accuracy: {report['intent_accuracy']:.2%}  "
        f"(mean conf {report['mean_confidence']:.2%})"
    )
    print(f"Slot mismatches: {report['slot_mismatch_count']}")
    print("\nTop intent confusions (gold -> pred):")
    for row in report["top_confusions"][:10]:
        print(f"  {row['count']:>3}  {row['gold']} -> {row['pred']}")
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
