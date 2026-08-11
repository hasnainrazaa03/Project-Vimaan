import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import get_latest_model_path
from vimaan_nlu import normalize_slot_value
from vimaan_nlu.inference import predict
from vimaan_nlu.model_loader import ModelLoader


class ModelEvaluator:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = ModelLoader(self.device)

        if model_path is None:
            model_path = get_latest_model_path()

        self.model_path = model_path
        self.model_version = self._extract_version()
        print(f"Loading model from: {model_path}")

        try:
            self.results = self.loader.load_all(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model with tokenizer: {str(e)}")
            print(f"This is likely v{self.model_version} has missing tokenizer files")
            print("Attempting to load with default tokenizer...")

            try:
                self.results = {
                    "model": self.loader.load_model(model_path),
                    "tokenizer": {"tokenizer_loaded": True},
                    "maps": {
                        "intents": len(self.loader.intent_map),
                        "slots": len(self.loader.slot_map),
                    },
                }
                print("Model loaded (with default tokenizer)")
                from transformers import DistilBertTokenizerFast

                self.loader.tokenizer = DistilBertTokenizerFast.from_pretrained(
                    "distilbert-base-uncased"
                )
                print("Default tokenizer loaded")
            except Exception as e2:
                print(f"Failed to load model: {str(e2)}")
                raise

        print(f"Model Version: {self.model_version}")
        print(f"Intents: {self.results['maps']['intents']}")
        print(f"Slots: {self.results['maps']['slots']}\n")

    def _extract_version(self):
        if "v" in self.model_path:
            parts = self.model_path.split("v")
            if len(parts) > 1:
                version_str = parts[-1].split("/")[0].split("\\")[0]
                try:
                    return int(version_str)
                except:
                    return version_str
        return "unknown"

    def load_dataset(self, dataset_path):
        print(f"Loading dataset from: {dataset_path}")

        with open(dataset_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        print(f"Total examples: {len(data)}")

        # Prefer a true held-out test set saved alongside the model. Older models
        # have none — the previous code sliced the last 15% of the SAME file the
        # (shuffled, seed-42) training run consumed in full, so ~85% of the "test"
        # rows were actually trained on. Reproduce that split and evaluate on the
        # VALIDATION partition instead (used for early stopping — mild leakage,
        # but not directly trained on) and label it honestly.
        holdout = os.path.join(self.model_path, "holdout_test.jsonl") if self.model_path else None
        if holdout and os.path.isfile(holdout):
            with open(holdout, encoding="utf-8") as f:
                self.test_data = [json.loads(line) for line in f if line.strip()]
            self.eval_set_kind = "held-out test set (saved with the model)"
        else:
            _train, self.test_data = train_test_split(data, test_size=0.15, random_state=42)
            self.eval_set_kind = (
                "validation split (legacy model — reproduced seed-42 training "
                "split; NOT a true held-out test)"
            )

        print(f"Eval set: {len(self.test_data)} examples  [{self.eval_set_kind}]\n")

    def evaluate_dataset(self, dataset, dataset_name="Test"):
        print(f"\n{'=' * 70}")
        print(f"EVALUATING ON {dataset_name.upper()} SET ({len(dataset)} examples)")
        print(f"{'=' * 70}\n")

        all_predictions = []
        all_ground_truth = []
        all_confidences = []
        inference_times = []
        all_pred_slots = []
        all_gold_slots = []

        for item in tqdm(dataset, desc="Evaluating", unit="example"):
            text = item["text"]
            true_intent = item["intent"]

            start_time = time.time()
            result = predict(
                text,
                self.loader.model,
                self.loader.tokenizer,
                self.device,
                self.loader.intent_map_rev,
                self.loader.slot_map_rev,
                do_postprocess=True,
            )
            inference_time = (time.time() - start_time) * 1000

            all_predictions.append(result["intent"])
            all_ground_truth.append(true_intent)
            all_confidences.append(result["confidence"])
            all_pred_slots.append(result.get("slots") or {})
            all_gold_slots.append(item.get("slots") or {})
            inference_times.append(inference_time)

        metrics = self._calculate_metrics(
            all_ground_truth,
            all_predictions,
            all_confidences,
            inference_times,
            all_pred_slots,
            all_gold_slots,
        )

        return metrics, all_predictions, all_ground_truth, all_confidences

    @staticmethod
    def _slot_pair_metrics(pred_slots, gold_slots):
        """Micro precision/recall/F1 over (slot_name, normalized_value) pairs.

        Both sides are normalized (`normalize_slot_value`) so word/digit forms
        compare equal. This catches slot regressions the intent metrics miss.
        """
        tp = fp = fn = 0
        for pred, gold in zip(pred_slots, gold_slots):
            pred_pairs = {(k, normalize_slot_value(v)) for k, v in pred.items()}
            gold_pairs = {(k, normalize_slot_value(v)) for k, v in gold.items()}
            tp += len(pred_pairs & gold_pairs)
            fp += len(pred_pairs - gold_pairs)
            fn += len(gold_pairs - pred_pairs)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "slot_precision": float(precision),
            "slot_recall": float(recall),
            "slot_f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def _calculate_metrics(
        self, y_true, y_pred, confidences, inference_times, pred_slots=None, gold_slots=None
    ):
        metrics = {}
        metrics["eval_set_kind"] = getattr(self, "eval_set_kind", "unknown")

        if not y_true:
            metrics["error"] = "empty evaluation set"
            return metrics

        metrics["overall_accuracy"] = accuracy_score(y_true, y_pred)

        labels = sorted(set(y_true + y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=labels
        )

        metrics["per_intent"] = {}
        for idx, intent in enumerate(labels):
            metrics["per_intent"][intent] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }

        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["weighted_f1"] = float(f1_weighted)

        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["macro_f1"] = float(f1_macro)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}

        if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
            metrics["cohens_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        else:
            metrics["cohens_kappa"] = 0.0

        metrics["confidence"] = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences)),
        }

        metrics["inference_time_ms"] = {
            "mean": float(np.mean(inference_times)),
            "std": float(np.std(inference_times)),
            "min": float(np.min(inference_times)),
            "max": float(np.max(inference_times)),
            "p95": float(np.percentile(inference_times, 95)),
        }

        metrics["dataset_stats"] = {
            "total_examples": len(y_true),
            "unique_intents": len(set(y_true)),
            "correct_predictions": sum(1 for t, p in zip(y_true, y_pred) if t == p),
            "incorrect_predictions": sum(1 for t, p in zip(y_true, y_pred) if t != p),
        }

        if pred_slots is not None and gold_slots is not None:
            metrics["slots"] = self._slot_pair_metrics(pred_slots, gold_slots)

        return metrics

    def save_metrics(self, metrics, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "results")

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_v{self.model_version}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        metrics["metadata"] = {
            "model_version": self.model_version,
            "model_path": self.model_path,
            "timestamp": timestamp,
            "device": str(self.device),
        }

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetrics saved to: {filepath}")
        return filepath

    def print_metrics_summary(self, metrics):
        print(f"\n{'=' * 70}")
        print(f"MODEL v{self.model_version} - EVALUATION SUMMARY")
        print(f"{'=' * 70}\n")
        print(f"Eval set: {metrics.get('eval_set_kind', 'unknown')}\n")

        if metrics.get("error"):
            print(f"  {metrics['error']}")
            return

        print("OVERALL METRICS")
        print("-" * 70)
        print(f"  Accuracy:           {metrics['overall_accuracy']:.2%}")
        print(f"  Weighted F1-Score:  {metrics['weighted_f1']:.4f}")
        print(f"  Macro F1-Score:     {metrics['macro_f1']:.4f}")
        print(f"  Cohen's Kappa:      {metrics['cohens_kappa']:.4f}")

        if "slots" in metrics:
            s = metrics["slots"]
            print("\nSLOT METRICS (micro over name-value pairs)")
            print("-" * 70)
            print(
                f"  Precision: {s['slot_precision']:.2%} | "
                f"Recall: {s['slot_recall']:.2%} | F1: {s['slot_f1']:.4f}"
            )
            print(f"  TP {s['tp']}  FP {s['fp']}  FN {s['fn']}")

        print("\nDATASET STATISTICS")
        print("-" * 70)
        stats = metrics["dataset_stats"]
        print(f"  Total Examples:     {stats['total_examples']}")
        print(
            f"  Correct:            {stats['correct_predictions']} ({stats['correct_predictions'] / stats['total_examples']:.1%})"
        )
        print(
            f"  Incorrect:          {stats['incorrect_predictions']} ({stats['incorrect_predictions'] / stats['total_examples']:.1%})"
        )
        print(f"  Unique Intents:     {stats['unique_intents']}")

        print("\nCONFIDENCE METRICS")
        print("-" * 70)
        conf = metrics["confidence"]
        print(f"  Mean:               {conf['mean']:.4f}")
        print(f"  Std Dev:            {conf['std']:.4f}")
        print(f"  Range:              [{conf['min']:.4f}, {conf['max']:.4f}]")
        print(f"  Median:             {conf['median']:.4f}")

        print("\nINFERENCE TIME (milliseconds)")
        print("-" * 70)
        inf = metrics["inference_time_ms"]
        print(f"  Mean:               {inf['mean']:.2f}ms")
        print(f"  Std Dev:            {inf['std']:.2f}ms")
        print(f"  95th Percentile:    {inf['p95']:.2f}ms")
        print(f"  Range:              [{inf['min']:.2f}ms, {inf['max']:.2f}ms]")

        print("\nPER-INTENT PERFORMANCE (Top 10 by support)")
        print("-" * 70)

        sorted_intents = sorted(
            metrics["per_intent"].items(), key=lambda x: x[1]["support"], reverse=True
        )[:10]

        for intent, scores in sorted_intents:
            print(f"\n  {intent}")
            print(
                f"    Precision: {scores['precision']:.2%} | Recall: {scores['recall']:.2%} | F1: {scores['f1']:.4f}"
            )
            print(f"    Support: {scores['support']} examples")
