import json
import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluator import ModelEvaluator


class BatchEvaluator:
    def __init__(self, models_dir=None):
        if models_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, "..", "models", "vimaan_nlu_model_best")

        self.models_dir = models_dir
        self.results = {}
        self.output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def find_all_versions(self):
        if not os.path.exists(self.models_dir):
            print(f"ERROR: Models directory not found: {self.models_dir}")
            return []

        versions = []
        for item in sorted(os.listdir(self.models_dir)):
            if item.startswith("v") and os.path.isdir(os.path.join(self.models_dir, item)):
                try:
                    version_num = int(item[1:])
                    model_path = os.path.join(self.models_dir, item)
                    versions.append((version_num, model_path))
                except ValueError:
                    pass

        versions.sort()
        print(f"\n{'=' * 80}")
        print(f"FOUND {len(versions)} MODEL VERSIONS")
        print(f"{'=' * 80}")
        for ver, path in versions:
            print(f"  v{ver}: {path}")

        return versions

    def evaluate_version(self, version_num, model_path, dataset_path):
        print(f"\n\n{'=' * 80}")
        print(f"EVALUATING MODEL v{version_num}")
        print(f"{'=' * 80}")

        try:
            evaluator = ModelEvaluator(model_path=model_path)
            evaluator.load_dataset(dataset_path)

            metrics, predictions, ground_truth, confidences = evaluator.evaluate_dataset(
                evaluator.test_data, dataset_name="Test"
            )

            evaluator.print_metrics_summary(metrics)
            evaluator.save_metrics(metrics, output_dir=self.output_dir)

            self.results[version_num] = {
                "model_path": model_path,
                "metrics": metrics,
                "status": "success",
            }

            print(f"\nv{version_num} evaluation COMPLETE")
            return True

        except Exception as e:
            print(f"\nv{version_num} evaluation FAILED")
            print(f"Error: {str(e)}")
            traceback.print_exc()

            self.results[version_num] = {
                "model_path": model_path,
                "error": str(e),
                "status": "failed",
            }
            return False

    def create_comparison_table(self):
        print(f"\n\n{'=' * 120}")
        print("CROSS-VERSION COMPARISON TABLE")
        print(f"{'=' * 120}\n")

        print(
            f"{'Version':<10} "
            f"{'Accuracy':<12} "
            f"{'Weighted F1':<13} "
            f"{'Macro F1':<12} "
            f"{'Kappa':<10} "
            f"{'Conf':<8} "
            f"{'InfTime':<10} "
            f"{'Intents':<10} "
            f"{'Status':<10}"
        )
        print("-" * 120)

        for version in sorted(self.results.keys()):
            result = self.results[version]

            if result["status"] == "failed":
                print(f"v{version:<9} {'FAILED':<100}")
            else:
                m = result["metrics"]
                print(
                    f"v{version:<9} "
                    f"{m['overall_accuracy']:<12.2%} "
                    f"{m['weighted_f1']:<13.4f} "
                    f"{m['macro_f1']:<12.4f} "
                    f"{m['cohens_kappa']:<10.4f} "
                    f"{m['confidence']['mean']:<8.4f} "
                    f"{m['inference_time_ms']['mean']:<10.2f} "
                    f"{m['dataset_stats']['unique_intents']:<10} "
                    f"{'Check':<10}"
                )

    def create_progress_visualization(self):
        print(f"\n\n{'=' * 80}")
        print("PROGRESS VISUALIZATION")
        print(f"{'=' * 80}\n")

        successful = [v for v, r in self.results.items() if r["status"] == "success"]

        print(f"Total: {len(self.results)} versions | Successful: {len(successful)}\n")

        print("ACCURACY PROGRESSION:")
        print("-" * 60)
        for version in sorted(successful):
            acc = self.results[version]["metrics"]["overall_accuracy"]
            bar_length = int(acc * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"v{version:>2}: {bar} {acc:.2%}")

        print("\n\nWEIGHTED F1 PROGRESSION:")
        print("-" * 60)
        for version in sorted(successful):
            f1 = self.results[version]["metrics"]["weighted_f1"]
            bar_length = int(f1 * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"v{version:>2}: {bar} {f1:.4f}")

    def create_rankings(self):
        print(f"\n\n{'=' * 80}")
        print("MODEL RANKINGS")
        print(f"{'=' * 80}\n")

        successful = {v: r for v, r in self.results.items() if r["status"] == "success"}

        if not successful:
            print("No successful evaluations to rank.")
            return

        best_acc = max(successful.items(), key=lambda x: x[1]["metrics"]["overall_accuracy"])
        print("Best Accuracy:")
        print(f"   v{best_acc[0]}: {best_acc[1]['metrics']['overall_accuracy']:.2%}")

        best_f1 = max(successful.items(), key=lambda x: x[1]["metrics"]["weighted_f1"])
        print("\nBest Weighted F1:")
        print(f"   v{best_f1[0]}: {best_f1[1]['metrics']['weighted_f1']:.4f}")

        fastest = min(
            successful.items(), key=lambda x: x[1]["metrics"]["inference_time_ms"]["mean"]
        )
        print("\nFastest Inference:")
        print(f"   v{fastest[0]}: {fastest[1]['metrics']['inference_time_ms']['mean']:.2f}ms")

        most_conf = max(successful.items(), key=lambda x: x[1]["metrics"]["confidence"]["mean"])
        print("\nHighest Confidence:")
        print(f"   v{most_conf[0]}: {most_conf[1]['metrics']['confidence']['mean']:.4f}")

    def save_comparison_json(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_all_versions_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        comparison_data = {
            "timestamp": timestamp,
            "total_versions": len(self.results),
            "successful": sum(1 for r in self.results.values() if r["status"] == "success"),
            "versions": {},
        }

        for version, result in sorted(self.results.items()):
            if result["status"] == "success":
                m = result["metrics"]
                comparison_data["versions"][f"v{version}"] = {
                    "accuracy": m["overall_accuracy"],
                    "weighted_f1": m["weighted_f1"],
                    "macro_f1": m["macro_f1"],
                    "cohens_kappa": m["cohens_kappa"],
                    "avg_confidence": m["confidence"]["mean"],
                    "inference_time_ms": m["inference_time_ms"]["mean"],
                    "dataset_size": m["dataset_stats"]["total_examples"],
                    "unique_intents": m["dataset_stats"]["unique_intents"],
                }
            else:
                comparison_data["versions"][f"v{version}"] = {
                    "status": "failed",
                    "error": result.get("error", "Unknown"),
                }

        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f"\n\nComparison JSON saved to: {filepath}")
        return filepath

    def run_all(self, dataset_path):
        """Main workflow: evaluate all versions and generate reports"""
        versions = self.find_all_versions()

        if not versions:
            print("No model versions found!")
            return

        print(f"\n\nStarting batch evaluation of {len(versions)} versions...\n")

        for version_num, model_path in versions:
            self.evaluate_version(version_num, model_path, dataset_path)

        self.create_comparison_table()
        self.create_progress_visualization()
        self.create_rankings()
        self.save_comparison_json()

        print(f"\n\n{'=' * 80}")
        print("BATCH EVALUATION COMPLETE!")
        print(f"{'=' * 80}\n")
