import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.evaluator import ModelEvaluator
from utils import find_latest_version_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "datasets", "05_final_merged")
    base_filename = os.path.join(data_dir, "aviation_cmds_final_training_set.jsonl")

    dataset_path = find_latest_version_path(base_filename)

    if not dataset_path:
        print(f"ERROR: Dataset not found in {data_dir}")
        return

    print(f"Using dataset: {dataset_path}\n")

    model_path = None
    if len(sys.argv) > 1:
        version = sys.argv[1]
        models_dir = os.path.join(script_dir, "..", "models", "vimaan_nlu_model_best")
        model_path = os.path.join(models_dir, f"v{version}")

        if not os.path.exists(model_path):
            print(f"ERROR: Model v{version} not found at {model_path}")
            return

        print(f"Evaluating specified version: v{version}")
    else:
        print("Evaluating latest model version")

    evaluator = ModelEvaluator(model_path=model_path)
    evaluator.load_dataset(dataset_path)

    metrics, predictions, ground_truth, confidences = evaluator.evaluate_dataset(
        evaluator.test_data, dataset_name="Test"
    )

    evaluator.print_metrics_summary(metrics)

    evaluator.save_metrics(metrics)

    print(f"\n\n{'=' * 70}")
    print("EVALUATION COMPLETE!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
