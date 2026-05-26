import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.batch_evaluator import BatchEvaluator
from utils import find_latest_version_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "datasets", "05_final_merged")
    base_filename = os.path.join(data_dir, "aviation_cmds_final_training_set.jsonl")
    
    dataset_path = find_latest_version_path(base_filename)
    
    if not dataset_path:
        print(f"ERROR: Dataset not found in {data_dir}")
        return
    
    print(f"Using dataset: {dataset_path}")
    
    batch_eval = BatchEvaluator()
    batch_eval.run_all(dataset_path)


if __name__ == "__main__":
    main()
