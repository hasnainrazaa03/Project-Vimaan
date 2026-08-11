"""Dynamic int8 quantization + fp32-vs-int8 benchmark for the joint NLU model.

Phase 4B. Loads a trained DistilBERT checkpoint, applies PyTorch dynamic
quantization to its ``nn.Linear`` layers, and measures the size / latency /
accuracy trade-off so we can decide whether to ship the quantized model.

The quantized weights are *derived deterministically* from the fp32 weights via
``quantize_dynamic`` (no fine-tuning), so the runtime never needs the saved
``.bin`` -- the plugin reproduces the quantized model on the fly (see
``ModelLoader.load_all(..., quantize=True)``). We still write the artifact and a
``quant_manifest.json`` next to it for provenance and disk-size measurement.

Usage::

    python3 ML/quantize_model.py                      # latest checkpoint
    python3 ML/quantize_model.py --model-path ML/models/.../v9
    python3 ML/quantize_model.py --eval-rows 800 --latency-reps 5
"""

import argparse
import copy
import json
import os
import shutil
import statistics
import sys
import time
from datetime import UTC, datetime

import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))

from utils import find_latest_version_path, get_latest_model_path  # noqa: E402
from vimaan_nlu.inference import DEFAULT_MAX_LENGTH, predict  # noqa: E402
from vimaan_nlu.model_loader import ModelLoader  # noqa: E402

# Representative commands used for the latency benchmark.
LATENCY_COMMANDS = [
    "set heading to two seven zero",
    "tune comm one to one two one decimal five",
    "climb and maintain flight level three five zero",
    "set vertical speed to fifteen hundred feet per minute",
    "engage autopilot one",
    "lower the landing gear",
    "set the altitude to ten thousand feet",
    "contact tower on one one eight decimal one",
    "reduce speed to two one zero knots",
    "disengage the flight director",
]

# Resolve the latest merged dataset (matching the rest of the pipeline) rather
# than a hardcoded _v5_v1 file that find_latest_version_path never selects and
# that FileNotFoundErrors if removed/renamed.
DEFAULT_DATASET = find_latest_version_path(
    os.path.join(
        os.path.dirname(__file__),
        "datasets",
        "05_final_merged",
        "aviation_cmds_final_training_set.jsonl",
    )
)


def quantize(model):
    """Return an int8 dynamically-quantized copy of ``model`` (Linear layers)."""
    # On Apple Silicon the only built quantized backend is qnnpack, and it is
    # not selected by default (engine == "none"), which makes linear_prepack
    # fail. Select it explicitly before converting.
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    return torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def _fp32_weight_size_bytes(model_path):
    """Size of the fp32 weight files we would otherwise load at runtime."""
    total = 0
    for name in ("model.safetensors", "pytorch_model.bin", "intent_classifier.bin"):
        fp = os.path.join(model_path, name)
        if os.path.isfile(fp):
            total += os.path.getsize(fp)
    return total


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def benchmark_latency(model, loader, device, reps, max_length):
    """Return (p50_ms, p95_ms, mean_ms) over LATENCY_COMMANDS x reps."""
    # Warm-up so the first-call graph/cache cost is not counted.
    for cmd in LATENCY_COMMANDS[:3]:
        predict(
            cmd,
            model,
            loader.tokenizer,
            device,
            loader.intent_map_rev,
            loader.slot_map_rev,
            max_length=max_length,
        )

    timings = []
    for _ in range(reps):
        for cmd in LATENCY_COMMANDS:
            start = time.perf_counter()
            predict(
                cmd,
                model,
                loader.tokenizer,
                device,
                loader.intent_map_rev,
                loader.slot_map_rev,
                max_length=max_length,
            )
            timings.append((time.perf_counter() - start) * 1000.0)

    return _percentile(timings, 50), _percentile(timings, 95), statistics.mean(timings)


def _slot_pair_f1(true_rows, pred_rows):
    """Micro F1 over (slot_name, value) pairs across all examples."""
    tp = fp = fn = 0
    for true_slots, pred_slots in zip(true_rows, pred_rows):
        true_set = {(k, str(v)) for k, v in true_slots.items()}
        pred_set = {(k, str(v)) for k, v in pred_slots.items()}
        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def evaluate_accuracy(model, loader, device, rows, max_length):
    """Return dict with intent macro-F1, intent accuracy, slot pair-F1."""
    true_intents, pred_intents = [], []
    true_slots, pred_slots = [], []
    correct = 0

    for row in rows:
        result = predict(
            row["text"],
            model,
            loader.tokenizer,
            device,
            loader.intent_map_rev,
            loader.slot_map_rev,
            max_length=max_length,
        )
        true_intents.append(row["intent"])
        pred_intents.append(result["intent"])
        correct += int(result["intent"] == row["intent"])
        true_slots.append(row.get("slots", {}) or {})
        pred_slots.append(result.get("slots", {}) or {})

    labels = sorted(set(true_intents) | set(pred_intents))
    return {
        "intent_macro_f1": float(
            f1_score(true_intents, pred_intents, labels=labels, average="macro", zero_division=0)
        ),
        "intent_accuracy": correct / len(rows) if rows else 0.0,
        "slot_pair_f1": _slot_pair_f1(true_slots, pred_slots),
        "n_examples": len(rows),
    }


def load_eval_rows(dataset_path, n):
    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]
    # Use the tail (held-out-ish) slice so we don't only measure memorised rows.
    tail = data[int(0.85 * len(data)) :]
    return tail[:n] if n and n < len(tail) else tail


def main():
    parser = argparse.ArgumentParser(description="Quantize + benchmark the joint NLU model.")
    parser.add_argument("--model-path", default=None, help="Checkpoint dir (default: latest).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Eval .jsonl path.")
    parser.add_argument("--eval-rows", type=int, default=500, help="Accuracy sample size.")
    parser.add_argument("--latency-reps", type=int, default=5, help="Latency repetitions.")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--f1-tolerance",
        type=float,
        default=0.01,
        help="Max allowed intent macro-F1 drop to recommend shipping int8.",
    )
    args = parser.parse_args()

    model_path = args.model_path or get_latest_model_path()
    if not model_path:
        raise SystemExit("No model path provided and no latest checkpoint found.")
    print(f"Loading fp32 model from: {model_path}")

    device = torch.device("cpu")  # dynamic quantization targets CPU inference
    loader = ModelLoader(device)
    loader.load_all(model_path)
    fp32_model = loader.model.eval()

    print("Quantizing (dynamic int8, nn.Linear)...")
    int8_model = quantize(fp32_model)

    # --- Persist artifact + provenance -----------------------------------
    quant_dir = os.path.join(model_path, "quantized")
    os.makedirs(quant_dir, exist_ok=True)
    weights_file = os.path.join(quant_dir, "pytorch_model_quantized.bin")
    torch.save(int8_model.state_dict(), weights_file)

    src_manifest = os.path.join(model_path, "train_manifest.json")
    if os.path.isfile(src_manifest):
        shutil.copy2(src_manifest, os.path.join(quant_dir, "train_manifest.json"))

    fp32_bytes = _fp32_weight_size_bytes(model_path)
    int8_bytes = os.path.getsize(weights_file)

    # --- Benchmarks ------------------------------------------------------
    print("Benchmarking latency (fp32)...")
    fp32_p50, fp32_p95, fp32_mean = benchmark_latency(
        fp32_model, loader, device, args.latency_reps, args.max_length
    )
    print("Benchmarking latency (int8)...")
    int8_p50, int8_p95, int8_mean = benchmark_latency(
        int8_model, loader, device, args.latency_reps, args.max_length
    )

    rows = load_eval_rows(args.dataset, args.eval_rows)
    print(f"Evaluating accuracy on {len(rows)} rows (fp32)...")
    fp32_acc = evaluate_accuracy(fp32_model, loader, device, rows, args.max_length)
    print(f"Evaluating accuracy on {len(rows)} rows (int8)...")
    int8_acc = evaluate_accuracy(int8_model, loader, device, rows, args.max_length)

    f1_drop = fp32_acc["intent_macro_f1"] - int8_acc["intent_macro_f1"]
    ship = f1_drop <= args.f1_tolerance

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_model_path": model_path,
        "quantization": "dynamic_int8_linear",
        "max_length": args.max_length,
        "size_bytes": {"fp32_weights": fp32_bytes, "int8_weights": int8_bytes},
        "size_reduction_pct": (100.0 * (1 - int8_bytes / fp32_bytes) if fp32_bytes else None),
        "latency_ms": {
            "fp32": {"p50": fp32_p50, "p95": fp32_p95, "mean": fp32_mean},
            "int8": {"p50": int8_p50, "p95": int8_p95, "mean": int8_mean},
            "speedup_mean": fp32_mean / int8_mean if int8_mean else None,
        },
        "accuracy": {"fp32": fp32_acc, "int8": int8_acc, "intent_macro_f1_drop": f1_drop},
        "f1_tolerance": args.f1_tolerance,
        "recommend_ship_int8": ship,
    }
    with open(os.path.join(quant_dir, "quant_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- Report ----------------------------------------------------------
    mb = 1024 * 1024
    size_pct = manifest["size_reduction_pct"]
    speedup = manifest["latency_ms"]["speedup_mean"]
    size_pct_str = f"{size_pct:.1f}% smaller" if size_pct is not None else "n/a"
    speedup_str = f"{speedup:.2f}x mean" if speedup is not None else "n/a"
    print("\n" + "=" * 64)
    print("QUANTIZATION BENCHMARK")
    print("=" * 64)
    print(f"Disk    fp32 weights : {fp32_bytes / mb:7.1f} MB")
    print(f"        int8 weights : {int8_bytes / mb:7.1f} MB   ({size_pct_str})")
    print(f"Latency fp32  p50/p95: {fp32_p50:6.1f} / {fp32_p95:6.1f} ms")
    print(f"        int8  p50/p95: {int8_p50:6.1f} / {int8_p95:6.1f} ms   ({speedup_str})")
    print(
        f"Intent  fp32 macroF1 : {fp32_acc['intent_macro_f1']:.4f}"
        f"  acc {fp32_acc['intent_accuracy']:.4f}"
    )
    print(
        f"        int8 macroF1 : {int8_acc['intent_macro_f1']:.4f}"
        f"  acc {int8_acc['intent_accuracy']:.4f}   (drop {f1_drop:+.4f})"
    )
    print(f"Slot    fp32 pairF1  : {fp32_acc['slot_pair_f1']:.4f}")
    print(f"        int8 pairF1  : {int8_acc['slot_pair_f1']:.4f}")
    print("-" * 64)
    print(
        f"RECOMMENDATION: {'SHIP int8' if ship else 'KEEP fp32'} "
        f"(F1 drop {f1_drop:+.4f}, tolerance {args.f1_tolerance})"
    )
    print(f"Manifest written to: {os.path.join(quant_dir, 'quant_manifest.json')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
