"""Export the joint intent+slot model to ONNX and validate it (Phase 4C).

The model's ``forward`` returns ``(loss, intent_logits, slot_logits)`` where
``loss`` is a Python ``0`` in eval mode -- not exportable. We wrap it so the
ONNX graph has clean inputs ``(input_ids, attention_mask)`` and outputs
``(intent_logits, slot_logits)`` with dynamic batch and sequence axes.

After export the script:
  * checks numerical parity (max abs logit diff) against PyTorch,
  * benchmarks ONNX-Runtime vs PyTorch latency + cold start,
  * writes ``<model>/onnx/model.onnx`` + ``onnx_manifest.json``.

Usage::

    python3 ML/export_onnx.py                       # latest checkpoint
    python3 ML/export_onnx.py --model-path ML/models/.../v9 --opset 17
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))

from quantize_model import LATENCY_COMMANDS, benchmark_latency  # noqa: E402
from utils import get_latest_model_path  # noqa: E402
from vimaan_nlu import normalize_aviation_input  # noqa: E402
from vimaan_nlu.inference import DEFAULT_MAX_LENGTH  # noqa: E402
from vimaan_nlu.model_loader import ModelLoader  # noqa: E402
from vimaan_nlu.onnx_backend import OnnxBackend, onnx_model_path  # noqa: E402


class _ExportWrapper(nn.Module):
    """Exposes only ``(intent_logits, slot_logits)`` for a clean ONNX graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        bert_output = self.model.bert_for_slots.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state
        intent_logits = self.model.intent_classifier(sequence_output[:, 0, :])
        slot_logits = self.model.bert_for_slots.classifier(sequence_output)
        return intent_logits, slot_logits


def export(model, out_path, max_length, opset):
    wrapper = _ExportWrapper(model).eval()
    dummy_ids = torch.ones(1, max_length, dtype=torch.long)
    dummy_mask = torch.ones(1, max_length, dtype=torch.long)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return out_path


def check_parity(model, backend, tokenizer, max_length, commands):
    """Return the worst-case max abs logit difference torch vs onnx."""
    worst_intent = 0.0
    worst_slot = 0.0
    mismatched = 0
    model.eval()
    for text in commands:
        enc = tokenizer(
            normalize_aviation_input(text),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids, mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            _, t_intent, t_slot = model(ids, mask)
        _, o_intent, o_slot = backend(ids, mask)
        worst_intent = max(worst_intent, (t_intent - o_intent).abs().max().item())
        worst_slot = max(worst_slot, (t_slot - o_slot).abs().max().item())
        if t_intent.argmax(1).item() != o_intent.argmax(1).item():
            mismatched += 1
    return {
        "max_intent_logit_diff": worst_intent,
        "max_slot_logit_diff": worst_slot,
        "argmax_mismatches": mismatched,
        "n_commands": len(commands),
    }


def cold_start_ms(onnx_path):
    start = time.perf_counter()
    backend = OnnxBackend(onnx_path)
    ids = torch.ones(1, DEFAULT_MAX_LENGTH, dtype=torch.long)
    backend(ids, ids)
    return (time.perf_counter() - start) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Export + validate the ONNX model.")
    parser.add_argument("--model-path", default=None, help="Checkpoint dir (default: latest).")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--latency-reps", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model_path or get_latest_model_path()
    if not model_path:
        raise SystemExit("No model path provided and no latest checkpoint found.")
    print(f"Loading fp32 model from: {model_path}")

    device = torch.device("cpu")
    loader = ModelLoader(device)
    loader.load_all(model_path)
    fp32_model = loader.model.eval()

    out_path = onnx_model_path(model_path)
    print(f"Exporting ONNX (opset {args.opset}) -> {out_path}")
    export(fp32_model, out_path, args.max_length, args.opset)

    backend = OnnxBackend(out_path)
    parity = check_parity(fp32_model, backend, loader.tokenizer, args.max_length, LATENCY_COMMANDS)
    print(
        f"Parity: max intent diff {parity['max_intent_logit_diff']:.2e}, "
        f"max slot diff {parity['max_slot_logit_diff']:.2e}, "
        f"argmax mismatches {parity['argmax_mismatches']}/{parity['n_commands']}"
    )

    print("Benchmarking latency (torch fp32)...")
    t_p50, t_p95, t_mean = benchmark_latency(
        fp32_model, loader, device, args.latency_reps, args.max_length
    )
    print("Benchmarking latency (onnx)...")
    o_p50, o_p95, o_mean = benchmark_latency(
        backend, loader, device, args.latency_reps, args.max_length
    )
    cold = cold_start_ms(out_path)

    onnx_bytes = os.path.getsize(out_path)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_model_path": model_path,
        "opset": args.opset,
        "max_length": args.max_length,
        "onnx_size_bytes": onnx_bytes,
        "parity": parity,
        "latency_ms": {
            "torch": {"p50": t_p50, "p95": t_p95, "mean": t_mean},
            "onnx": {"p50": o_p50, "p95": o_p95, "mean": o_mean},
            "speedup_mean": t_mean / o_mean if o_mean else None,
        },
        "onnx_cold_start_ms": cold,
    }
    with open(os.path.join(os.path.dirname(out_path), "onnx_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    mb = 1024 * 1024
    print("\n" + "=" * 60)
    print("ONNX EXPORT BENCHMARK")
    print("=" * 60)
    print(f"ONNX size        : {onnx_bytes / mb:7.1f} MB")
    print(
        f"Parity (argmax)  : {parity['n_commands'] - parity['argmax_mismatches']}"
        f"/{parity['n_commands']} match  (max logit diff "
        f"{max(parity['max_intent_logit_diff'], parity['max_slot_logit_diff']):.1e})"
    )
    print(f"Latency torch    : p50 {t_p50:6.1f} / p95 {t_p95:6.1f} ms")
    print(
        f"Latency onnx     : p50 {o_p50:6.1f} / p95 {o_p95:6.1f} ms"
        f"   ({manifest['latency_ms']['speedup_mean']:.2f}x mean)"
    )
    print(f"ONNX cold start  : {cold:6.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
