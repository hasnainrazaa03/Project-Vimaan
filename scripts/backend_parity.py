"""Backend parity: does the plugin act the same on every backend? (PHASE_PLAN E1.10)

``ML/export_onnx.py`` checks logits on 10 sentences. This runs every text the
model gate uses (new and old held-out rows, gold, fragments, real voice)
through three backends and compares what the plugin would *do* with each
prediction (``model_gate.decision``: act with these values, or reject why):

* **int8**: torch with dynamic int8 quantization, what the plugin runs;
* **fp32**: the unquantized checkpoint ONNX is exported from;
* **onnx**: ONNX Runtime on ``<model>/onnx/model.onnx``.

Passes when int8 quantization was actually applied (not a silent fp32
fallback) and ONNX makes the same decision as fp32 on every text. int8 vs
ONNX differences are scored against the labels, for information: they are
what switching the plugin's backend would change. Also counts texts longer than the runtime's
token limit (inference truncates at ``DEFAULT_MAX_LENGTH``).

Writes ``<model>/onnx/parity.json``.

Usage::

    python ML/export_onnx.py --model-path ML/models/vimaan_nlu_model_best/v17
    python scripts/backend_parity.py v17 --baseline v11
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_gate import (  # noqa: E402
    MODELS,
    ROOT,
    canonical_expected,
    decision,
    effective_action,
    read_jsonl,
    same_value,
)

Run = Callable[[str], dict[str, Any]]
Expected = tuple[str, dict[str, Any]]  # (intent, action) the text should produce


def gate_texts(candidate: str, baseline: str) -> dict[str, list[tuple[str, Expected]]]:
    """Every text the model gate scores, by set, with what it should do."""
    from test_gold_nlu import GOLD, MUST_NOT_ACT

    voice = [json.loads(line) for line in open(ROOT / "tests" / "fixtures" / "real_voice.jsonl")]
    return {
        "new_phrasings": [
            (r["text"], canonical_expected(r))
            for r in read_jsonl(MODELS / candidate / "holdout_test.jsonl")
        ],
        "old_data": [
            (r["text"], canonical_expected(r))
            for r in read_jsonl(MODELS / baseline / "holdout_test.jsonl")
        ],
        "gold": [(t, (intent, slots)) for t, intent, slots in GOLD]
        + [(t, ("None", {})) for t in MUST_NOT_ACT],
        "real_voice": [
            (r["heard"], (r["intent"], effective_action(r["intent"], r["slots"])))
            for r in voice
            if r["heard"]
        ],
    }


def score(p: dict[str, Any], want: Expected) -> str:
    """right / rejected / wrong, as the gate counts them. A text that should
    not act (negatives, fragments) is right when rejected, wrong when acted on."""
    from vimaan_nlu.commands import command

    want_intent, want_action = want
    verdict, action = decision(p)
    cmd = command(want_intent)
    if cmd is None or not cmd.actionable:
        return "wrong" if verdict == "act" else "right"
    if verdict != "act":
        return "rejected"
    ok = p["intent"] == want_intent and all(
        same_value(k, action.get(k), v) for k, v in want_action.items()
    )
    return "right" if ok else "wrong"


def outcome(p: dict[str, Any]) -> tuple[str, str]:
    """What the plugin would do, as a comparable pair."""
    verdict, action = decision(p)
    return verdict, json.dumps(action, sort_keys=True) if verdict == "act" else ""


def compare(texts: dict[str, Expected], runners: dict[str, Run]) -> dict[str, Any]:
    """Decisions of each runner on each text, where they differ, and on those
    texts how each runner scores (right / rejected / wrong)."""
    preds = {name: {t: run(t) for t in texts} for name, run in runners.items()}
    decided = {name: {t: outcome(p) for t, p in by.items()} for name, by in preds.items()}
    names = list(runners)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            differ = [t for t in texts if decided[a][t] != decided[b][t]]
            scored = {name: {s: 0 for s in ("right", "rejected", "wrong")} for name in (a, b)}
            examples = []
            for t in differ:
                row: dict[str, Any] = {"text": t, "expected": list(texts[t])}
                for name in (a, b):
                    verdict = score(preds[name][t], texts[t])
                    scored[name][verdict] += 1
                    row[name] = [*decided[name][t], verdict]
                examples.append(row)
            pairs[f"{a}_vs_{b}"] = {"differ": len(differ), "scored": scored, "examples": examples}
    return {"n_texts": len(texts), "pairs": pairs}


def too_long(texts: list[str], tokenizer, limits: tuple[int, ...]) -> dict[str, Any]:
    from vimaan_nlu.normalization import normalize_aviation_input

    lengths = {t: len(tokenizer(normalize_aviation_input(t))["input_ids"]) for t in texts}
    out: dict[str, Any] = {"longest": max(lengths.values()) if lengths else 0}
    for limit in limits:
        over = sorted((t for t, n in lengths.items() if n > limit), key=lambda t: -lengths[t])
        out[f"over_{limit}"] = {"count": len(over), "examples": over[:10]}
    return out


def load_runner(model_dir: Path, backend: str, quantize: bool) -> tuple[Run, Any, Any]:
    import torch
    from vimaan_nlu.inference import predict
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader(torch.device("cpu"))
    info = loader.load_all(model_path=str(model_dir), quantize=quantize, backend=backend)

    def run(text: str) -> dict[str, Any]:
        return predict(
            text,
            loader.model,
            loader.tokenizer,
            loader.device,
            loader.intent_map_rev,
            loader.slot_map_rev,
        )

    return run, info, loader.tokenizer


def main(argv: list[str] | None = None) -> int:
    from vimaan_nlu.inference import DEFAULT_MAX_LENGTH
    from vimaan_nlu.onnx_backend import onnx_model_path

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidate")
    parser.add_argument("--baseline", required=True, help="its held-out file is the old data")
    parser.add_argument("--limit", type=int, default=None, help="max texts per set")
    args = parser.parse_args(argv)

    model_dir = MODELS / args.candidate
    if not Path(onnx_model_path(str(model_dir))).exists():
        print(f"No ONNX export: run python ML/export_onnx.py --model-path {model_dir}")
        return 2

    sets = {k: v[: args.limit] for k, v in gate_texts(args.candidate, args.baseline).items()}
    texts: dict[str, Expected] = {}
    for rows in sets.values():
        for text, want in rows:
            texts.setdefault(text, want)  # a text in two sets keeps its first label

    int8, info, tokenizer = load_runner(model_dir, "torch", quantize=True)
    fp32, _, _ = load_runner(model_dir, "torch", quantize=False)
    onnx, _, _ = load_runner(model_dir, "onnx", quantize=False)
    quantized = bool(info.get("quantized"))

    result = compare(texts, {"int8": int8, "fp32": fp32, "onnx": onnx})
    manifest = json.loads((model_dir / "train_manifest.json").read_text())
    train_max = manifest.get("hyperparams", {}).get("max_length")
    limits = tuple(sorted({DEFAULT_MAX_LENGTH, train_max or DEFAULT_MAX_LENGTH}))
    length = too_long(list(texts), tokenizer, limits)
    onnx_ok = result["pairs"]["fp32_vs_onnx"]["differ"] == 0
    report = {
        "candidate": args.candidate,
        "sets": {k: len(v) for k, v in sets.items()},
        "int8_applied": quantized,
        "runtime_max_length": DEFAULT_MAX_LENGTH,
        "train_max_length": train_max,
        "token_lengths": length,
        **result,
        "pass": quantized and onnx_ok,
    }
    out = Path(onnx_model_path(str(model_dir))).parent / "parity.json"
    out.write_text(json.dumps(report, indent=1))

    print(f"\nBackend parity: {args.candidate}, {len(texts)} unique texts {report['sets']}\n")
    print(f"  {'PASS' if quantized else 'FAIL'}  int8 quantization applied: {quantized}")
    for pair, d in result["pairs"].items():
        verdict = ("PASS" if onnx_ok else "FAIL") if pair == "fp32_vs_onnx" else "info"
        print(f"  {verdict}  {pair:<14} decisions differ on {d['differ']}")
        if d["differ"]:
            for name, counts in d["scored"].items():
                print(f"          on those, {name:<5} right/rejected/wrong {counts}")
    print(f"  info  longest text {length['longest']} tokens", end="")
    for key, value in length.items():
        if key.startswith("over_"):
            print(f"; {key.replace('_', ' ')}: {value['count']}", end="")
    print()
    print(f"\n  {'ALL PASS' if report['pass'] else 'FAIL'}  -> {out}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
