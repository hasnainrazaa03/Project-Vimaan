"""Model gate: may this model replace the previous one? (docs/PHASE_PLAN.md)

Runs a candidate and a baseline model through the same checks and prints
PASS/FAIL per gate; writes ``gate.json`` next to the candidate.

Gates
-----
1. **New phrasings** — the candidate's held-out split (templates it never
   trained on): per-intent accuracy >= 98.5% and slot F1 >= 0.99.
2. **No regression on old data** — the baseline's own held-out file: the
   candidate is within 0.5 point of the baseline.
3. **Gold utterances** (tests/test_gold_nlu.py) all right.
4. **Real voice** (tests/fixtures/real_voice.jsonl, what Vosk heard in
   flight): no more wrong actions than the baseline, and at least as many right.
5. **Negatives** — fragments / word salad / chatter in the held-out split:
   >= 95% end up rejected (not acted on).
6. **Latency** — p95 NLU < 25 ms on CPU (quantized, as in the plugin).

Usage::

    python scripts/model_gate.py v12 --baseline v11
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "ML"), str(ROOT / "tests")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS = ROOT / "ML" / "models" / "vimaan_nlu_model_best"
CONFIDENCE_FLOOR = 0.55  # the plugin's PythonInterface.CONFIDENCE_FLOOR
NUMERIC = {"degrees", "altitude", "flight_level", "frequency", "com_port"}

INTENT_ACC_MIN = 0.985
SLOT_F1_MIN = 0.99
OLD_DATA_TOLERANCE = 0.005
NEGATIVE_REJECT_MIN = 0.95
P95_MS_MAX = 25.0


def load(version: str):
    import torch
    from vimaan_nlu.inference import predict
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader(torch.device("cpu"))
    loader.load_all(model_path=str(MODELS / version), quantize=True)

    def run(text: str) -> dict[str, Any]:
        return predict(
            text,
            loader.model,
            loader.tokenizer,
            loader.device,
            loader.intent_map_rev,
            loader.slot_map_rev,
        )

    return run


# ---------------------------------------------------------------------------
# What the plugin would do with a prediction (mirrors _execute_command)
# ---------------------------------------------------------------------------
def decision(p: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """("act", slots) when the plugin would act on it, else (reason, {})."""
    from vimaan_nlu.commands import command, required_slots
    from vimaan_nlu.validators import has_command_word, heading_too_short, validate_slot

    intent, slots = p["intent"], p["slots"]
    cmd = command(intent)
    if p["confidence"] < CONFIDENCE_FLOOR:
        return "low_confidence", {}
    if cmd is None or not cmd.actionable:
        return "not_a_command", {}
    if not has_command_word(intent, p.get("normalized_text") or ""):
        return "no_command_word", {}
    if intent == "set_autopilot_heading" and heading_too_short(slots.get("degrees")):
        return "one_digit_heading", {}
    for name in required_slots().get(intent, ()):
        if not slots.get(name):
            return f"missing_{name}", {}
    for name, value in slots.items():
        if not validate_slot(name, value)[1]:
            return f"invalid_{name}", {}
    return "act", slots


def same_value(name: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a == b
    if name in NUMERIC:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except ValueError:
            return False
    return str(a).lower() == str(b).lower()


def canonical_expected(row: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Expected (family intent, canonical slots) for a dataset row of either
    format (v1 rows carry pre-family intents and raw values)."""
    from vimaan_nlu.commands import LEGACY_INTENTS
    from vimaan_nlu.normalization import normalize_aviation_input
    from vimaan_nlu.postprocessor import canonicalize_choices, resolve_instance

    intent, slots = row["intent"], dict(row.get("slots") or {})
    if intent in LEGACY_INTENTS:
        # v1 rows: old intent name. Their numbered labels were a coin flip
        # when no number was spoken, so the instance comes from the words,
        # exactly as at runtime (resolve_instance below).
        intent, _implied, renames = LEGACY_INTENTS[intent]
        for old, new in renames.items():
            if old in slots:
                slots[new] = slots.pop(old)
        slots.pop("instance", None)
    norm = normalize_aviation_input(row["text"])
    slots = canonicalize_choices(slots, norm, intent)
    slots = resolve_instance(slots, norm, intent)
    return intent, slots


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def score_rows(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        want_intent, want = canonical_expected(row)
        p = run(row["text"])
        c = by_intent[want_intent]
        c["n"] += 1
        c["intent_ok"] += p["intent"] == want_intent
        got = p["slots"] if p["intent"] == want_intent else {}
        for name, value in want.items():
            if same_value(name, got.get(name), value):
                c["tp"] += 1
            else:
                c["fn"] += 1
        for name, value in got.items():
            if name not in want or not same_value(name, value, want[name]):
                c["fp"] += 1
    out = {}
    for intent, c in sorted(by_intent.items()):
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[intent] = {"n": c["n"], "intent_acc": c["intent_ok"] / c["n"], "slot_f1": f1}
    total = sum(v["n"] for v in out.values())
    out["_all"] = {
        "n": total,
        "intent_acc": sum(v["intent_acc"] * v["n"] for v in out.values()) / total,
        "slot_f1": sum(v["slot_f1"] * v["n"] for v in out.values()) / total,
    }
    return out


def real_voice(run) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(ROOT / "tests" / "fixtures" / "real_voice.jsonl")]
    result: Counter[str] = Counter()
    wrong = []
    for r in rows:
        p = run(r["heard"]) if r["heard"] else None
        if p is None:
            result["safe"] += 1
            continue
        verdict, slots = decision(p)
        if verdict != "act":
            result["safe"] += 1
        elif p["intent"] == r["intent"] and all(
            same_value(k, slots.get(k), v) for k, v in r["slots"].items()
        ):
            result["right"] += 1
        else:
            result["wrong"] += 1
            wrong.append(f"{r['heard']!r} -> {p['intent']} {slots} (said {r['said']!r})")
    return {**result, "n": len(rows), "wrong_examples": wrong}


def negatives(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg = [r for r in rows if str(r.get("template", "")).startswith("None#neg")]
    acted = [r["text"] for r in neg if decision(run(r["text"]))[0] == "act"]
    return {
        "n": len(neg),
        "rejected": 1 - len(acted) / len(neg) if neg else 1.0,
        "acted": acted[:15],
    }


def gold(run) -> dict[str, Any]:
    from test_gold_nlu import GOLD

    bad = []
    for text, intent, slots in GOLD:
        p = run(text)
        if p["intent"] != intent or any(
            not same_value(k, p["slots"].get(k), v) for k, v in slots.items()
        ):
            bad.append(f"{text!r} -> {p['intent']} {p['slots']}")
    return {"n": len(GOLD), "failed": bad}


def latency(run, texts: list[str]) -> float:
    times = []
    for t in texts:
        start = time.perf_counter()
        run(t)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.quantiles(times, n=20)[18]  # p95


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidate")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--limit", type=int, default=6000, help="max held-out rows per set")
    args = parser.parse_args(argv)

    cand_dir, base_dir = MODELS / args.candidate, MODELS / args.baseline
    new_rows = read_jsonl(cand_dir / "holdout_test.jsonl")[: args.limit]
    old_rows = read_jsonl(base_dir / "holdout_test.jsonl")[: args.limit]
    cand, base = load(args.candidate), load(args.baseline)

    report: dict[str, Any] = {"candidate": args.candidate, "baseline": args.baseline}
    gates: list[tuple[str, bool, str]] = []

    new_c = score_rows(cand, new_rows)
    new_b = score_rows(base, new_rows)
    weak = {
        i: v
        for i, v in new_c.items()
        if not i.startswith("_")
        and (v["intent_acc"] < INTENT_ACC_MIN or v["slot_f1"] < SLOT_F1_MIN)
    }
    gates.append(
        (
            "1 new phrasings",
            not weak,
            f"intent {new_c['_all']['intent_acc']:.4f} slot F1 {new_c['_all']['slot_f1']:.4f} "
            f"(baseline {new_b['_all']['intent_acc']:.4f} / {new_b['_all']['slot_f1']:.4f})"
            + (f"; below bar: {sorted(weak)}" if weak else ""),
        )
    )
    old_c = score_rows(cand, old_rows)
    old_b = score_rows(base, old_rows)
    ok = old_c["_all"]["intent_acc"] >= old_b["_all"]["intent_acc"] - OLD_DATA_TOLERANCE
    gates.append(
        (
            "2 old data",
            ok,
            f"intent {old_c['_all']['intent_acc']:.4f} vs baseline {old_b['_all']['intent_acc']:.4f}; "
            f"slot F1 {old_c['_all']['slot_f1']:.4f} vs {old_b['_all']['slot_f1']:.4f}",
        )
    )
    g = gold(cand)
    gates.append(("3 gold", not g["failed"], f"{g['n'] - len(g['failed'])}/{g['n']} {g['failed']}"))
    rv_c, rv_b = real_voice(cand), real_voice(base)
    ok = rv_c.get("wrong", 0) <= rv_b.get("wrong", 0) and rv_c.get("right", 0) >= rv_b.get(
        "right", 0
    )
    gates.append(
        (
            "4 real voice",
            ok,
            f"right {rv_c.get('right', 0)} safe {rv_c.get('safe', 0)} wrong {rv_c.get('wrong', 0)} "
            f"(baseline {rv_b.get('right', 0)}/{rv_b.get('safe', 0)}/{rv_b.get('wrong', 0)})",
        )
    )
    neg_c, neg_b = negatives(cand, new_rows), negatives(base, new_rows)
    gates.append(
        (
            "5 negatives",
            neg_c["rejected"] >= NEGATIVE_REJECT_MIN,
            f"{neg_c['rejected']:.3f} rejected of {neg_c['n']} (baseline {neg_b['rejected']:.3f})",
        )
    )
    p95 = latency(cand, [r["text"] for r in new_rows[:300]])
    gates.append(("6 latency", p95 < P95_MS_MAX, f"p95 {p95:.1f} ms"))

    report.update(
        {
            "new_phrasings": {"candidate": new_c, "baseline": new_b},
            "old_data": {"candidate": old_c["_all"], "baseline": old_b["_all"]},
            "gold": g,
            "real_voice": {"candidate": rv_c, "baseline": rv_b},
            "negatives": {"candidate": neg_c, "baseline": neg_b},
            "latency_p95_ms": p95,
            "gates": {name: {"pass": ok, "detail": d} for name, ok, d in gates},
        }
    )
    (cand_dir / "gate.json").write_text(json.dumps(report, indent=1))

    print(f"\nModel gate: {args.candidate} vs {args.baseline}\n")
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {detail}")
    print("\n  per intent (new phrasings): intent acc / slot F1")
    for intent, v in new_c.items():
        if not intent.startswith("_"):
            print(f"    {intent:<26} {v['intent_acc']:.3f} / {v['slot_f1']:.3f}  (n={v['n']})")
    if rv_c.get("wrong_examples"):
        print("\n  real voice, wrong actions:")
        for w in rv_c["wrong_examples"]:
            print(f"    {w}")
    if neg_c["acted"]:
        print("\n  negatives acted on (sample):", neg_c["acted"])
    passed = all(ok for _, ok, _ in gates)
    print(f"\n  {'ALL GATES PASS' if passed else 'GATE FAILED'}  -> {cand_dir / 'gate.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
