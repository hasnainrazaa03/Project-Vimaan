"""Model gate: may this model replace the previous one? (docs/PHASE_PLAN.md)

Runs a candidate and a baseline model through the same checks and prints
PASS/FAIL per gate, with raw counts and failing examples; writes ``gate.json``
next to the candidate.

Gates
-----
1. **New phrasings** — the candidate's held-out split (templates it never
   trained on): per-intent accuracy >= 98.5% and slot F1 >= 0.99.
2. **Old data, judged by the action taken** — the baseline's held-out file.
   Denominator: rows whose expected command is actionable (the same rows for
   both models). Each row ends as right (the plugin would perform the expected
   action), rejected (it would not act) or wrong (it would act differently).
   right rate = right / denominator. Passes when the candidate's right rate is
   within 0.5 percentage points of the baseline's AND its wrong count is no
   higher. The old label-based accuracy is reported separately and decides nothing.
3. **Gold utterances** (tests/test_gold_nlu.py) all right, and its fragments
   never acted on.
4. **Real voice** (tests/fixtures/real_voice.jsonl, what Vosk heard in
   flight): no more wrong actions than the baseline, and at least as many right.
5. **Negatives** — fragments / word salad / chatter in the held-out split:
   >= 95% end up rejected (not acted on).
6. **Latency** — p95 NLU < 25 ms on CPU (quantized, as in the plugin).

Usage::

    python scripts/model_gate.py v17 --baseline v11
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
RIGHT_RATE_TOLERANCE = 0.005  # 0.5 percentage points
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
def effective_action(intent: str, slots: dict[str, Any]) -> dict[str, Any]:
    """The slots the plugin acts with: no port spoken means COM 1
    (``_set_com_frequency``). Used for predictions and expectations alike."""
    out = dict(slots)
    if intent == "set_com_frequency" and not out.get("com_port"):
        out["com_port"] = "1"
    return out


def decision(p: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """("act", effective slots) when the plugin would act on it, else (reason, {}).
    Only the slots the command declares count: its handler ignores stray ones."""
    from vimaan_nlu.ambiguity import two_or_to
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
    if two_or_to(intent, p.get("original_text") or ""):
        return "ambiguous_two", {}
    for name in required_slots().get(intent, ()):
        if not slots.get(name):
            return f"missing_{name}", {}
    for name in cmd.slots:
        if name in slots and not validate_slot(name, slots[name])[1]:
            return f"invalid_{name}", {}
    action = effective_action(intent, slots)
    if intent == "set_com_frequency" and str(action["com_port"]) not in cmd.params["datarefs"]:
        return "no_such_port", {}
    instances = cmd.params.get("instances")
    if instances and intent != "toggle_engine" and action.get("instance", "1") not in instances:
        return "no_such_instance", {}
    return "act", action


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
    return intent, effective_action(intent, slots)


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
        got = effective_action(want_intent, p["slots"]) if p["intent"] == want_intent else {}
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
        if c["tp"] + c["fn"] == 0:
            f1 = 1.0  # this intent has no slots to find; F1 is undefined, not 0
        out[intent] = {"n": c["n"], "intent_acc": c["intent_ok"] / c["n"], "slot_f1": f1}
    total = sum(v["n"] for v in out.values())
    out["_all"] = {
        "n": total,
        "intent_acc": sum(v["intent_acc"] * v["n"] for v in out.values()) / total,
        "slot_f1": sum(v["slot_f1"] * v["n"] for v in out.values()) / total,
    }
    return out


def judge(run, text: str, want_intent: str, want: dict[str, Any]) -> tuple[str, str]:
    """(right | rejected | wrong, what happened)."""
    p = run(text)
    verdict, action = decision(p)
    if verdict != "act":
        return "rejected", verdict
    ok = p["intent"] == want_intent and all(
        same_value(k, action.get(k), v) for k, v in want.items()
    )
    return ("right" if ok else "wrong"), f"{p['intent']} {action}"


def real_voice(run) -> dict[str, Any]:
    rows = [json.loads(line) for line in open(ROOT / "tests" / "fixtures" / "real_voice.jsonl")]
    result: Counter[str] = Counter()
    sessions: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = {"rejected": [], "wrong": []}
    for r in rows:
        want = effective_action(r["intent"], r["slots"])
        outcome, detail = (
            judge(run, r["heard"], r["intent"], want)
            if r["heard"]
            else ("rejected", "nothing heard")
        )
        result[outcome] += 1
        sessions[r["clip"][:15]][outcome] += 1
        if outcome != "right":
            examples[outcome].append(f"said {r['said']!r} heard {r['heard']!r} -> {detail}")
    return {
        "by_session": {k: dict(v) for k, v in sorted(sessions.items())},
        "n": len(rows),
        "right": result["right"],
        "rejected": result["rejected"],
        "wrong": result["wrong"],
        "examples": examples,
    }


def actions(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate 2: outcome of every row whose expected command is actionable."""
    from vimaan_nlu.commands import command

    outcomes: dict[str, str] = {}
    details: dict[str, str] = {}
    for i, row in enumerate(rows):
        want_intent, want = canonical_expected(row)
        cmd = command(want_intent)
        if cmd is None or not cmd.actionable:
            continue
        key = f"{i}|{row['text']}"
        outcomes[key], details[key] = judge(run, row["text"], want_intent, want)
    counts = Counter(outcomes.values())
    n = len(outcomes)
    return {
        "n": n,
        "right": counts["right"],
        "rejected": counts["rejected"],
        "wrong": counts["wrong"],
        "right_rate": counts["right"] / n if n else 0.0,
        "outcomes": outcomes,
        "details": details,
    }


def negatives(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg = [r for r in rows if str(r.get("template", "")).startswith("None#neg")]
    acted = [r["text"] for r in neg if decision(run(r["text"]))[0] == "act"]
    return {
        "n": len(neg),
        "acted_count": len(acted),
        "rejected": 1 - len(acted) / len(neg) if neg else 1.0,
        "acted": acted[:15],
    }


def gold(run) -> dict[str, Any]:
    from test_gold_nlu import GOLD, MUST_NOT_ACT

    bad = []
    for text, intent, slots in GOLD:
        p = run(text)
        if p["intent"] != intent or any(
            not same_value(k, p["slots"].get(k), v) for k, v in slots.items()
        ):
            bad.append(f"{text!r} -> {p['intent']} {p['slots']}")
    acted = [t for t in MUST_NOT_ACT if decision(run(t))[0] == "act"]
    return {"n": len(GOLD), "failed": bad, "fragments": len(MUST_NOT_ACT), "fragments_acted": acted}


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
    parser.add_argument("--limit", type=int, default=None, help="max held-out rows per set")
    args = parser.parse_args(argv)

    cand_dir, base_dir = MODELS / args.candidate, MODELS / args.baseline
    new_rows = read_jsonl(cand_dir / "holdout_test.jsonl")[: args.limit]
    old_rows = read_jsonl(base_dir / "holdout_test.jsonl")[: args.limit]
    cand, base = load(args.candidate), load(args.baseline)

    gates: list[tuple[str, bool, str]] = []

    new_c, new_b = score_rows(cand, new_rows), score_rows(base, new_rows)
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
            f"n={new_c['_all']['n']} intent {new_c['_all']['intent_acc']:.4f} "
            f"slot F1 {new_c['_all']['slot_f1']:.4f} (baseline "
            f"{new_b['_all']['intent_acc']:.4f} / {new_b['_all']['slot_f1']:.4f})"
            + (f"; below bar: {sorted(weak)}" if weak else ""),
        )
    )

    act_c, act_b = actions(cand, old_rows), actions(base, old_rows)
    rate_ok = act_c["right_rate"] >= act_b["right_rate"] - RIGHT_RATE_TOLERANCE
    wrong_ok = act_c["wrong"] <= act_b["wrong"]
    gates.append(
        (
            "2 old data",
            rate_ok and wrong_ok,
            f"n={act_c['n']} right/rejected/wrong {act_c['right']}/{act_c['rejected']}/{act_c['wrong']} "
            f"rate {act_c['right_rate']:.2%}; baseline "
            f"{act_b['right']}/{act_b['rejected']}/{act_b['wrong']} rate {act_b['right_rate']:.2%} "
            f"(rate {'ok' if rate_ok else 'LOW'}, wrong {'ok' if wrong_ok else 'HIGH'})",
        )
    )
    label_c, label_b = score_rows(cand, old_rows), score_rows(base, old_rows)

    g = gold(cand)
    gates.append(
        (
            "3 gold",
            not g["failed"] and not g["fragments_acted"],
            f"{g['n'] - len(g['failed'])}/{g['n']} right {g['failed']}; "
            f"{g['fragments'] - len(g['fragments_acted'])}/{g['fragments']} fragments "
            f"not acted on {g['fragments_acted']}",
        )
    )

    rv_c, rv_b = real_voice(cand), real_voice(base)
    gates.append(
        (
            "4 real voice",
            rv_c["wrong"] <= rv_b["wrong"] and rv_c["right"] >= rv_b["right"],
            f"n={rv_c['n']} right/rejected/wrong {rv_c['right']}/{rv_c['rejected']}/{rv_c['wrong']} "
            f"(baseline {rv_b['right']}/{rv_b['rejected']}/{rv_b['wrong']})",
        )
    )

    neg_c, neg_b = negatives(cand, new_rows), negatives(base, new_rows)
    gates.append(
        (
            "5 negatives",
            neg_c["rejected"] >= NEGATIVE_REJECT_MIN,
            f"{neg_c['n'] - neg_c['acted_count']}/{neg_c['n']} rejected "
            f"({neg_c['rejected']:.1%}; baseline {neg_b['rejected']:.1%})",
        )
    )

    p95 = latency(cand, [r["text"] for r in new_rows[:300]])
    gates.append(("6 latency", p95 < P95_MS_MAX, f"p95 {p95:.1f} ms"))

    def lost(status: str) -> list[str]:
        return [
            f"{key.split('|', 1)[1]!r} -> {act_c['details'][key]}"
            for key, outcome in act_c["outcomes"].items()
            if outcome == status and act_b["outcomes"][key] == "right"
        ]

    def wrong_rows(act: dict[str, Any]) -> list[str]:
        return [
            f"{key.split('|', 1)[1]!r} -> {act['details'][key]}"
            for key, outcome in act["outcomes"].items()
            if outcome == "wrong"
        ]

    report: dict[str, Any] = {
        "candidate": args.candidate,
        "baseline": args.baseline,
        "new_phrasings": {"candidate": new_c, "baseline": new_b},
        "old_data_actions": {
            "candidate": {k: v for k, v in act_c.items() if k not in ("outcomes", "details")},
            "baseline": {k: v for k, v in act_b.items() if k not in ("outcomes", "details")},
            "candidate_wrong": wrong_rows(act_c),
            "baseline_wrong": wrong_rows(act_b),
            "baseline_right_candidate_rejected": lost("rejected"),
            "baseline_right_candidate_wrong": lost("wrong"),
        },
        "old_data_labels_information_only": {
            "candidate": label_c["_all"],
            "baseline": label_b["_all"],
        },
        "gold": g,
        "real_voice": {"candidate": rv_c, "baseline": rv_b},
        "negatives": {"candidate": neg_c, "baseline": neg_b},
        "latency_p95_ms": p95,
        "gates": {name: {"pass": ok, "detail": d} for name, ok, d in gates},
    }
    (cand_dir / "gate.json").write_text(json.dumps(report, indent=1))

    print(f"\nModel gate: {args.candidate} vs {args.baseline}\n")
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {detail}")
    print(
        f"\n  old data, label-based score (information only, decides nothing): intent "
        f"{label_c['_all']['intent_acc']:.4f} vs baseline {label_b['_all']['intent_acc']:.4f}; "
        f"slot F1 {label_c['_all']['slot_f1']:.4f} vs {label_b['_all']['slot_f1']:.4f}; "
        f"n={label_c['_all']['n']}"
    )
    print("\n  per intent (new phrasings): intent acc / slot F1")
    for intent, v in new_c.items():
        if not intent.startswith("_"):
            print(f"    {intent:<26} {v['intent_acc']:.3f} / {v['slot_f1']:.3f}  (n={v['n']})")
    for title, items in (
        ("old data: baseline right, candidate WRONG", lost("wrong")),
        ("old data: baseline right, candidate rejected", lost("rejected")),
        ("old data: candidate wrong (all)", wrong_rows(act_c)),
        ("old data: baseline wrong (all)", wrong_rows(act_b)),
        ("real voice: candidate wrong", rv_c["examples"]["wrong"]),
        ("real voice: baseline wrong", rv_b["examples"]["wrong"]),
        ("real voice: candidate rejected", rv_c["examples"]["rejected"]),
        ("negatives acted on", neg_c["acted"]),
    ):
        print(f"\n  {title}: {len(items)}")
        for item in items[:40]:
            print(f"    {item}")
    passed = all(ok for _, ok, _ in gates)
    print("\n  real voice by flight (right/rejected/wrong), candidate vs baseline:")
    for session, counts in rv_c["by_session"].items():
        base_counts = rv_b["by_session"][session]
        print(
            f"    {session}: "
            + "/".join(str(counts.get(k, 0)) for k in ("right", "rejected", "wrong"))
            + " vs "
            + "/".join(str(base_counts.get(k, 0)) for k in ("right", "rejected", "wrong"))
        )
    print(f"\n  {'ALL GATES PASS' if passed else 'GATE FAILED'}  -> {cand_dir / 'gate.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
