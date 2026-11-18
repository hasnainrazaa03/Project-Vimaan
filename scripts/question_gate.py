"""Question-model gate: may this question model (Tier 4, ``qN``) ship?

The question model sits beside the command model (``vN``) and only ever
answers: it knows no command, so nothing it predicts can change the sim. The
command model's own gate (scripts/model_gate.py) is untouched by it. These
gates reuse model_gate.py's bars; none is new:

Q1. **New phrasings**: the candidate's held-out templates: per-intent accuracy
    >= 98.5% and slot F1 >= 0.99 (gate 1's bars). "None" here is every
    command, greeting and negative: a command phrased in a way it never saw
    must still read as "not a question".
Q2. **Gold**: every Tier 4 gold question (tests/test_gold_nlu.py GOLD_TIER4)
    answered right, and no gold command or fragment answered (gate 3).
Q3. **Not a question**: >= 95% of the held-out commands, greetings and
    negatives not answered (gate 5's bar). An answer to a command sounds like
    the read-back of a command that was never done.
Q4. **Latency**: p95 < 25 ms on CPU, quantized (gate 6).

Every gate uses the plugin's answer rule (model_gate.answer_decision), with
its one-word rule. Also printed, deciding nothing: how many held-out lone
words the model alone would answer without that rule.

Usage::

    python scripts/question_gate.py q1
    python scripts/question_gate.py q2 --baseline q1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_gate as mg  # noqa: E402


def gold(run) -> dict[str, Any]:
    from test_gold_nlu import GOLD, GOLD_TIER4, MUST_NOT_ACT

    failed = []
    for text, intent, slots in GOLD_TIER4:
        p = run(text)
        why = mg.answer_decision(p)
        ok = (
            why == "answer"
            and p["intent"] == intent
            and all(mg.same_value(k, p["slots"].get(k), v) for k, v in slots.items())
        )
        if not ok:
            failed.append(f"{text!r} -> {p['intent']} {p['slots']} ({why})")
    others = [case[0] for case in GOLD if case not in GOLD_TIER4] + list(MUST_NOT_ACT)
    answered = []
    for text in others:
        p = run(text)
        if mg.answer_decision(p) == "answer":
            answered.append(f"{text!r} -> {p['intent']}")
    return {
        "n": len(GOLD_TIER4),
        "failed": failed,
        "others": len(others),
        "others_answered": answered,
    }


def not_a_question(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    others = [r for r in rows if r["intent"] == "None"]
    answered = [r["text"] for r in others if mg.answer_decision(run(r["text"])) == "answer"]
    return {
        "n": len(others),
        "answered_count": len(answered),
        "declined": 1 - len(answered) / len(others) if others else 1.0,
        "answered": answered[:20],
    }


def lone_words(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measured, not a gate: the held-out lone words ("armed", "the fast")
    the model itself would answer, without the one-word rule. The rule
    declines a word alone whatever the model says; this shows whether the
    data taught the model the same."""
    lone = [r for r in rows if r["template"].startswith("None#lone:")]
    answered = [
        r["text"]
        for r in lone
        if mg.answer_decision(run(r["text"]), one_word_rule=False) == "answer"
    ]
    return {"n": len(lone), "answered_count": len(answered), "answered": sorted(set(answered))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("candidate")
    parser.add_argument("--baseline", default=None, help="an earlier question model, to compare")
    parser.add_argument("--limit", type=int, default=None, help="max held-out rows")
    args = parser.parse_args(argv)

    cand_dir = mg.MODELS / args.candidate
    rows = mg.read_jsonl(cand_dir / "holdout_test.jsonl")[: args.limit]
    cand = mg.load(args.candidate)
    base = mg.load(args.baseline) if args.baseline else None

    gates: list[tuple[str, bool, str]] = []
    new_c = mg.score_rows(cand, rows)
    new_b = mg.score_rows(base, rows) if base else None
    weak = sorted(
        i
        for i, v in new_c.items()
        if not i.startswith("_")
        and (v["intent_acc"] < mg.INTENT_ACC_MIN or v["slot_f1"] < mg.SLOT_F1_MIN)
    )
    gates.append(
        (
            "Q1 new phrasings",
            not weak,
            f"n={new_c['_all']['n']} intent {new_c['_all']['intent_acc']:.4f} "
            f"slot F1 {new_c['_all']['slot_f1']:.4f}"
            + (
                f" (baseline {new_b['_all']['intent_acc']:.4f} / {new_b['_all']['slot_f1']:.4f})"
                if new_b
                else ""
            )
            + (f"; below bar: {weak}" if weak else ""),
        )
    )
    g = gold(cand)
    gates.append(
        (
            "Q2 gold",
            not g["failed"] and not g["others_answered"],
            f"{g['n'] - len(g['failed'])}/{g['n']} questions right {g['failed']}; "
            f"{g['others'] - len(g['others_answered'])}/{g['others']} commands and fragments "
            f"not answered {g['others_answered']}",
        )
    )
    nq_c = not_a_question(cand, rows)
    nq_b = not_a_question(base, rows) if base else None
    gates.append(
        (
            "Q3 not a question",
            nq_c["declined"] >= mg.NEGATIVE_REJECT_MIN,
            f"{nq_c['n'] - nq_c['answered_count']}/{nq_c['n']} not answered "
            f"({nq_c['declined']:.2%}"
            + (f"; baseline {nq_b['declined']:.2%}" if nq_b else "")
            + ")",
        )
    )
    lone = lone_words(cand, rows)
    p95 = mg.latency(cand, [r["text"] for r in rows[:300]])
    gates.append(("Q4 latency", p95 < mg.P95_MS_MAX, f"p95 {p95:.1f} ms"))

    report = {
        "candidate": args.candidate,
        "baseline": args.baseline,
        "kind": "question model",
        "new_phrasings": {"candidate": new_c, "baseline": new_b},
        "gold": g,
        "not_a_question": {"candidate": nq_c, "baseline": nq_b},
        "lone_words_without_rule": lone,
        "latency_p95_ms": p95,
        "gates": {name: {"pass": ok, "detail": d} for name, ok, d in gates},
    }
    (cand_dir / "gate.json").write_text(json.dumps(report, indent=1))

    vs = f" vs {args.baseline}" if args.baseline else ""
    print(f"\nQuestion-model gate: {args.candidate}{vs}\n")
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<18} {detail}")
    print("\n  per intent (new phrasings): intent acc / slot F1")
    for intent, v in new_c.items():
        if not intent.startswith("_"):
            print(f"    {intent:<22} {v['intent_acc']:.3f} / {v['slot_f1']:.3f}  (n={v['n']})")
    print(f"\n  commands and negatives answered: {nq_c['answered_count']}")
    for text in nq_c["answered"]:
        print(f"    {text!r}")
    print(
        f"\n  measured, not a gate: held-out lone words the model alone answers "
        f"(no one-word rule): {lone['answered_count']}/{lone['n']} {lone['answered']}"
    )
    passed = all(ok for _, ok, _ in gates)
    print(f"\n  {'ALL GATES PASS' if passed else 'GATE FAILED'}  -> {cand_dir / 'gate.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
