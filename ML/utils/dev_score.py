"""Dev scored the way the gate scores: what the plugin would do.

Dev loss is a poor guide once dev intent accuracy is ~99.8%: a handful of
confidently wrong rows decide it, and two or three of them make it spike
(v26-g: 0.019 -> 0.22 between two epochs). This counts outcomes instead, with
scripts/model_gate.py's own ``actions`` and ``negatives`` (the plugin's
checks), on the dev templates: never trained on, and not the gate's test.

score = right - wrong_weight * (wrong + negatives acted on). A wrong action is
counted as ``wrong_weight`` lost commands: a refusal costs the pilot a repeat,
a wrong action does something they didn't ask for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

WRONG_WEIGHT = 5.0
# A question model's dev: points per phrasing (dev template), shared by its
# rows, so every phrasing counts the same however many rows it has.
PHRASING_POINTS = 100.0
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _gate():
    if str(_SCRIPTS) not in sys.path:
        sys.path.append(str(_SCRIPTS))
    import model_gate

    return model_gate


def score(run, rows: list[dict[str, Any]], wrong_weight: float = WRONG_WEIGHT) -> dict[str, Any]:
    """``run(text) -> prediction`` scored on ``rows`` (dataset rows)."""
    mg = _gate()
    act = mg.actions(run, rows)
    neg = mg.negatives(run, rows)
    wrong = act["wrong"] + neg["acted_count"]
    return {
        "n": act["n"],
        "right": act["right"],
        "rejected": act["rejected"],
        "wrong": act["wrong"],
        "negatives": neg["n"],
        "negatives_acted": neg["acted_count"],
        "score": act["right"] - wrong_weight * wrong,
    }


def question_outcomes(run, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A question model's outcomes on ``rows``: each question row answered
    right, misanswered (another question, or the wrong slots) or declined;
    each other row (commands, negatives, greetings: "not a question")
    declined or answered. Uses the gate's answer rule (model_gate.answer_decision)."""
    from vimaan_nlu.commands import command

    mg = _gate()
    out: dict[str, Any] = {"right": 0, "misanswered": 0, "declined": 0, "questions": 0}
    out.update({"others": 0, "others_answered": 0, "answered_examples": []})
    # Per phrasing: [rows, right, misanswered].
    phrasings: dict[str, list[int]] = {}
    for row in rows:
        want, slots = mg.canonical_expected(row)
        cmd = command(want)
        p = run(row["text"])
        answered = mg.answer_decision(p) == "answer"
        if cmd is not None and cmd.is_question:
            out["questions"] += 1
            counts = phrasings.setdefault(row.get("template") or want, [0, 0, 0])
            counts[0] += 1
            if not answered:
                out["declined"] += 1
            elif p["intent"] == want and all(
                mg.same_value(k, p["slots"].get(k), v) for k, v in slots.items()
            ):
                out["right"] += 1
                counts[1] += 1
            else:
                out["misanswered"] += 1
                counts[2] += 1
        else:
            out["others"] += 1
            if answered:
                out["others_answered"] += 1
                if len(out["answered_examples"]) < 15:
                    out["answered_examples"].append(f"{row['text']!r} -> {p['intent']}")
    out["phrasings"] = phrasings
    return out


def question_score(
    run, rows: list[dict[str, Any]], wrong_weight: float = WRONG_WEIGHT
) -> dict[str, Any]:
    """A question model's dev score: per phrasing (dev template), its share of
    rows answered right less its share misanswered, ``PHRASING_POINTS`` each;
    less ``wrong_weight`` per command or negative it answered (an answer
    there sounds like a read-back of a command that was never done).

    Per phrasing, not per row: q1 attempt 2 was chosen by one dev phrasing,
    "where are we headed" (~110 of 3,422 rows), that flipped from run to run,
    and the run it chose failed on phrasings dev weighed no more than that
    one. The commands' own dev score (``score``) skips question rows: v27 was
    chosen blind to its questions (2026-11-17)."""
    o = question_outcomes(run, rows)
    points = sum((right - wrong) / n for n, right, wrong in o["phrasings"].values())
    return {
        "n": o["questions"],
        "phrasings": len(o["phrasings"]),
        "right": o["right"],
        "misanswered": o["misanswered"],
        "declined": o["declined"],
        "others": o["others"],
        "others_answered": o["others_answered"],
        "score": round(PHRASING_POINTS * points - wrong_weight * o["others_answered"], 3),
    }


def live_runner(model, tokenizer, device, intent_map: dict, slot_map: dict):
    """A ``run`` for a model being trained (the plugin's own predict)."""
    from vimaan_nlu.inference import predict

    intent_rev = {v: k for k, v in intent_map.items()}
    slot_rev = {v: k for k, v in slot_map.items()}

    def run(text: str) -> dict[str, Any]:
        return predict(text, model, tokenizer, device, intent_rev, slot_rev)

    return run


def saved_runner(path: str | Path):
    """A ``run`` for a saved model, loaded as the plugin loads it."""
    return _gate().load(str(path))
