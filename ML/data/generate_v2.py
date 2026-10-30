"""Training data v2: generated from the command table (Phase E1).

Every command's templates are filled with values said the ways pilots say
them (``spoken.py``), wrapped in polite / terse variations, and a share of
rows get recognition noise seen in real flights. Negatives teach the model
what is *not* a command: chatter, command-word salad, bare fragments.

Each row is checked before it's kept: the text is normalized exactly as at
runtime and every slot value must appear in it as a whole word, stored in
that normalized form (what the model is trained to tag). Rows that don't
round-trip are dropped and counted, never silently mislabelled.

About 15% of each command's templates are held out: rows from them form the
test split, so the score measures new phrasings, not memorised ones.

Usage::

    python ML/data/generate_v2.py                  # -> ML/datasets/07_v2/aviation_v2_vN.jsonl
    python ML/data/generate_v2.py --per-intent 500 --out /tmp/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from typing import Any

_ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ML not in sys.path:
    sys.path.insert(0, _ML)

from vimaan_nlu.commands import COMMANDS, PREFIXES, SUFFIXES, Command  # noqa: E402
from vimaan_nlu.normalization import normalize_aviation_input  # noqa: E402

from data import spoken  # noqa: E402

SCHEMA_VERSION = 2
HOLDOUT_TEMPLATE_SHARE = 0.15
NOISE_SHARE = 0.2
NEGATIVE_SHARE = 0.1  # of the actionable rows
CONVERSATION_ROWS = 400  # per conversational intent

INSTANCE_SAY = {"1": "one", "2": "two"}
INSTANCE_ORD = {"1": ("first", "left"), "2": ("second", "right")}

# Out-of-scope chatter (label: None). Must not overlap the conversational
# intents' templates.
CHATTER = (
    "what's the weather like today",
    "how are you doing",
    "tell me something interesting",
    "that's a beautiful sunset",
    "i'm feeling hungry",
    "can you see the city lights",
    "hold on a second",
    "let me check the chart",
    "okay sounds good to me",
    "where did i put my phone",
    "the coffee is cold",
    "can you pass me the water",
    "i think we should turn back soon",
    "that was a really nice landing",
    "call my mom later",
    "look at that view",
    "did you hear that noise",
    "i need a break",
    "this seat is uncomfortable",
    "how long have you been flying",
    "nice job everyone",
    "let me think about it",
    "what do you think",
    "never mind",
    "sorry wrong button",
    "testing testing",
    "can you hear me",
    "is this thing on",
    "wait a moment",
    "i'll be right back",
)
# Leading words the recogniser inserted in real flights ("feet engine two
# off", "eighty engine two off") and fillers pilots say.
LEADING_NOISE = ("the", "feet", "eighty", "a", "uh", "so", "okay", "and", "now")
DROPPABLE = frozenset({"the", "to", "set", "please", "a", "and", "now"})


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def _frequency(rng: random.Random) -> str:
    khz = rng.choice(
        [rng.randrange(118000, 137000, 25), rng.randrange(118000, 137000, 5)]
    )  # 25 kHz channels and 8.33 kHz channels
    mhz = f"{khz / 1000:.3f}".rstrip("0")
    return mhz + "0" if mhz.endswith(".") else mhz


def _altitude(rng: random.Random) -> int:
    if rng.random() < 0.5:
        return rng.randrange(1000, 45001, 1000)
    return rng.randrange(1000, 45001, 100)


def render_number(kind: str, rng: random.Random) -> str:
    """A spoken rendering of a random value of ``kind``."""
    if kind == "heading":
        return rng.choice(spoken.heading(rng.randrange(0, 360)))
    if kind == "altitude":
        return rng.choice(spoken.altitude(_altitude(rng)))
    if kind == "flight_level":
        return rng.choice(spoken.flight_level(rng.randrange(100, 431, 10)))
    if kind == "frequency":
        return rng.choice(spoken.frequency(_frequency(rng)))
    if kind == "port":
        return rng.choice(("one", "two"))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def fill(c: Command, template: str, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Fill ``template``; returns (text, slot -> spoken surface)."""
    surfaces: dict[str, str] = {}

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.endswith("_verb"):
            slot = c.slots[name[: -len("_verb")]]
            value = rng.choice([v for v in slot.synonyms if slot.synonyms[v]])
            word = rng.choice(slot.synonyms[value])
            surfaces[name[: -len("_verb")]] = word
            return word
        if name == "instance_ord":
            word = rng.choice(INSTANCE_ORD[rng.choice(("1", "2"))])
            surfaces["instance"] = word
            return word
        slot = c.slots[name]
        if slot.kind == "number":
            word = render_number(slot.render or name, rng)
        elif slot.kind == "instance":
            word = INSTANCE_SAY[rng.choice(("1", "2"))]
        else:
            word = rng.choice(list(slot.synonyms))
        surfaces[name] = word
        return word

    text = re.sub(r"\{(\w+)\}", sub, template)
    return text, surfaces


def wrap(text: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES) if rng.random() < 0.4 else ""
    suffix = rng.choice(SUFFIXES) if rng.random() < 0.3 else ""
    return " ".join(f"{prefix} {text} {suffix}".split())


def add_noise(text: str, protected: set[str], rng: random.Random) -> tuple[str, str]:
    """Recognition noise from real logs: a stray leading word, or a dropped
    filler word. Never touches slot words."""
    words = text.split()
    if rng.random() < 0.5:
        return f"{rng.choice(LEADING_NOISE)} {text}", "lead"
    droppable = [i for i, w in enumerate(words) if w in DROPPABLE and w not in protected]
    if droppable and len(words) > 2:
        del words[rng.choice(droppable)]
        return " ".join(words), "drop"
    return f"{rng.choice(LEADING_NOISE)} {text}", "lead"


def label(text: str, surfaces: dict[str, str]) -> dict[str, str] | None:
    """Slot values as they appear in the normalized text, or None if any
    slot doesn't survive normalization as a whole word."""
    norm = normalize_aviation_input(text)
    slots = {}
    for name, surface in surfaces.items():
        value = normalize_aviation_input(surface)
        if not re.search(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])", norm):
            return None
        slots[name] = value
    return slots


def holdout_templates(c: Command, rng: random.Random) -> set[str]:
    n = max(2, math.ceil(len(c.templates) * HOLDOUT_TEMPLATE_SHARE))
    return set(rng.sample(sorted(c.templates), n))


def command_rows(
    c: Command, per_intent: int, rng: random.Random, stats: Counter[str]
) -> list[dict[str, Any]]:
    held = holdout_templates(c, rng)
    rows = []
    attempts = 0
    while len(rows) < per_intent and attempts < per_intent * 5:
        attempts += 1
        template = rng.choice(c.templates)
        text, surfaces = fill(c, template, rng)
        text = wrap(text, rng)
        noise = "none"
        if rng.random() < NOISE_SHARE:
            protected = {w for s in surfaces.values() for w in s.split()} | set(c.command_words)
            text, noise = add_noise(text, protected, rng)
        slots = label(text, surfaces)
        if slots is None:
            stats[f"dropped:{c.intent}"] += 1
            continue
        rows.append(
            {
                "v": SCHEMA_VERSION,
                "text": text,
                "intent": c.intent,
                "slots": slots,
                "template": f"{c.intent}#{c.templates.index(template)}",
                "split": "test" if template in held else "train",
                "noise": noise,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Negatives (label: None)
# ---------------------------------------------------------------------------
def _vocab() -> tuple[list[str], set[str], set[str]]:
    """(filler words, command words, value words) from the table."""
    command_words = {w for c in COMMANDS for w in c.command_words}
    value_words = (
        set(spoken.ONES)
        | set(spoken.TEENS)
        | set(spoken.TENS)
        | {
            "niner",
            "hundred",
            "thousand",
            "decimal",
            "point",
        }
    )
    for c in COMMANDS:
        for slot in c.slots.values():
            for value, syns in slot.synonyms.items():
                value_words.add(value)
                for s in syns:
                    value_words.update(s.split())
    words = set()
    for c in COMMANDS:
        if not c.actionable:
            continue  # "hello vimaan" is a greeting, not salad
        for t in c.templates:
            words.update(re.sub(r"\{\w+\}", " ", t).split())
    filler = sorted(words - command_words - value_words - {"_"})
    return filler, command_words, value_words


def negative_rows(n: int, rng: random.Random) -> list[dict[str, Any]]:
    filler, command_words, value_words = _vocab()
    numbers = [w for w in value_words if w in spoken.ONES or w in spoken.TENS or w in spoken.TEENS]
    states = ["on", "off", "up", "down"]
    rows = []
    for i in range(n):
        kind = i % 4
        if kind == 0:  # chatter
            text = rng.choice(CHATTER)
        elif kind == 1:  # word salad: at most one command word, no values
            words = rng.sample(filler, rng.randint(2, 4))
            if rng.random() < 0.6:
                words.insert(rng.randrange(len(words) + 1), rng.choice(sorted(command_words)))
            text = " ".join(words)
        elif kind == 2:  # bare numbers ("seventy", "two forty")
            text = rng.choice(
                [
                    rng.choice(numbers),
                    spoken.pilot_group(str(rng.randrange(101, 999))) or "two forty",
                    spoken.two_digit(rng.randrange(10, 99)),
                ]
            )
        else:  # numbers + state, no command word ("eighty two on")
            text = f"{rng.choice([spoken.two_digit(rng.randrange(10, 99)), rng.choice(spoken.ONES)])} {rng.choice(states)}"
            if rng.random() < 0.3:
                text = rng.choice(states)
        rows.append(
            {
                "v": SCHEMA_VERSION,
                "text": text,
                "intent": "None",
                "slots": {},
                "template": f"None#neg{kind}",
                "split": "test" if rng.random() < HOLDOUT_TEMPLATE_SHARE else "train",
                "noise": "none",
            }
        )
    return rows


def conversation_rows(c: Command, n: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for _ in range(n):
        template = rng.choice(c.templates)
        rows.append(
            {
                "v": SCHEMA_VERSION,
                "text": wrap(template, rng),
                "intent": c.intent,
                "slots": {},
                "template": f"{c.intent}#{c.templates.index(template)}",
                "split": "test" if rng.random() < HOLDOUT_TEMPLATE_SHARE else "train",
                "noise": "none",
            }
        )
    return rows


def generate(per_intent: int = 3000, seed: int = 42) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    actionable = [c for c in COMMANDS if c.actionable]
    for c in actionable:
        rows += command_rows(c, per_intent, rng, stats)
    rows += negative_rows(int(len(rows) * NEGATIVE_SHARE), rng)
    for c in COMMANDS:
        if not c.actionable and c.intent != "None":
            rows += conversation_rows(c, CONVERSATION_ROWS, rng)
    rows += [
        {
            "v": SCHEMA_VERSION,
            "text": wrap(t, rng),
            "intent": "None",
            "slots": {},
            "template": "None#table",
            "split": "train",
            "noise": "none",
        }
        for t in next(c for c in COMMANDS if c.intent == "None").templates
        for _ in range(20)
    ]
    # A negative must never repeat a text that is labelled as something else.
    labelled = {r["text"] for r in rows if r["intent"] != "None"}
    kept = [r for r in rows if r["intent"] != "None" or r["text"] not in labelled]
    stats["dropped:negative_collides"] = len(rows) - len(kept)
    rows = kept
    rng.shuffle(rows)
    stats["rows"] = len(rows)
    for r in rows:
        stats[f"intent:{r['intent']}"] += 1
        stats[f"split:{r['split']}"] += 1
        stats[f"noise:{r['noise']}"] += 1
    return rows, stats


def _next_path(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(directory, f"aviation_v2_v{n}.jsonl")):
        n += 1
    return os.path.join(directory, f"aviation_v2_v{n}.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-intent", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    rows, stats = generate(args.per_intent, args.seed)
    out = args.out or _next_path(os.path.join(_ML, "datasets", "07_v2"))
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out.replace(".jsonl", ".stats.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(stats.items())), fh, indent=1)
    print(f"wrote {len(rows)} rows to {out}")
    for k, v in sorted(stats.items()):
        if k.startswith(("split:", "noise:", "dropped:")):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
