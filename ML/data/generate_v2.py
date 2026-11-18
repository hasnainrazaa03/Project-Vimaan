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
import hashlib
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

from vimaan_nlu.commands import (  # noqa: E402
    COMMANDS,
    INSTANCE_WORDS,
    PREFIXES,
    QUESTION_PREFIXES,
    QUESTION_SUFFIXES,
    SUFFIXES,
    Command,
    Slot,
)
from vimaan_nlu.normalization import normalize_aviation_input  # noqa: E402
from vimaan_nlu.questions import QUESTION_WORDS  # noqa: E402
from vimaan_nlu.stt_vocabulary import NUMBER_WORDS, entry_words, load_vocabulary  # noqa: E402

from data import spoken  # noqa: E402

SCHEMA_VERSION = 2
HOLDOUT_TEMPLATE_SHARE = 0.15
# Development: one more template per command, the next in the same hash order,
# also never trained on. The trainer keeps the checkpoint that does best on
# them, and training ideas are compared on them, so the test templates (the
# gate's new phrasings) are only read for a final candidate.
DEV_TEMPLATES = 1
# A question family has more templates (35-85), and with one dev template its
# dev score was all or nothing: v27's ask_autopilot read 0.000 on dev from its
# one dev template ("is the flight director on") and 0.873 on the test ones.
# Five, not three: with three, q1 attempt 2's dev chose a run that failed on
# test phrasings (0.877 ask_autopilot) that dev had no phrasing like.
QUESTION_DEV_TEMPLATES = 5
# The question model (Tier 4): rows per command, labelled None ("not a
# question"), so it declines a command the command model refused.
QUESTION_MODEL_PER_COMMAND = 1000
# A question family's word said alone ("armed", "torque", "next") is not a
# question: q1 attempt 1 never saw one and each seed answered different ones
# (14-23 of 64), "armed" as the autopilot. Whole words are held out, like
# templates, so dev and test ask whether the idea carries to unseen words.
LONE_WORD_FORMS = ("{w}", "uh {w}", "um {w}", "okay {w}", "{w} uh", "the {w}")
LONE_WORD_REPEAT = 4  # rows per form
LONE_DEV_WORDS = 6
# Fragments the gold set checks (tests/test_gold_nlu.py MUST_NOT_ACT) are
# never trained on, so gold still asks whether the idea carries over.
GOLD_FRAGMENTS = frozenset({"armed"})
DEV_ROW_SHARE = 0.05  # of the negatives and conversation rows
NOISE_SHARE = 0.2
NEGATIVE_SHARE = 0.1  # of the actionable rows
CONVERSATION_ROWS = 400  # per conversational intent
TWO_DIGIT_HEADING_SHARE = 0.15  # "change heading to eighty", beside three-digit forms
# "speed eighty" (60-95 kt), beside three-digit speeds. Without it v20 (data
# v9) left "eighty" untagged after "speed": the stray leading "eighty" of
# LEADING_NOISE grows with the number of commands, the speed rows don't.
TWO_DIGIT_SPEED_SHARE = 0.15

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
# A stray word in a command: the first sound lost at key-down ("ice director
# on" for "flight director on"; the mic opens ~85 ms after the press) or noise
# at key-up. Since Tier 3 many of the words the recogniser lands on belong to
# other commands, and v26 took the stray word for the command: "speed
# autopilot on" -> "which speed brake?", "fuel com two to ..." -> "on or
# off?". The stray word is one that can't make its own command with the rest
# of the sentence ("fuel" never goes before "... on"), so the label stays right.
STRAY_SHARE = 0.3  # of the noisy rows
# Where it goes. On synthetic voices through Vosk (400 dev rows x 4 voices,
# whole and with 150 ms cut from the start) the first word was replaced
# 163-234 times and the last 17-25 times, and no word was ever added at either
# end. Data v15 put 55% of its stray words in front of a whole command: the
# second v26 learned "parking start engine one" as the engine and took a
# leading "parking" for noise even where it was the command. Insertions stay,
# rare, for the noise of a real microphone.
STRAY_REPLACE_FIRST = 0.75
STRAY_REPLACE_LAST = 0.15  # the rest: added before or after, evenly

# How the recogniser hears a question. The grammar has no "what's" (only plain
# words go in) and the owner's "our" / "are" came back as "are" or the letter
# "r" (40 recordings, 2026-11-16: "what are altitude", "what r airspeed",
# "what the heading", "r the landing lights on").
HEARD_QUESTION_SHARE = 0.5


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def _frequency(rng: random.Random) -> str:
    khz = rng.choice(
        [rng.randrange(118000, 137000, 25), rng.randrange(118000, 137000, 5)]
    )  # 25 kHz channels and 8.33 kHz channels
    mhz = f"{khz / 1000:.3f}".rstrip("0")
    return mhz + "0" if mhz.endswith(".") else mhz


def _nav_frequency(rng: random.Random) -> str:
    """108.00-117.95 MHz on 50 kHz channels."""
    mhz = f"{rng.randrange(10800, 11796, 5) / 100:.2f}".rstrip("0")
    return mhz + "0" if mhz.endswith(".") else mhz


def _squawk(rng: random.Random) -> str:
    if rng.random() < 0.2:  # the codes pilots hear most
        return rng.choice(("1200", "7000", "2000", "0120", "4000"))
    return "".join(rng.choice("01234567") for _ in range(4))


def _altitude(rng: random.Random) -> int:
    if rng.random() < 0.5:
        return rng.randrange(1000, 45001, 1000)
    return rng.randrange(1000, 45001, 100)


def render_number(kind: str, rng: random.Random) -> str:
    """A spoken rendering of a random value of ``kind``."""
    if kind == "heading":
        if rng.random() < TWO_DIGIT_HEADING_SHARE:
            return spoken.two_digit(rng.randrange(10, 100))
        return rng.choice(spoken.heading(rng.randrange(0, 360)))
    if kind == "altitude":
        return rng.choice(spoken.altitude(_altitude(rng)))
    if kind == "flight_level":
        return rng.choice(spoken.flight_level(rng.randrange(100, 431, 10)))
    if kind == "frequency":
        return rng.choice(spoken.frequency(_frequency(rng)))
    if kind == "port":
        return rng.choice(("one", "two"))
    # Tier 2
    if kind == "nav_frequency":
        return rng.choice(spoken.frequency(_nav_frequency(rng)))
    if kind == "squawk":
        return rng.choice(spoken.squawk(_squawk(rng)))
    # Tier 1
    if kind == "speed":
        if rng.random() < TWO_DIGIT_SPEED_SHARE:
            return rng.choice(spoken.speed(rng.randrange(60, 100, 5)))
        return rng.choice(spoken.speed(rng.randrange(60, 401, 5)))
    if kind == "mach":
        return rng.choice(spoken.mach(f"0.{rng.randrange(40, 96):02d}"))
    if kind == "vs":
        return rng.choice(spoken.vertical_speed(rng.randrange(100, 6001, 100)))
    if kind == "pressure":
        if rng.random() < 0.6:  # inHg (the Americas) or hPa (most of the world)
            return rng.choice(spoken.pressure(f"{rng.randrange(2800, 3151) / 100:.2f}"))
        return rng.choice(spoken.pressure(str(rng.randrange(940, 1061))))
    if kind == "autobrake":
        return rng.choice(("one", "two", "three"))
    if kind == "flaps":
        return rng.choice(spoken.flaps(rng.choice((1, 2, 3, 5, 10, 15, 20, 25, 30, 40))))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
# {slot}, {slot_verb}, {instance_ord} or {slot=word}.
PLACEHOLDER = re.compile(r"\{(\w+)(?:=([^}]+))?\}")


def fill(c: Command, template: str, rng: random.Random) -> tuple[str, dict[str, str]]:
    """Fill ``template``; returns (text, slot -> spoken surface)."""
    surfaces: dict[str, str] = {}

    def sub(match: re.Match[str]) -> str:
        name, fixed = match.group(1), match.group(2)
        if fixed is not None:  # {slot=word}: that word, tagged as the slot
            surfaces[name] = fixed
            return fixed
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

    text = re.sub(PLACEHOLDER, sub, template)
    return text, surfaces


def wrap(text: str, rng: random.Random) -> str:
    prefix = rng.choice(PREFIXES) if rng.random() < 0.4 else ""
    suffix = rng.choice(SUFFIXES) if rng.random() < 0.3 else ""
    return " ".join(f"{prefix} {text} {suffix}".split())


def wrap_question(text: str, rng: random.Random) -> str:
    prefix = rng.choice(QUESTION_PREFIXES) if rng.random() < 0.3 else ""
    suffix = rng.choice(QUESTION_SUFFIXES) if rng.random() < 0.2 else ""
    return " ".join(f"{prefix} {text} {suffix}".split())


def heard_question(text: str, rng: random.Random) -> str:
    """The question as the recogniser writes it: "what's" as "what", "our"
    as "are" or "r", "are the" as "r the". Slot words are never touched:
    only these function words change."""
    words = text.split()
    out = []
    for i, w in enumerate(words):
        if w == "what's":
            out += rng.choice((["what"], ["what"], ["what", "is"]))
        elif w == "our" and out and out[-1] != "is":
            out.append(rng.choice(("are", "r", "our")))
        elif w == "are" and i + 1 < len(words) and words[i + 1] == "the":
            out.append(rng.choice(("r", "are")))
        else:
            out.append(w)
    return " ".join(out)


def add_noise(
    text: str, protected: set[str], rng: random.Random, slot_words: frozenset[str] = frozenset()
) -> tuple[str, str]:
    """Recognition noise from real logs: a mishearing, a stray leading word,
    or a dropped filler word. Never touches slot words."""
    words = text.split()
    # Heard in flight: "parking the off" for "parking brake off". Not where
    # "parking brake" is the slot ("is the parking brake set").
    if "parking brake" in text and "brake" not in slot_words and rng.random() < 0.4:
        return text.replace("parking brake", "parking the", 1), "mishear"
    if rng.random() < 0.5:
        return f"{rng.choice(LEADING_NOISE)} {text}", "lead"
    droppable = [i for i, w in enumerate(words) if w in DROPPABLE and w not in protected]
    if droppable and len(words) > 2:
        del words[rng.choice(droppable)]
        return " ".join(words), "drop"
    return f"{rng.choice(LEADING_NOISE)} {text}", "lead"


def _could_make(b: Command, words: set[str]) -> bool:
    """Whether ``words`` hold something for every slot ``b`` requires."""
    for name in b.required:
        slot = b.slots[name]
        if slot.kind == "number":
            found = bool(words & set(NUMBER_WORDS))
        elif slot.kind == "instance":
            found = bool(words & set(INSTANCE_WORDS))
        else:
            surfaces = {
                w for v, syns in slot.synonyms.items() for s in (v, *syns) for w in str(s).split()
            }
            found = bool(words & surfaces)
        if not found:
            return False
    return True


def _stray_pool() -> dict[str, list[Command]]:
    """Command words that may turn up stray, each with the commands it is a
    word of. Not: whole commands ("ident"), numbers, single letters (they
    join neighbours: "v s"), or words the recogniser can't output."""
    heard = entry_words(load_vocabulary())
    owners: dict[str, list[Command]] = {}
    for c in COMMANDS:
        if c.actionable:
            for w in c.command_words:
                owners.setdefault(w, []).append(c)
    skip = whole_commands() | set(NUMBER_WORDS)
    return {w: owners[w] for w in sorted(owners) if w in heard and w not in skip and len(w) > 1}


_STRAY_POOL: dict[str, list[Command]] | None = None


def stray_words(c: Command, text: str) -> list[str]:
    """Words that could stand stray beside ``text`` (a command ``c``): none of
    the commands they belong to could be made with the sentence's words."""
    global _STRAY_POOL
    if _STRAY_POOL is None:
        _STRAY_POOL = _stray_pool()
    words = set(text.split())
    # Nor a value of this command's own: "auto strobes on" is no stray word.
    own = set(c.command_words) | {
        w
        for slot in c.slots.values()
        for v, syns in slot.synonyms.items()
        for s in (v, *syns)
        for w in str(s).split()
    }
    return [
        w
        for w, owners in _STRAY_POOL.items()
        if w not in words
        and w not in own
        and not any(b.intent != c.intent and _could_make(b, words | {w}) for b in owners)
    ]


_TEMPLATE_WORDS: dict[str, tuple[frozenset[str], ...]] = {}


def _phrasings(c: Command) -> tuple[frozenset[str], ...]:
    """Each of ``c``'s templates as the words it is made of, besides values."""
    if c.intent not in _TEMPLATE_WORDS:
        _TEMPLATE_WORDS[c.intent] = tuple(
            frozenset(PLACEHOLDER.sub(lambda m: f" {m.group(2) or ''} ", t).split())
            for t in c.templates
        )
    return _TEMPLATE_WORDS[c.intent]


def _template_words(c: Command) -> frozenset[str]:
    """The words ``c``'s phrasings are made of, besides its values."""
    return frozenset().union(*_phrasings(c))


def _takes(d: Command, slot: Slot, surface: str) -> bool:
    """Could ``d`` take this value: a number of the same sort (a port, a
    frequency, an altitude ...), an instance, or a choice it says the same way."""
    for other in d.slots.values():
        if other.kind != slot.kind:
            continue
        if slot.kind == "number":
            if other.render == slot.render:
                return True
            continue
        if slot.kind == "instance":
            return True
        if any(surface == str(s) for v, syns in other.synonyms.items() for s in (v, *syns)):
            return True
    return False


def _alternatives(c: Command, rest: set[str], surfaces: dict[str, str]) -> list[str]:
    """Other commands the words ``rest`` could be, while they no longer say
    ``c``: ``d`` could be made of them, could take every value in them and
    shares a word with them, and every phrasing of ``c`` has a word that
    tells it from ``d`` and is not among them. "two transfer" is the NAV
    swap's or the COM swap's alike (every NAV swap says "nav"); "com one
    one one nine decimal seven" is still the active frequency (it is said
    without a word of its own, and the standby always says "standby"); "nav
    two standby one one six decimal nine" is no swap (a swap takes no
    frequency)."""
    mine = _template_words(c)
    out = []
    for d in COMMANDS:
        if not d.actionable or d.intent == c.intent or not _could_make(d, rest):
            continue
        if not all(_takes(d, c.slots[name], surface) for name, surface in surfaces.items()):
            continue
        theirs = _template_words(d) | set(d.command_words)
        markers = mine - theirs  # words that say c rather than d
        if not rest & theirs or rest & markers:
            continue
        if not markers or all(p & markers for p in _phrasings(c)):
            out.append(d.intent)
    return out


def _replaceable(
    c: Command,
    words: list[str],
    i: int,
    slot_words: set[str],
    surfaces: dict[str, str] | None = None,
) -> bool:
    """May ``words[i]`` be lost to a stray word? Never a slot word, and never
    a word without which what is left is another command's as much:
    "nav two transfer" without "nav" is the COM swap's too (data v16: 28 train
    and 12 dev rows nobody could answer, 12 of the 16 dev rows any v26 run
    acted on wrongly). One of the command's own words only when another stays
    and nothing else could be made of what is left: "ice director on" is
    still the director, while "... speed one thousand" without "vertical" is a
    speed of its own."""
    w = words[i]
    if w in slot_words:
        return False
    rest = set(words[:i] + words[i + 1 :])
    if _alternatives(c, rest, surfaces or {}):
        return False
    if w not in c.command_words:
        return True
    if not rest & set(c.command_words):
        return False
    return not any(
        d.actionable
        and d.intent != c.intent
        and rest & set(d.command_words)
        and _could_make(d, rest)
        for d in COMMANDS
    )


def add_stray(
    c: Command,
    text: str,
    word: str,
    slot_words: set[str],
    rng: random.Random,
    surfaces: dict[str, str] | None = None,
) -> str | None:
    """``word`` in place of the first or the last word, or rarely added at an
    end; None when the place drawn can't take it (the row gets other noise)."""
    words = text.split()
    r = rng.random()
    if r < STRAY_REPLACE_FIRST + STRAY_REPLACE_LAST:
        i = 0 if r < STRAY_REPLACE_FIRST else len(words) - 1
        if len(words) < 2 or not _replaceable(c, words, i, slot_words, surfaces):
            return None
        return " ".join([*words[:i], word, *words[i + 1 :]])
    return f"{word} {text}" if rng.random() < 0.5 else f"{text} {word}"


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


def _ranked(c: Command) -> list[str]:
    return sorted(c.templates, key=lambda t: hashlib.sha256(f"{c.intent}|{t}".encode()).hexdigest())


def _holdout_count(c: Command) -> int:
    return max(2, math.ceil(len(c.templates) * HOLDOUT_TEMPLATE_SHARE))


def holdout_templates(c: Command) -> set[str]:
    """Held-out templates, picked by a stable hash of the template: adding a
    phrasing elsewhere must not reshuffle them, or scores from different
    dataset versions can't be compared."""
    return set(_ranked(c)[: _holdout_count(c)])


def dev_templates(c: Command) -> set[str]:
    """Development templates: the next DEV_TEMPLATES in the same order, so
    the test templates are exactly what they would be without them."""
    n = _holdout_count(c)
    k = QUESTION_DEV_TEMPLATES if c.is_question else DEV_TEMPLATES
    return set(_ranked(c)[n : n + k])


def _row_split(rng: random.Random) -> str:
    """test / dev / train for a row with no template of its own (negatives,
    conversation). One draw, as before dev existed, so the test rows are the
    same."""
    r = rng.random()
    if r < HOLDOUT_TEMPLATE_SHARE:
        return "test"
    return "dev" if r < HOLDOUT_TEMPLATE_SHARE + DEV_ROW_SHARE else "train"


def command_rows(
    c: Command, per_intent: int, rng: random.Random, stats: Counter[str]
) -> list[dict[str, Any]]:
    held = holdout_templates(c)
    dev = dev_templates(c)
    rows = []
    attempts = 0
    while len(rows) < per_intent and attempts < per_intent * 5:
        attempts += 1
        template = rng.choice(c.templates)
        text, surfaces = fill(c, template, rng)
        text = wrap_question(text, rng) if c.is_question else wrap(text, rng)
        noise = "none"
        if c.is_question and rng.random() < HEARD_QUESTION_SHARE:
            heard = heard_question(text, rng)
            if heard != text:
                text, noise = heard, "heard"
        if noise == "none" and rng.random() < NOISE_SHARE:
            slot_words = {w for s in surfaces.values() for w in s.split()}
            # A question keeps its question word: without it "is the gear
            # down" is the command "gear down" (vimaan_nlu.questions).
            protected = slot_words | set(c.command_words) | set(QUESTION_WORDS)
            # Stray words come from other commands: never in a question.
            strays = (
                stray_words(c, text) if not c.is_question and rng.random() < STRAY_SHARE else []
            )
            strayed = strays and add_stray(c, text, rng.choice(strays), slot_words, rng, surfaces)
            if strayed:
                text, noise = strayed, "stray"
            else:
                text, noise = add_noise(text, protected, rng, frozenset(slot_words))
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
                "split": "test" if template in held else "dev" if template in dev else "train",
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
            # Every placeholder, "{xpdr_mode=ident}" too: left in, it reached
            # the negatives verbatim and taught "ident" as noise (data v9/v10).
            words.update(re.sub(r"\{[^}]*\}", " ", t).split())
    filler = sorted(words - command_words - value_words - {"_"})
    return filler, command_words, value_words


def whole_commands() -> set[str]:
    """Words that are a complete command on their own: a template that is just
    a fixed value ("{xpdr_mode=ident}": "ident" presses IDENT)."""
    return {
        m.group(1)
        for c in COMMANDS
        for t in c.templates
        if (m := re.fullmatch(r"\{\w+=([^}]+)\}", t))
    }


def negative_rows(n: int, rng: random.Random) -> list[dict[str, Any]]:
    filler, command_words, value_words = _vocab()
    # Salad may hold one command word, but not one that is a whole command:
    # "put ident both switch" as None contradicts bare "ident", and v22 then
    # called "put the transponder in ident" not a command.
    salad_commands = sorted(command_words - whole_commands())
    # Sorted: a set's order changes with every process, and the dataset must
    # come out the same for the same seed.
    numbers = sorted(
        w for w in value_words if w in spoken.ONES or w in spoken.TENS or w in spoken.TEENS
    )
    states = ["on", "off", "up", "down"]
    rows = []
    for i in range(n):
        kind = i % 4
        if kind == 0:  # chatter
            text = rng.choice(CHATTER)
        elif kind == 1:  # word salad: at most one command word, no values
            words = rng.sample(filler, rng.randint(2, 4))
            if rng.random() < 0.6:
                words.insert(rng.randrange(len(words) + 1), rng.choice(salad_commands))
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
                "split": _row_split(rng),
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
                "split": _row_split(rng),
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
    # Tier 4: questions, built like commands (slots, held-out templates).
    commands_text = {r["text"] for r in rows if r["intent"] != "None"}
    for c in COMMANDS:
        if c.is_question:
            asked = command_rows(c, per_intent, rng, stats)
            # A question never repeats a command's words: the same text can't
            # be both "act" and "answer".
            kept = [r for r in asked if r["text"] not in commands_text]
            stats[f"dropped:question_is_command:{c.intent}"] = len(asked) - len(kept)
            rows += kept
    for c in COMMANDS:
        if not c.actionable and not c.is_question and c.intent != "None":
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


def _not_a_question(r: dict[str, Any], kind: str) -> dict[str, Any]:
    """A command or greeting row relabelled for the question model: None, no
    slots, its template kept (``None#cmd:set_flaps#3``) so its split and the
    gate's per-template view still hold."""
    return {**r, "intent": "None", "slots": {}, "template": f"None#{kind}:{r['template']}"}


def lone_words() -> list[str]:
    """The question families' words that mean nothing alone: every command
    word of a question family that is neither a reading answered alone
    (questions.BARE_READINGS) nor a system name a command asks about
    (validators.bare_command: "flaps" asks which way)."""
    from vimaan_nlu.questions import BARE_READINGS
    from vimaan_nlu.validators import bare_command

    words = {w for c in COMMANDS if c.is_question for w in c.command_words}
    return sorted(
        w
        for w in words
        if w not in BARE_READINGS and w not in QUESTION_WORDS and bare_command(w) is None
    )


def lone_word_split() -> dict[str, str]:
    """test / dev / train per lone word, by a stable hash of the word (adding
    a word elsewhere doesn't reshuffle them). The gold fragments are test."""
    words = sorted(lone_words(), key=lambda w: hashlib.sha256(f"lone|{w}".encode()).hexdigest())
    n_test = max(2, math.ceil(len(words) * HOLDOUT_TEMPLATE_SHARE))
    split = {}
    for i, w in enumerate(words):
        if i < n_test or w in GOLD_FRAGMENTS:
            split[w] = "test"
        elif i < n_test + LONE_DEV_WORDS:
            split[w] = "dev"
        else:
            split[w] = "train"
    return split


def lone_word_rows() -> list[dict[str, Any]]:
    return [
        {
            "v": SCHEMA_VERSION,
            "text": form.format(w=w),
            "intent": "None",
            "slots": {},
            "template": f"None#lone:{w}",
            "split": split,
            "noise": "none",
        }
        for w, split in sorted(lone_word_split().items())
        for form in LONE_WORD_FORMS
        for _ in range(LONE_WORD_REPEAT)
    ]


def generate_questions(
    per_intent: int = 4000,
    seed: int = 42,
    per_command: int = QUESTION_MODEL_PER_COMMAND,
    extra: str | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """The question model's data (Tier 4): every question family, and the
    commands, greetings and negatives as None ("not a question").

    The command model (v26) never sees these questions, so question data
    can't move a command; the question model has no commands, so whatever it
    gets wrong, it can only answer. Commands keep their own held-out
    templates: the question model's test then asks whether it declines
    commands phrased in ways it never saw.

    ``extra`` (one of QUESTION_EXTRAS) appends that kind's extra training rows
    after the rest, which stay exactly as they are without it."""
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for c in COMMANDS:
        if c.actionable:
            rows += [_not_a_question(r, "cmd") for r in command_rows(c, per_command, rng, stats)]
    rows += negative_rows(int(len(rows) * NEGATIVE_SHARE), rng)
    commands_text = {r["text"] for r in rows}
    for c in COMMANDS:
        if c.is_question:
            asked = command_rows(c, per_intent, rng, stats)
            kept = [r for r in asked if r["text"] not in commands_text]
            stats[f"dropped:question_is_command:{c.intent}"] = len(asked) - len(kept)
            rows += kept
    for c in COMMANDS:
        if not c.actionable and not c.is_question and c.intent != "None":
            rows += [
                _not_a_question(r, "conv") for r in conversation_rows(c, CONVERSATION_ROWS, rng)
            ]
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
    rows += lone_word_rows()
    asked_text = {r["text"] for r in rows if r["intent"] != "None"}
    kept = [r for r in rows if r["intent"] != "None" or r["text"] not in asked_text]
    stats["dropped:negative_collides"] = len(rows) - len(kept)
    rows = kept
    rng.shuffle(rows)
    if extra:
        rows += extra_rows(extra, rows, seed, stats)
    stats["rows"] = len(rows)
    for r in rows:
        stats[f"intent:{r['intent']}"] += 1
        stats[f"split:{r['split']}"] += 1
        stats[f"noise:{r['noise']}"] += 1
    return rows, stats


# ---------------------------------------------------------------------------
# Extra question rows (q1 round 4, 2026-11-17)
# ---------------------------------------------------------------------------
# q1 attempt 3 failed only on a request with no carrier ("quick update"), and
# the dev probe (ML/data/question_probe_v1.jsonl) found the same gap in every
# family ("transponder code please", "autopilot modes please"). Three ways to
# teach requests without "what's", "give me", "tell me":
#   terse         "{topic} {word}": a family's topic and a request word
#   terse_update  the same, "update" one of the request words
#   carrier       each training template with its carrier taken off
# The rows are appended after every other row (their own random stream), so
# each dev and test row is exactly what it was without them; they are train
# only, and never a test or dev template, a test, dev or None row, a probe
# phrasing, or "quick" (attempt 3's failed test phrasing must stay unseen).
QUESTION_EXTRAS = ("terse", "terse_update", "carrier")
EXTRA_PER_FAMILY = 300
TERSE_WORDS = ("check", "status", "reading", "report")
TERSE_TOPICS: dict[str, tuple[str, ...]] = {
    "ask_flight_data": ("{reading}", "current {reading}", "our {reading}"),
    "ask_fuel": ("fuel", "fuel quantity", "fuel on board", "fuel remaining", "fuel total"),
    "ask_fuel_time": ("endurance", "fuel endurance", "endurance remaining", "flying time"),
    "ask_engine": (
        "{engine_reading}",
        "engine {instance} {engine_reading}",
        "engine {instance}",
        "engine",
        "engines",
    ),
    "ask_com": ("com {com_port}", "com {com_port} {which}", "com {com_port} frequency", "radio"),
    "ask_nav": ("nav {nav_port}", "nav {nav_port} {which}", "nav {nav_port} frequency"),
    "ask_transponder": ("transponder", "squawk", "transponder code", "transponder mode"),
    "ask_autopilot": (
        "autopilot",
        "autopilot modes",
        "auto throttle",
        "flight director",
        "flight directors",
        "{ap_target} target",
        "selected {ap_target}",
        "{ap_target} bug",
    ),
    "ask_switch": ("{light} lights", "{sign} sign", "{ice} heat", "{power}", "{switch_system}"),
    "ask_waypoint": ("next way point", "next fix", "distance to the next fix"),
    "ask_time": ("time", "{zone} time", "clock", "current time"),
    "ask_status_generic": ("status", "flight status", "situation", "flight", "everything"),
}
# A template's leading carrier, longest first. "what's the" / "what's our"
# go only when what is left asks nothing else ("what's the autopilot doing").
CARRIERS = (
    "can you tell me",
    "can you check",
    "let me know",
    "give me",
    "tell me",
    "read me",
    "i need",
    "say",
)
WHAT_CARRIERS = ("what's the", "what's our", "what is the", "what is our")
VERBISH = frozenset(
    "set doing tuned showing going blowing flying happening wrong read reading look looking "
    "running doing last is are do does".split()
)
NEVER_EXTRA = frozenset({"quick"})
PROBE_FILE = os.path.join(_ML, "data", "question_probe_v1.jsonl")
# A phrasing's core: the generator's fillers, leading noise and question
# suffixes dropped, "what's" as "what" (heard_question): two texts with one
# core are one phrasing.
_CORE_DROP = frozenset(
    "uh um er oh okay ok so hey the a an and now feet eighty vimaan please right".split()
)


def core(text: str) -> str:
    return " ".join(
        "what" if w == "what's" else w for w in text.lower().split() if w not in _CORE_DROP
    )


def _held_templates(c: Command) -> set[str]:
    return holdout_templates(c) | dev_templates(c)


def terse_templates(c: Command, words: tuple[str, ...] = TERSE_WORDS) -> list[str]:
    out = []
    for topic in TERSE_TOPICS.get(c.intent, ()):
        for w in words:
            if w not in topic.split():
                out.append(f"{topic} {w}")
    return [t for t in out if t not in _held_templates(c)]


def _uncarried(template: str) -> str | None:
    for carrier in CARRIERS:
        if template.startswith(carrier + " "):
            rest = template[len(carrier) + 1 :]
            return rest[len("the ") :] if rest.startswith("the ") else rest
    for carrier in WHAT_CARRIERS:
        if template.startswith(carrier + " "):
            rest = template[len(carrier) + 1 :]
            if not VERBISH & set(rest.split()):
                return rest
    return None


def carrier_templates(c: Command) -> list[str]:
    """The family's training templates without their carrier (none of its
    templates, held out or not)."""
    held = _held_templates(c)
    out = []
    for t in c.templates:
        if t in held:
            continue
        rest = _uncarried(t)
        if rest and rest not in c.templates and rest not in out:
            out.append(rest)
    return out


def extra_templates(c: Command, kind: str) -> list[str]:
    if kind == "terse":
        return terse_templates(c)
    if kind == "terse_update":
        return terse_templates(c, (*TERSE_WORDS, "update"))
    if kind == "carrier":
        return carrier_templates(c)
    raise ValueError(f"extra must be one of {QUESTION_EXTRAS}, not {kind!r}")


def _probe_cores() -> set[str]:
    if not os.path.exists(PROBE_FILE):
        return set()
    with open(PROBE_FILE, encoding="utf-8") as fh:
        return {core(json.loads(line)["text"]) for line in fh if line.strip()}


def extra_rows(
    kind: str, rows: list[dict[str, Any]], seed: int, stats: Counter[str]
) -> list[dict[str, Any]]:
    """EXTRA_PER_FAMILY rows a question family of ``kind``, never one whose
    core is a dev, test or None row's of ``rows`` or a probe phrasing's."""
    from vimaan_nlu.questions import one_word
    from vimaan_nlu.stt_vocabulary import load_question_vocabulary

    heard = set(load_question_vocabulary())
    protected = _probe_cores() | {
        core(r["text"]) for r in rows if r["split"] != "train" or r["intent"] == "None"
    }
    rng = random.Random(f"{seed}|{kind}")
    out = []
    for c in COMMANDS:
        if not c.is_question:
            continue
        templates = extra_templates(c, kind)
        stats[f"extra_templates:{c.intent}"] = len(templates)
        attempts = 0
        made = 0
        while templates and made < EXTRA_PER_FAMILY and attempts < EXTRA_PER_FAMILY * 20:
            attempts += 1
            k = rng.randrange(len(templates))
            text, surfaces = fill(c, templates[k], rng)
            text = wrap_question(text, rng)
            words = set(text.split())
            why = (
                "protected"
                if core(text) in protected
                else "not_heard"
                if words - heard
                else "never"
                if words & NEVER_EXTRA
                else "one_word"
                if one_word(text)
                else None
            )
            slots = label(text, surfaces) if why is None else None
            if why is None and slots is None:
                why = "label"
            if why:
                stats[f"extra_dropped:{why}"] += 1
                continue
            made += 1
            out.append(
                {
                    "v": SCHEMA_VERSION,
                    "text": text,
                    "intent": c.intent,
                    "slots": slots,
                    "template": f"{c.intent}#{kind}:{k}",
                    "split": "train",
                    "noise": "none",
                }
            )
    return out


def _next_path(directory: str, stem: str = "aviation_v2_v") -> str:
    os.makedirs(directory, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(directory, f"{stem}{n}.jsonl")):
        n += 1
    return os.path.join(directory, f"{stem}{n}.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--per-intent", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--questions",
        action="store_true",
        help="the question model's data (questions_vN.jsonl): questions, and "
        "commands, greetings and negatives as None",
    )
    parser.add_argument(
        "--question-extra",
        choices=QUESTION_EXTRAS,
        default=None,
        help="with --questions: append this kind of extra training rows (round 4)",
    )
    args = parser.parse_args(argv)
    if args.question_extra and not args.questions:
        parser.error("--question-extra needs --questions")
    if args.questions:
        rows, stats = generate_questions(args.per_intent, args.seed, extra=args.question_extra)
        stem = "questions_v"
    else:
        rows, stats = generate(args.per_intent, args.seed)
        stem = "aviation_v2_v"
    out = args.out or _next_path(os.path.join(_ML, "datasets", "07_v2"), stem)
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
