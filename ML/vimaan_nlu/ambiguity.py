"""Spot a target value that depends on hearing "to" or "two".

The two words sound the same. In-sim, Vosk scored both readings of "set altitude
to one thousand" identically and printed "two", which is also a correct way to
say 21,000, so the wrong target was set with no sign of doubt. Nothing in the
transcript can settle it, so when both readings are valid and different the
plugin asks instead of picking one. A bare-number reply to that question
("one thousand") is accepted only when it is one of the two values offered.

The other way round (``to_reading``): a frequency, auto brake setting or
squawk that is impossible as heard ("standby two one two two decimal eight",
212.28) but valid with one "two" read as "to" (122.8). The worker finds that
reading with the model; the plugin asks the pilot to confirm it.

Pure Python: it works on the transcript, before anything reaches the sim.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vimaan_nlu.commands import command
from vimaan_nlu.normalization import (
    _MAGNITUDE_WORDS,
    _TEEN_WORDS,
    _TENS_WORDS,
    _UNIT_WORDS,
    normalize_aviation_input,
)
from vimaan_nlu.validators import validate_slot

# The commands whose target is a number that can be misread this way.
TARGET_SLOTS = {
    "set_autopilot_heading": "degrees",
    "set_autopilot_altitude": "altitude",
    "set_flight_level": "flight_level",
    # Tier 1: "speed to eighty" (280?), "vertical speed to eight hundred"
    # (2,800?), "flaps to five" (25?).
    "set_autopilot_speed": "speed",
    "set_vertical_speed": "vs",
    "set_flaps": "flap_setting",
    # Tier 2: "course to one eight zero" heard as "course two one eight zero".
    "set_course": "degrees",
}
# Said as three digits ("two seven zero", "speed two one zero"), the "two" is
# taken as a digit.
THREE_DIGIT_SLOTS = {"degrees", "flight_level", "speed"}

_NUMBER_WORDS = set(_UNIT_WORDS) | set(_TEEN_WORDS) | set(_TENS_WORDS) | _MAGNITUDE_WORDS


def _as_number(words: list[str]) -> str | None:
    text = normalize_aviation_input(" ".join(words))
    return text if text.isdigit() else None


def two_or_to(intent: str, text: str) -> tuple[str, str] | None:
    """(value if "two", value if "to") when they differ and the "two" reading is
    valid, else None. The "to" reading may be out of range: "altitude to five
    hundred" is heard as "two five hundred", and setting 2,500 because 500 is
    below the minimum hides the mishearing (seen in flight). Asked, the pilot
    hears both and a reply of 500 is then refused as out of range."""
    slot = TARGET_SLOTS.get(intent)
    if slot is None:
        return None
    words = text.lower().split()
    for i, word in enumerate(words):
        if word != "two" or (i > 0 and words[i - 1] in _NUMBER_WORDS):
            continue  # not "two", or the middle of a number ("one two zero")
        end = i + 1
        while end < len(words) and words[end] in _NUMBER_WORDS:
            end += 1
        run = words[i + 1 : end]
        if not run or run[0] in _MAGNITUDE_WORDS:
            continue  # "two thousand", "two hundred seventy": nobody says "to thousand"
        if slot in THREE_DIGIT_SLOTS and len(run) == 2 and all(w in _UNIT_WORDS for w in run):
            continue
        as_two, as_to = _as_number(["two", *run]), _as_number(run)
        if as_two is None or as_to is None or int(as_two) == int(as_to):
            continue
        if validate_slot(slot, as_two)[1]:
            return as_two, as_to
    return None


# Words a pilot may add around a bare number reply ("one thousand feet").
_REPLY_FILLER = {"feet", "foot", "ft", "degrees", "degree", "please", "uh", "um", "er"}


def bare_number(text: str) -> str | None:
    """The number, when the utterance is nothing but one ("one thousand",
    "zero two zero"): no command word, so on its own it means nothing."""
    words = [w for w in text.lower().split() if w not in _REPLY_FILLER]
    if not words or any(w not in _NUMBER_WORDS for w in words):
        return None
    return _as_number(words)


def chosen(offered: tuple[str, str], text: str) -> str | None:
    """Which of the two offered values a bare reply names, or None.
    Compared as numbers: "zero two zero" names 20."""
    value = bare_number(text)
    if value is None:
        return None
    for option in offered:
        if int(option) == int(value):
            return option
    return None


# ---------------------------------------------------------------------------
# A number that is impossible as heard, but valid with one "two" read as "to"
# ---------------------------------------------------------------------------
# The owner's flight on 2026-11-16 (session 20261116_205022): "tune nav one to
# one one zero decimal five" was heard as "... nav one two one one zero ...",
# "set the auto brake to three" as "... auto brake two three" and "set com one
# standby to one two two decimal eight" as "... standby two one two two ...".
# Decoded again without the command-word grammar, the same clips came out as
# "to" every time: the grammar tips Vosk to "two" before a number. All three
# were refused as out of range or missing, so nothing wrong was set, but the
# pilot was given no way forward. two_or_to doesn't cover them: these aren't
# whole numbers, and its "two" after a number word ("nav one two ...") is
# taken as part of the number.
#
# The commands covered, and the slots one of which must hold a valid value.
TO_READING_SLOTS: dict[str, tuple[str, ...]] = {
    "set_com_frequency": ("frequency",),
    "set_com_standby": ("frequency",),
    "set_nav_frequency": ("nav_frequency",),
    "set_nav_standby": ("nav_frequency",),
    "set_autobrake": ("autobrake_level", "autobrake"),
    "set_transponder_code": ("squawk",),
}
# The rule name a "did you mean" question is held under (it is not a safety
# rule: nothing about the command is risky, only how it was heard).
TO_HEARD_AS_TWO = "to_heard_as_two"
# A radio number that came out wrong ("nav 121") makes the reading invalid too.
_PORT_SLOTS = ("com_port", "nav_port")
# Each "two" tried costs one pass of the model (about 10 ms).
MAX_TO_READINGS = 4


def has_valid_target(intent: str, slots: dict[str, Any]) -> bool:
    """Whether the command's number is there and in range, and any radio
    number it names is a real one."""
    names = TO_READING_SLOTS.get(intent, ())
    if not any(slots.get(n) and validate_slot(n, slots[n])[1] for n in names):
        return False
    return all(validate_slot(p, slots[p])[1] for p in _PORT_SLOTS if slots.get(p))


def to_reading(
    intent: str,
    slots: dict[str, Any],
    text: str,
    predict: Callable[[str], dict[str, Any]],
) -> dict[str, Any] | None:
    """The reading of ``text`` with one "two" taken as "to", when the command
    as heard has no valid number and exactly one such reading gives the same
    command with a valid one. ``predict`` runs the model on a transcript.

    Never acted on directly: the plugin asks the pilot to confirm it. None
    when the heard command is fine, when no "two" helps, when the readings
    disagree, or when a reading is a different command ("q now one to one one
    zero decimal five" was the altimeter as heard and NAV 1 as "to")."""
    cmd = command(intent)
    if cmd is None or intent not in TO_READING_SLOTS or has_valid_target(intent, slots):
        return None
    words = text.lower().split()
    spots = [i for i, word in enumerate(words) if word == "two"][:MAX_TO_READINGS]
    declared = cmd.slots
    found: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    for i in spots:
        variant = " ".join([*words[:i], "to", *words[i + 1 :]])
        p = predict(variant)
        if p["intent"] != intent or not has_valid_target(intent, p["slots"]):
            continue
        kept = {k: v for k, v in p["slots"].items() if k in declared and v not in (None, "")}
        key = tuple(sorted((k, str(v)) for k, v in kept.items()))
        found.setdefault(
            key,
            {
                "text": variant,
                "intent": intent,
                "slots": kept,
                "confidence": p["confidence"],
                "normalized": p.get("normalized_text"),
            },
        )
    return next(iter(found.values())) if len(found) == 1 else None
