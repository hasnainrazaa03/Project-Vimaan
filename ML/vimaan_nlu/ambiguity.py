"""Spot a target value that depends on hearing "to" or "two".

The two words sound the same. In-sim, Vosk scored both readings of "set altitude
to one thousand" identically and printed "two", which is also a correct way to
say 21,000, so the wrong target was set with no sign of doubt. Nothing in the
transcript can settle it, so when both readings are valid and different the
plugin asks instead of picking one. A bare-number reply to that question
("one thousand") is accepted only when it is one of the two values offered.

Pure Python: it works on the transcript, before anything reaches the sim.
"""

from __future__ import annotations

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
