"""Spot a target value that depends on hearing "to" or "two".

The two words sound the same. In-sim, Vosk scored both readings of "set altitude
to one thousand" identically and printed "two", which is also a correct way to
say 21,000, so the wrong target was set with no sign of doubt. Nothing in the
transcript can settle it, so when both readings are valid and different the
plugin asks instead of picking one.

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
}
# Said as three digits ("two seven zero"), the "two" is taken as a digit.
THREE_DIGIT_SLOTS = {"degrees", "flight_level"}

_NUMBER_WORDS = set(_UNIT_WORDS) | set(_TEEN_WORDS) | set(_TENS_WORDS) | _MAGNITUDE_WORDS


def _as_number(words: list[str]) -> str | None:
    text = normalize_aviation_input(" ".join(words))
    return text if text.isdigit() else None


def two_or_to(intent: str, text: str) -> tuple[str, str] | None:
    """(value if "two", value if "to") when both are valid and different, else None."""
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
        if validate_slot(slot, as_two)[1] and validate_slot(slot, as_to)[1]:
            return as_two, as_to
    return None
