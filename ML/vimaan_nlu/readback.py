"""Spoken read-back helpers (Phase 6D).

Aviation confirmations read numbers digit-by-digit ("heading two seven zero",
"flight level three five zero", "one two one decimal five") rather than as
cardinals. This is clearer through TTS than letting the synth guess
("two hundred seventy"). Pure-Python and unit-tested.
"""

from __future__ import annotations

_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "niner",  # aviation convention
    ".": "decimal",
    "-": "minus",
}


def spell_digits(value) -> str:
    """Read a number digit-by-digit, aviation style.

    ``"270" -> "two seven zero"``; ``"121.5" -> "one two one decimal five"``.
    Non-digit characters are dropped. Returns ``""`` for empty input.
    """
    return " ".join(_DIGIT_WORDS[c] for c in str(value) if c in _DIGIT_WORDS)
