"""Declarative slot validation (Phase 6A).

The constraints themselves are declared in the command table
(``commands.py``); this module is the validation API. Two consumers:

- ``postprocessor`` uses the numeric ranges when selecting a value from the
  utterance, so the bounds live in one place instead of as literals scattered
  across the selection loops.
- the plugin / dashboard can call :func:`validate_slot` for explicit
  accept / reject feedback before acting on a value. Phase 6B's safety
  interlocks build on this.

Pure-Python (no torch), so it imports anywhere the package does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vimaan_nlu import commands as _commands
from vimaan_nlu.slot_types import AnyOf, Digits, Enum, Range

__all__ = [
    "COMMAND_WORDS",
    "SLOT_VALIDATORS",
    "Enum",
    "Range",
    "clipped_number",
    "has_command_word",
    "heading_too_short",
    "in_range",
    "slot_validator",
    "validate_slot",
]


# Slot constraints and command words live in the command table; this module
# keeps the validation API the postprocessor, plugin and dashboard use.
SLOT_VALIDATORS: Mapping[str, Range | Enum | AnyOf | Digits] = _commands.SLOT_VALIDATORS


def slot_validator(name: str) -> Range | Enum | AnyOf | Digits | None:
    """Return the validator for ``name``, or ``None`` if the slot is unconstrained."""
    return SLOT_VALIDATORS.get(name)


def validate_slot(name: str, value: Any) -> tuple[Any, bool, str]:
    """Validate ``value`` for slot ``name``.

    Returns ``(value, ok, reason)``. Unknown slots pass through unchanged
    (``ok=True``).
    """
    v = SLOT_VALIDATORS.get(name)
    if v is None:
        return value, True, "no validator"
    return v.validate(value)


def in_range(name: str, value: Any) -> bool:
    """True if ``value`` satisfies the slot's constraint."""
    v = SLOT_VALIDATORS.get(name)
    return bool(v and v.contains(value))


# ---------------------------------------------------------------------------
# Plausibility guards on the whole utterance (not just one slot value)
# ---------------------------------------------------------------------------
# An actionable intent is only acted on when the (normalized) utterance
# contains one of its command words. In-sim, misrecognised fragments were
# classified confidently and executed: "seventy" -> flight level 70,
# "city eight on" -> flight director on. A pilot always names the thing.
COMMAND_WORDS: dict[str, frozenset[str]] = _commands.command_words()
# Actionable intents with a choice (up/down, on/off, a mode) and no required
# number: their handlers ask ("Gear up or down?", "Which mode?") when it's
# missing, so a bare name can be sent to them.
CHOICE_INTENTS: frozenset[str] = frozenset(
    c.intent
    for c in _commands.COMMANDS
    if c.actionable
    and any(slot.kind == "choice" for slot in c.slots.values())
    and all(c.slots[n].kind == "choice" for n in c.required if n in c.slots)
)


def has_command_word(intent: str, text: str) -> bool:
    """True if ``text`` names what ``intent`` acts on (or the intent has no
    command-word list)."""
    words = COMMAND_WORDS.get(intent)
    if words is None:
        return True
    return not words.isdisjoint(text.lower().split())


# Words that can come before "thousand" / "hundred" in a spoken number.
_COUNT_WORDS = frozenset(
    "zero one two three four five six seven eight nine niner ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty "
    "seventy eighty ninety".split()
)
_MULTIPLIERS = frozenset({"thousand", "hundred"})


def clipped_number(text: str) -> bool:
    """True if "thousand" or "hundred" has no number before it. Pilots always
    say one ("eight thousand", "fifteen hundred"), so a bare one means the
    number's start went unheard: "altitude thousand" was "altitude eight
    thousand" (and would fly 1,000 ft), "thousand five hundred see minute"
    was "descend at one thousand five hundred feet per minute" (and would
    climb). No command phrasing in the training data has one."""
    words = text.lower().split()
    return any(
        w in _MULTIPLIERS
        and (i == 0 or not (words[i - 1] in _COUNT_WORDS or words[i - 1].isdigit()))
        for i, w in enumerate(words)
    )


def heading_too_short(value: Any) -> bool:
    """A heading given as one digit ("heading one") is almost always a
    clipped "one eighty" / "one two zero": pilots say headings as three
    digits ("zero zero five"). Ask again instead of turning to 001."""
    s = str(value).strip()
    return s.isdigit() and len(s) == 1


# Words that can come with a bare command noun: "the landing gear".
_BARE_EXTRA = frozenset({"the", "landing", "please", "uh", "um", "er"})


def bare_command(text: str) -> str | None:
    """The intent when ``text`` is nothing but its name ("landing gear",
    "flaps", "parking brake") and the intent needs a choice (up/down,
    on/off), so its handler asks which; None otherwise. v17 classed "landing
    gear" as no command and stayed silent, twice in one flight."""
    words = set(text.lower().split())
    if not words or words <= _BARE_EXTRA:
        return None
    # Exactly one command may own the words: "heading" names the heading
    # value and the heading mode, so it is not a bare name.
    found = [i for i, names in COMMAND_WORDS.items() if words <= names | _BARE_EXTRA]
    return found[0] if len(found) == 1 and found[0] in CHOICE_INTENTS else None
