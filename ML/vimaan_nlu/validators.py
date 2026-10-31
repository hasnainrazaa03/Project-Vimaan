"""Declarative slot validation (Phase 6A).

The constraints themselves are declared in the command table
(``commands.py``); this module is the validation API. Two consumers:

- ``postprocessor`` uses the numeric ranges when selecting a value from the
  utterance, so the bounds live in one place instead of as literals scattered
  across the selection loops.
- the plugin / dashboard can call :func:`validate_slot` for explicit
  accept / wrap / reject feedback before acting on a value. Phase 6B's safety
  interlocks build on this.

Pure-Python (no torch), so it imports anywhere the package does.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vimaan_nlu import commands as _commands
from vimaan_nlu.slot_types import Enum, Range

__all__ = [
    "COMMAND_WORDS",
    "SLOT_VALIDATORS",
    "Enum",
    "Range",
    "has_command_word",
    "heading_too_short",
    "in_range",
    "slot_validator",
    "validate_slot",
]


# Slot constraints and command words live in the command table; this module
# keeps the validation API the postprocessor, plugin and dashboard use.
SLOT_VALIDATORS: Mapping[str, Range | Enum] = _commands.SLOT_VALIDATORS


def slot_validator(name: str) -> Range | Enum | None:
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
    """True if ``value`` satisfies the slot's constraint (no wrap/coercion)."""
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


def has_command_word(intent: str, text: str) -> bool:
    """True if ``text`` names what ``intent`` acts on (or the intent has no
    command-word list)."""
    words = COMMAND_WORDS.get(intent)
    if words is None:
        return True
    return not words.isdisjoint(text.lower().split())


def heading_too_short(value: Any) -> bool:
    """A heading given as one digit ("heading one") is almost always a
    clipped "one eighty" / "one two zero": pilots say headings as three
    digits ("zero zero five"). Ask again instead of turning to 001."""
    s = str(value).strip()
    return s.isdigit() and len(s) == 1
