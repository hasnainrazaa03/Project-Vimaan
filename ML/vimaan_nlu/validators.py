"""Declarative slot validation (Phase 6A).

A single source of truth for slot value constraints. Two consumers:

- ``postprocessor`` uses the numeric ranges when selecting a value from the
  utterance, so the bounds live in one place instead of as literals scattered
  across the selection loops.
- the plugin / dashboard can call :func:`validate_slot` for explicit
  accept / wrap / reject feedback before acting on a value. Phase 6B's safety
  interlocks build on this.

Pure-Python (no torch), so it imports anywhere the package does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Range:
    """An inclusive numeric range.

    ``integer`` compares the value as an int (matching the historical
    ``int(float(num))`` checks in the postprocessor). ``wrap`` wraps an
    out-of-range value modulo the upper bound (headings: 450 -> 90) instead of
    rejecting it.
    """

    lo: float
    hi: float
    integer: bool = True
    wrap: bool = False

    def _num(self, value: Any) -> float | None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return int(v) if self.integer else v

    def _fmt(self, v: float) -> str:
        return str(int(v)) if self.integer else str(v)

    def contains(self, value: Any) -> bool:
        v = self._num(value)
        return v is not None and self.lo <= v <= self.hi

    def validate(self, value: Any) -> tuple[str | None, bool, str]:
        """Return ``(coerced_value | None, ok, reason)``."""
        v = self._num(value)
        if v is None:
            return None, False, "not a number"
        if self.lo <= v <= self.hi:
            return self._fmt(v), True, "in range"
        if self.wrap:
            wrapped = v % self.hi
            return self._fmt(wrapped), True, f"wrapped into [0, {self._fmt(self.hi)}]"
        return None, False, f"out of range [{self._fmt(self.lo)}, {self._fmt(self.hi)}]"


@dataclass(frozen=True)
class Enum:
    """A closed set of allowed string values (case-insensitive)."""

    values: tuple[str, ...]

    def contains(self, value: Any) -> bool:
        return str(value).strip().lower() in self.values

    def validate(self, value: Any) -> tuple[str | None, bool, str]:
        s = str(value).strip().lower()
        if s in self.values:
            return s, True, "valid"
        return None, False, f"not one of {self.values}"


# Single source of truth for slot constraints. The numeric bounds match the
# README and were previously hard-coded as literals in postprocessor.py.
SLOT_VALIDATORS: dict[str, Range | Enum] = {
    "altitude": Range(1000, 50000),
    "degrees": Range(0, 360, wrap=True),
    "flight_level": Range(10, 430),
    "frequency": Range(118.0, 137.0, integer=False),
    "com_port": Range(1, 4),
    "state": Enum(("on", "off", "up", "down")),
}


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
COMMAND_WORDS: dict[str, frozenset[str]] = {
    "set_autopilot_heading": frozenset({"heading", "turn", "hdg", "degrees", "degree"}),
    "set_autopilot_altitude": frozenset(
        {"altitude", "climb", "descend", "feet", "fly", "maintain", "thousand", "alt"}
    ),
    "set_flight_level": frozenset({"level", "fl"}),
    "set_com_frequency": frozenset({"com", "comm", "radio", "frequency", "tune"}),
    "toggle_landing_gear": frozenset({"gear"}),
    "toggle_flaps": frozenset({"flaps", "flap"}),
    "toggle_autopilot_1": frozenset({"autopilot", "pilot", "ap"}),
    "toggle_autopilot_2": frozenset({"autopilot", "pilot", "ap"}),
    "toggle_flight_director_1": frozenset({"director", "fd"}),
    "toggle_flight_director_2": frozenset({"director", "fd"}),
    "toggle_parking_brake": frozenset({"parking", "brake", "brakes"}),
    "toggle_engine_1": frozenset({"engine", "engines"}),
    "toggle_engine_2": frozenset({"engine", "engines"}),
}


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
