"""Slot value types: numeric ranges and closed sets of words.

Shared by the command table (``commands.py``) and the validators. Pure Python.
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
