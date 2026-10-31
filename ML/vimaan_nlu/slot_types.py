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
    ``int(float(num))`` checks in the postprocessor). ``exact`` bounds-checks
    the raw number instead of the truncated one, so 360.9 is above 360.
    """

    lo: float
    hi: float
    integer: bool = True
    exact: bool = False

    def _num(self, value: Any) -> float | None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return int(v) if self.integer else v

    def _fmt(self, v: float) -> str:
        return str(int(v)) if self.integer else str(v)

    def _within(self, value: Any, v: float) -> bool:
        return self.lo <= (float(value) if self.exact else v) <= self.hi

    def contains(self, value: Any) -> bool:
        v = self._num(value)
        return v is not None and self._within(value, v)

    def validate(self, value: Any) -> tuple[str | None, bool, str]:
        """Return ``(coerced_value | None, ok, reason)``."""
        v = self._num(value)
        if v is None:
            return None, False, "not a number"
        if self._within(value, v):
            return self._fmt(v), True, "in range"
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
