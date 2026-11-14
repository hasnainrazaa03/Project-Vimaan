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
    # When set, the value must also be a multiple of it.
    step: float = 0

    def _num(self, value: Any) -> float | None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return int(v) if self.integer else v

    def _fmt(self, v: float) -> str:
        return str(int(v)) if self.integer else str(v)

    def _within(self, value: Any, v: float) -> bool:
        x = float(value) if self.exact else v
        if not self.lo <= x <= self.hi:
            return False
        return not self.step or abs(x / self.step - round(x / self.step)) < 1e-9

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
        steps = f" in steps of {self._fmt(self.step)}" if self.step else ""
        return None, False, f"out of range [{self._fmt(self.lo)}, {self._fmt(self.hi)}]{steps}"


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


@dataclass(frozen=True)
class AnyOf:
    """Valid when any of ``options`` accepts the value: an altimeter setting is
    29.92 inHg said with a decimal, 2992 said as four digits, or 1013 hPa."""

    options: tuple[Range | Enum, ...]

    def contains(self, value: Any) -> bool:
        return any(o.contains(value) for o in self.options)

    def validate(self, value: Any) -> tuple[str | None, bool, str]:
        for option in self.options:
            coerced, ok, reason = option.validate(value)
            if ok:
                return coerced, True, reason
        return None, False, "not a valid value"


@dataclass(frozen=True)
class Digits:
    """A fixed number of digits, each from ``allowed``, kept as a string so
    leading zeros survive: a squawk code is four octal digits ("0120",
    "7700"); 8 and 9 aren't on a transponder."""

    length: int
    allowed: str = "0123456789"

    def _clean(self, value: Any) -> str:
        s = str(value).strip().replace(" ", "")
        return s[:-2] if s.endswith(".0") else s  # 1200.0 from a float

    def contains(self, value: Any) -> bool:
        s = self._clean(value)
        return len(s) == self.length and all(c in self.allowed for c in s)

    def validate(self, value: Any) -> tuple[str | None, bool, str]:
        s = self._clean(value)
        if not s.isdigit():
            return None, False, "not digits"
        if len(s) != self.length:
            return None, False, f"{len(s)} digits, need {self.length}"
        if any(c not in self.allowed for c in s):
            return None, False, f"digits must be {self.allowed[0]}-{self.allowed[-1]}"
        return s, True, "valid"
