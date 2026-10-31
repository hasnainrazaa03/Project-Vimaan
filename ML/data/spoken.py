"""Render slot values the ways pilots say them (training data generator v2).

Every renderer returns a list of spoken variants for one value, e.g. heading
270 -> ["two seven zero", "two seventy", "two hundred seventy"]. The
normalizer must turn every variant back into the value's digits; that
round trip is property-tested (tests/test_spoken.py), so the model is never
trained on text whose numbers the runtime would read differently (the "one
eighty" -> 81 bug class).
"""

from __future__ import annotations

ONES = "zero one two three four five six seven eight nine".split()
TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def digit_word(d: str, niner: bool = False) -> str:
    return "niner" if (niner and d == "9") else ONES[int(d)]


def digits(value: str, niner: bool = False) -> str:
    """Digit by digit: "270" -> "two seven zero" (decimal point -> "decimal")."""
    return " ".join("decimal" if c == "." else digit_word(c, niner) for c in value)


def two_digit(n: int) -> str:
    """0-99 as a cardinal: 7 -> "seven", 15 -> "fifteen", 75 -> "seventy five"."""
    if n < 10:
        return ONES[n]
    if n < 20:
        return TEENS[n - 10]
    tens, unit = divmod(n, 10)
    return TENS[tens] if unit == 0 else f"{TENS[tens]} {ONES[unit]}"


def cardinal(n: int, use_and: bool = False) -> str:
    """Cardinal English for 0-99,999: 7500 -> "seven thousand five hundred"."""
    if n == 0:
        return "zero"
    parts = []
    thousands, rest = divmod(n, 1000)
    if thousands:
        parts.append(f"{two_digit(thousands)} thousand")
    hundreds, rest = divmod(rest, 100)
    if hundreds:
        parts.append(f"{ONES[hundreds]} hundred")
    if rest:
        if parts and use_and:
            parts.append("and")
        parts.append(two_digit(rest))
    return " ".join(parts)


def pilot_group(value: str) -> str | None:
    """A three-digit number as pilots group it: "270" -> "two seventy",
    "121" -> "one twenty one", "305" -> "three oh five". None when the
    grouping would be ambiguous or isn't how it's said ("200", "100", "007")."""
    if len(value) != 3 or not value.isdigit() or value[0] == "0":
        return None
    head, tail = value[0], int(value[1:])
    if tail == 0:
        return None
    if tail < 10:
        return f"{ONES[int(head)]} oh {ONES[tail]}"
    return f"{ONES[int(head)]} {two_digit(tail)}"


# ---------------------------------------------------------------------------
# Per-slot renderers: value -> spoken variants (most common first)
# ---------------------------------------------------------------------------
def heading(deg: int) -> list[str]:
    """Headings are three digits: 90 -> "zero niner zero"; also groups."""
    v = f"{deg:03d}"
    out = [digits(v), digits(v, niner=True)]
    if (g := pilot_group(v)) is not None:
        out.append(g)
    if deg >= 100 and deg % 10 == 0:
        out.append(cardinal(deg))  # "two hundred seventy"
    if 10 <= deg < 100:
        out.append(two_digit(deg))  # "change heading to eighty"
    return list(dict.fromkeys(out))


def altitude(ft: int) -> list[str]:
    """7500 -> "seven thousand five hundred"; 15000 -> also "one five thousand";
    2500 -> also "twenty five hundred"."""
    out = [cardinal(ft)]
    thousands, rest = divmod(ft, 1000)
    if thousands >= 10 and rest == 0:
        out.append(f"{digits(str(thousands))} thousand")  # ATC group form
    if 1000 < ft < 10000 and ft % 100 == 0 and rest:
        out.append(f"{two_digit(ft // 100)} hundred")  # "twenty five hundred"
    return list(dict.fromkeys(out))


def flight_level(fl: int) -> list[str]:
    """350 -> "three five zero", "three fifty"; 100 -> "one zero zero"."""
    v = f"{fl:03d}"
    out = [digits(v), digits(v, niner=True)]
    if (g := pilot_group(v)) is not None:
        out.append(g)
    return list(dict.fromkeys(out))


def frequency(mhz: str) -> list[str]:
    """ "121.5" -> "one two one decimal five", "... point five",
    "one twenty one decimal five"."""
    whole, frac = mhz.split(".")
    frac_words = digits(frac)
    out = [
        f"{digits(whole)} decimal {frac_words}",
        f"{digits(whole)} point {frac_words}",
        f"{digits(whole, niner=True)} decimal {digits(frac, niner=True)}",
    ]
    if (g := pilot_group(whole)) is not None:
        out.append(f"{g} decimal {frac_words}")
        out.append(f"{g} point {frac_words}")
    return list(dict.fromkeys(out))


def instance(n: str) -> list[str]:
    """Which engine / autopilot: "1" -> "one", also "left"/"first"."""
    return {
        "1": ["one", "number one", "left", "first"],
        "2": ["two", "number two", "right", "second"],
        "both": ["both"],
    }[n]
