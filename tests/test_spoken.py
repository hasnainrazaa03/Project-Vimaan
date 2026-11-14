"""Round trip: every way the generator says a value, the normalizer must read
back as that value's digits. Training on text the runtime reads differently
is how "heading one eighty" came to mean 81."""

import random
import re

import pytest
from data import spoken
from vimaan_nlu.normalization import normalize_aviation_input

DIGIT_WORDS = set(spoken.ONES) | {"niner"}


def reads_as(text: str, context: str, expected: str) -> bool:
    norm = normalize_aviation_input(context.format(text))
    return re.search(r"(?<![\d.])" + re.escape(expected) + r"(?![\d.])", norm) is not None


def test_renderers():
    assert spoken.heading(90) == ["zero nine zero", "zero niner zero", "ninety"]
    assert spoken.heading(7) == ["zero zero seven"]  # no one-digit form
    assert "two seventy" in spoken.heading(270) and "two hundred seventy" in spoken.heading(270)
    assert spoken.altitude(15000) == ["fifteen thousand", "one five thousand"]
    assert "twenty five hundred" in spoken.altitude(2500)
    assert "three fifty" in spoken.flight_level(350)
    assert "one twenty one point five" in spoken.frequency("121.5")
    assert spoken.pilot_group("305") == "three oh five"
    assert spoken.pilot_group("200") is None


@pytest.mark.parametrize("deg", range(360))
def test_every_heading(deg):
    for text in spoken.heading(deg):
        digit_by_digit = set(text.split()) <= DIGIT_WORDS
        expected = f"{deg:03d}" if digit_by_digit else str(deg)
        assert reads_as(text, "set heading {}", expected), text


def test_every_altitude():
    bad = [
        t
        for ft in range(1000, 45001, 100)
        for t in spoken.altitude(ft)
        if not reads_as(t, "set altitude {}", str(ft))
    ]
    assert bad == []


def test_every_flight_level():
    bad = [
        t
        for fl in range(100, 431, 10)
        for t in spoken.flight_level(fl)
        if not reads_as(t, "climb to flight level {}", str(fl))
    ]
    assert bad == []


def test_frequencies():
    rng = random.Random(1)
    bad = []
    for _ in range(600):
        mhz = f"{rng.randrange(118000, 137000, 5) / 1000:.3f}".rstrip("0")
        mhz = mhz + "0" if mhz.endswith(".") else mhz
        bad += [t for t in spoken.frequency(mhz) if not reads_as(t, "tune com one to {}", mhz)]
    assert bad == []


# ---------------------------------------------------------------------------
# Tier 1 values
# ---------------------------------------------------------------------------
def same_number(norm: str, context: str, expected: float) -> bool:
    """The slot's number in the normalized text equals ``expected``."""
    found = re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", norm)
    return any(abs(float(n) - expected) < 1e-9 for n in found)


def test_tier1_renderers():
    assert spoken.speed(180) == ["one eight zero", "one eighty", "one hundred eighty"]
    assert spoken.speed(85) == ["eighty five", "eight five"]
    assert spoken.mach("0.78") == ["point seven eight", "decimal seven eight"]
    assert "point eight" in spoken.mach("0.80")
    assert spoken.vertical_speed(1500) == ["one thousand five hundred", "fifteen hundred"]
    assert spoken.pressure("29.92") == [
        "two niner niner two",
        "two nine nine two",
        "twenty nine ninety two",
        "two nine point nine two",
    ]
    assert spoken.pressure("1013") == ["one zero one three"]
    assert spoken.flaps(15) == ["fifteen"]


@pytest.mark.parametrize("kt", range(60, 401, 5))
def test_every_speed(kt):
    for text in spoken.speed(kt):
        assert reads_as(text, "set speed {} knots", str(kt)), text


@pytest.mark.parametrize("hundredths", range(40, 96))
def test_every_mach(hundredths):
    value = f"0.{hundredths:02d}"
    for text in spoken.mach(value):
        norm = normalize_aviation_input(f"set mach {text}")
        assert same_number(norm, text, float(value)), (text, norm)


@pytest.mark.parametrize("fpm", range(100, 6001, 100))
def test_every_vertical_speed(fpm):
    for text in spoken.vertical_speed(fpm):
        assert reads_as(text, "descend at {} feet per minute", str(fpm)), text


@pytest.mark.parametrize("hundredths", range(2800, 3151))
def test_every_inch_setting(hundredths):
    value = f"{hundredths / 100:.2f}"
    for text in spoken.pressure(value):
        norm = normalize_aviation_input(f"altimeter {text}")
        # Four digits ("2992") or with the decimal ("29.92"): both mean 29.92.
        assert same_number(norm, text, float(value)) or same_number(norm, text, hundredths), (
            text,
            norm,
        )


@pytest.mark.parametrize("hpa", range(940, 1061))
def test_every_hectopascal_setting(hpa):
    for text in spoken.pressure(str(hpa)):
        assert reads_as(text, "q n h {}", str(hpa)), text


@pytest.mark.parametrize("setting", (1, 2, 3, 5, 10, 15, 20, 25, 30, 40))
def test_every_flap_setting(setting):
    for text in spoken.flaps(setting):
        assert reads_as(text, "flaps {}", str(setting)), text


def test_fraction_rule_leaves_frequencies_alone():
    assert normalize_aviation_input("tune one two one point five") == "tune 121.5"
    assert normalize_aviation_input("set mach point seven eight") == "set mach 0.78"


# ---------------------------------------------------------------------------
# Tier 2 values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("port", ["one", "two", ""])
def test_every_nav_frequency(port):
    # Every 50 kHz channel, also after a NAV port the recogniser may run in.
    bad = []
    for units in range(10800, 11796, 5):
        mhz = f"{units / 100:.2f}".rstrip("0")
        mhz = mhz + "0" if mhz.endswith(".") else mhz
        context = f"tune nav {port} to {{}}".replace("  ", " ")
        bad += [t for t in spoken.frequency(mhz) if not reads_as(t, context, mhz)]
    assert bad == []


def test_squawk_renderer():
    assert spoken.squawk("0120") == ["zero one two zero"]
    assert spoken.squawk("1200") == ["one two zero zero", "twelve hundred"]
    assert spoken.squawk("7000") == ["seven zero zero zero", "seven thousand"]
    assert spoken.squawk("4217") == ["four two one seven"]


def test_every_squawk():
    # All 4096 codes: leading zeros survive ("squawk 0120" is not 120).
    bad = [
        t
        for n in range(8**4)
        for code in [f"{int(oct(n)[2:]):04d}"]
        for t in spoken.squawk(code)
        if not reads_as(t, "squawk {}", code)
    ]
    assert bad == []
