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
    assert spoken.heading(90) == ["zero nine zero", "zero niner zero"]
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
