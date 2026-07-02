"""Tests for vimaan_nlu.readback (Phase 6D) — phonetic digit read-back."""

from vimaan_nlu.readback import spell_digits


class TestSpellDigits:
    def test_heading(self):
        assert spell_digits("270") == "two seven zero"

    def test_zero_padded_heading(self):
        assert spell_digits("090") == "zero niner zero"

    def test_flight_level(self):
        assert spell_digits("350") == "three five zero"

    def test_frequency_with_decimal(self):
        assert spell_digits("121.5") == "one two one decimal five"

    def test_nine_uses_niner(self):
        assert spell_digits("9") == "niner"

    def test_integer_input(self):
        assert spell_digits(270) == "two seven zero"

    def test_drops_unknown_characters(self):
        assert spell_digits("FL350") == "three five zero"

    def test_empty(self):
        assert spell_digits("") == ""
