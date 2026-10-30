"""Tests for vimaan_nlu.normalization.

Covers the public API:
- normalize_aviation_input(text) — text-level transform used at inference time.
- normalize_slot_value(value)    — slot-value normalizer used on dataset items.
- normalize_dataset(data)        — bulk version applied during training prep.
"""

import pytest
from vimaan_nlu.normalization import (
    PHONETIC_MAP,
    normalize_aviation_input,
    normalize_dataset,
    normalize_slot_value,
)


class TestPhoneticMap:
    def test_all_digits_present(self):
        for digit in "0123456789":
            assert digit in PHONETIC_MAP.values()

    def test_decimal_aliases(self):
        assert PHONETIC_MAP["point"] == "."
        assert PHONETIC_MAP["decimal"] == "."

    def test_zero_aliases(self):
        assert PHONETIC_MAP["zero"] == "0"
        assert PHONETIC_MAP["oh"] == "0"

    def test_nine_aliases(self):
        assert PHONETIC_MAP["nine"] == "9"
        assert PHONETIC_MAP["niner"] == "9"


class TestNormalizeAviationInput:
    def test_three_digit_heading(self):
        # "two seven zero" -> "270"
        out = normalize_aviation_input("set heading two seven zero")
        assert "270" in out

    def test_phonetic_with_oh(self):
        # "zero niner zero" -> "090"
        out = normalize_aviation_input("heading zero niner zero")
        assert "090" in out

    def test_com_frequency_with_decimal(self):
        # The decimal pass runs before the phonetic-run pass, so a full
        # "phonetic ... point/decimal ... phonetic" sequence is joined into a
        # single decimal number at the text level.
        out = normalize_aviation_input("tune com one two one decimal niner")
        assert "121.9" in out

    def test_com_frequency_decimal_five(self):
        # Regression: "one two one decimal five" must become "121.5", not
        # "121 decimal 5" (the phonetic-run pass used to eat the integer part
        # before the decimal pass could join it).
        assert "121.5" in normalize_aviation_input("tune comm one to one two one decimal five")
        assert "118.1" in normalize_aviation_input("contact tower one one eight point one")

    def test_compound_thousand(self):
        # "fifteen thousand" -> "15000"
        out = normalize_aviation_input("climb to fifteen thousand feet")
        assert "15000" in out

    def test_atc_group_form_altitude(self):
        # ATC "one five thousand" = 15000 (not the old "15 1000" -> altitude 1000)
        assert "15000" in normalize_aviation_input("climb to one five thousand feet").split()
        assert "25000" in normalize_aviation_input("descend to two five thousand").split()

    def test_thousand_hundred_compound(self):
        # "seven thousand five hundred" = 7500 (old code mis-split it to 7005)
        out = normalize_aviation_input("descend to seven thousand five hundred feet")
        assert "7500" in out.split()

    def test_digit_run_without_magnitude_stays_a_digit_run(self):
        # No magnitude word -> phonetic concat, not a cardinal/group form.
        assert "270" in normalize_aviation_input("heading two seven zero").split()

    def test_compound_hundred(self):
        # "two hundred" -> "200"
        out = normalize_aviation_input("flaps two hundred")
        assert "200" in out

    def test_no_numbers_pass_through(self):
        out = normalize_aviation_input("gear up")
        assert "gear" in out
        assert "up" in out

    def test_lowercases_input(self):
        out = normalize_aviation_input("SET HEADING")
        assert "set heading" in out

    def test_empty_string_safe(self):
        assert normalize_aviation_input("") == ""

    def test_already_digits_passes_through(self):
        out = normalize_aviation_input("set heading 270")
        assert "270" in out

    @pytest.mark.parametrize(
        "phrase, expected",
        [
            ("one two three", "123"),
            ("seven eight zero", "780"),
            ("niner zero zero", "900"),
        ],
    )
    def test_digit_sequences(self, phrase, expected):
        out = normalize_aviation_input(f"frequency {phrase}")
        assert expected in out


class TestNormalizeSlotValue:
    def test_passes_through_int_string(self):
        assert normalize_slot_value("270") == "270"

    def test_passes_through_decimal(self):
        assert normalize_slot_value("121.9") == "121.9"

    def test_word_to_int(self):
        assert normalize_slot_value("fifteen thousand") == "15000"

    def test_compound_word_number(self):
        # word2number handles "two hundred seventy" → 270.
        # The phonetic-sequence variant ("two seven zero") goes through
        # normalize_aviation_input at the text layer, not here.
        assert normalize_slot_value("two hundred seventy") == "270"

    def test_invalid_input_returns_original(self):
        # value_str is lowercased+stripped; non-numeric returns original (lowercased).
        out = normalize_slot_value("abc")
        assert out == "abc"

    def test_handles_non_string_input(self):
        assert normalize_slot_value(270) == "270"


class TestNormalizeDataset:
    def test_normalizes_slots_in_place(self):
        # normalize_slot_value relies on word2number; use word-form inputs it
        # can parse. Phonetic-digit sequences ("two seven zero") are handled
        # at the text layer by normalize_aviation_input, not here.
        data = [
            {
                "text": "climb to fifteen thousand",
                "intent": "set_autopilot_altitude",
                "slots": {"altitude": "fifteen thousand"},
            },
            {
                "text": "heading two hundred seventy",
                "intent": "set_autopilot_heading",
                "slots": {"degrees": "two hundred seventy"},
            },
        ]
        out = normalize_dataset(data)
        assert out[0]["slots"]["altitude"] == "15000"
        assert out[1]["slots"]["degrees"] == "270"

    def test_items_without_slots_pass_through(self):
        data = [{"text": "gear up", "intent": "toggle_landing_gear"}]
        out = normalize_dataset(data)
        assert out == data

    def test_empty_dataset(self):
        assert normalize_dataset([]) == []


class TestRecognizerFormatting:
    """Real STT (Google/Whisper) output: capitals, punctuation, thousands
    separators, abbreviations, digit-by-digit dictation."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Set altitude to 7,500.", "set altitude to 7500"),
            ("Climb to 10,000 ft", "climb to 10000 feet"),
            ("Set altitude 35,000", "set altitude 35000"),
            ("Climb to FL 350.", "climb to flight level 350"),
            ("FL350", "flight level 350"),
            ("Climb to 15000ft.", "climb to 15000 feet"),
            ("Heading 2 7 0", "heading 270"),
            ("Set heading 270.", "set heading 270"),
            ("Gear up!", "gear up"),
            ("Tune COM1 to 118.75.", "tune com 1 to 118.75"),
            ("Set com 2 to 121.5.", "set com 2 to 121.5"),
        ],
    )
    def test_asr_formatting(self, text, expected):
        assert normalize_aviation_input(text) == expected

    def test_decimal_point_survives_punctuation_cleanup(self):
        assert normalize_aviation_input("contact 118.75, thanks") == "contact 118.75 thanks"

    def test_instance_digit_not_merged_into_value(self):
        assert normalize_aviation_input("com 1 1 2 1 decimal 5") == "com 1 121.5"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("tune com one one one eight decimal seven five", "tune com 1 118.75"),
            ("com two one three two decimal four two five", "com 2 132.425"),
            # 3 integer digits: the frequency itself, not a port.
            ("com one two one decimal five", "com 121.5"),
        ],
    )
    def test_absorbed_com_port_split_out(self, text, expected):
        assert normalize_aviation_input(text) == expected


class TestGoogleMishearings:
    """Seen in the first in-sim session (Google Web Speech)."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("tune on 1 to 1:18 75", "tune on 1 to 118.75"),  # clock-time format
            ("Tune COM 1 to 1:18 75.", "tune com 1 to 118.75"),
            ("tune cam 1 to 118.75", "tune com 1 to 118.75"),
            ("come one one two one decimal five", "com 1 121.5"),
            ("set altitude 1:30", "set altitude 130"),
            ("Dune cam 12118.75", "dune com 1 118.75"),  # "one TO one one eight"
            ("Secom frequency 121.70", "set com frequency 121.70"),
            ("slaps up", "flaps up"),
        ],
    )
    def test_mishearings(self, text, expected):
        assert normalize_aviation_input(text) == expected

    def test_ordinary_words_untouched(self):
        assert normalize_aviation_input("come here") == "come here"
