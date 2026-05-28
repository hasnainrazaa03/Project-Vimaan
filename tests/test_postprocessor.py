"""Tests for vimaan_nlu.postprocessor — numeric range clamps + implicit state."""

import pytest
from vimaan_nlu.postprocessor import (
    ACTION_STATE_MAP,
    IMPLICIT_STATE_INTENTS,
    add_implicit_state,
    extract_digit_sequence_frequency,
    extract_numbers_from_text,
    postprocess_slots,
)


class TestExtractNumbersFromText:
    def test_simple_ints(self):
        assert extract_numbers_from_text("set heading 270") == ["270"]

    def test_decimal_value(self):
        assert extract_numbers_from_text("com 121.9") == ["121.9"]

    def test_multiple_numbers(self):
        out = extract_numbers_from_text("climb to 15000 ft on heading 270")
        assert out == ["15000", "270"]

    def test_no_numbers(self):
        assert extract_numbers_from_text("gear up") == []


class TestExtractDigitSequenceFrequency:
    def test_basic_frequency(self):
        out = extract_digit_sequence_frequency("tune com one two one point niner")
        # "one two one . niner" -> "121.9", within 118-137 range
        assert out == "121.9"

    def test_out_of_band_rejected(self):
        # "two zero zero point zero" = 200.0, outside 118-137
        out = extract_digit_sequence_frequency("com frequency two zero zero point zero")
        assert out is None

    def test_no_com_context_returns_none(self):
        out = extract_digit_sequence_frequency("set heading two seven zero")
        assert out is None


class TestAddImplicitState:
    def test_skips_non_toggle_intents(self):
        slots = {}
        out = add_implicit_state(slots, "climb to 15000", "set_autopilot_altitude")
        assert out == {}

    def test_existing_state_preserved(self):
        slots = {"state": "on"}
        out = add_implicit_state(slots, "engage autopilot", "toggle_autopilot_1")
        assert out == {"state": "on"}

    @pytest.mark.parametrize(
        "phrase, expected",
        [
            ("raise the landing gear", "up"),
            ("lower the gear", "down"),
            ("retract gear", "up"),
            ("extend gear", "down"),
        ],
    )
    def test_gear_action_inference(self, phrase, expected):
        slots = {}
        out = add_implicit_state(slots, phrase, "toggle_landing_gear")
        assert out["state"] == expected

    @pytest.mark.parametrize(
        "phrase, expected",
        [
            ("engage autopilot one", "on"),
            ("disengage autopilot one", "off"),
            ("turn on autopilot", "on"),
            ("turn off autopilot", "off"),
            ("activate autopilot", "on"),
        ],
    )
    def test_autopilot_action_inference(self, phrase, expected):
        slots = {}
        out = add_implicit_state(slots, phrase, "toggle_autopilot_1")
        assert out["state"] == expected

    def test_unknown_action_leaves_slots_alone(self):
        slots = {}
        out = add_implicit_state(slots, "pancakes", "toggle_landing_gear")
        assert "state" not in out


class TestPostprocessSlots:
    def test_strips_spaces_inside_digits(self):
        # "1 5 0 0 0" -> "15000"
        slots = {"altitude": "1 5 0 0 0"}
        out = postprocess_slots(slots, "climb to 15000")
        assert out["altitude"] == "15000"

    def test_altitude_clamp(self):
        # 270 is below valid altitude range; the loop should pick a value
        # >= 1000. The model can produce "270" via heading confusion.
        slots = {"altitude": "0"}
        out = postprocess_slots(slots, "climb to 15000 feet")
        assert out["altitude"] == "15000"

    def test_heading_in_range_kept(self):
        slots = {"degrees": "0"}
        out = postprocess_slots(slots, "heading 270")
        assert out["degrees"] == "270"

    def test_flight_level_picks_realistic_value(self):
        slots = {"flight_level": "0"}
        # last realistic FL value 10..430 selected (extractor scans reversed)
        out = postprocess_slots(slots, "climb to flight level 250")
        assert out["flight_level"] == "250"

    def test_com_freq_picks_in_band_decimal(self):
        # postprocess_slots only enriches slots that already have some value;
        # an empty string is treated as "no slot extracted" and left alone.
        # Simulate the model returning the raw token run as the slot value.
        slots = {"frequency": "one two one point niner"}
        out = postprocess_slots(
            slots,
            "tune com one two one point niner",
            intent="set_com_frequency",
        )
        assert out["frequency"] == "121.9"

    def test_implicit_state_dispatched_when_intent_provided(self):
        slots = {}
        out = postprocess_slots(slots, "raise the landing gear", intent="toggle_landing_gear")
        assert out.get("state") == "up"

    def test_does_not_invent_state_without_intent(self):
        slots = {}
        out = postprocess_slots(slots, "raise the landing gear")
        assert "state" not in out


class TestActionStateMapCoverage:
    def test_all_actions_resolve_to_valid_state(self):
        for action, state in ACTION_STATE_MAP.items():
            assert state in {"up", "down", "on", "off"}, action

    def test_implicit_state_intents_are_only_toggles(self):
        for intent in IMPLICIT_STATE_INTENTS:
            assert intent.startswith("toggle_"), intent
