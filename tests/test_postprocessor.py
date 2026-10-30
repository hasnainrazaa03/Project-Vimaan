"""Tests for vimaan_nlu.postprocessor — numeric range clamps + implicit state."""

import pytest
from vimaan_nlu.postprocessor import (
    CHOICE_SYNONYMS,
    add_implicit_state,
    canonicalize_choices,
    extract_digit_sequence_frequency,
    extract_numbers_from_text,
    postprocess_slots,
    resolve_instance,
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
        out = add_implicit_state(slots, "engage autopilot", "toggle_autopilot")
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
        out = add_implicit_state(slots, phrase, "toggle_autopilot")
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


class TestChoiceSynonymCoverage:
    def test_every_verb_maps_to_a_valid_value(self):
        from vimaan_nlu.validators import SLOT_VALIDATORS

        for intent, slots in CHOICE_SYNONYMS.items():
            for slot, values in slots.items():
                for value in values:
                    assert SLOT_VALIDATORS[slot].contains(value), (intent, slot, value)

    def test_no_verb_means_two_things_for_one_intent(self):
        for intent, slots in CHOICE_SYNONYMS.items():
            for slot, values in slots.items():
                verbs = [v for syns in values.values() for v in syns]
                assert len(verbs) == len(set(verbs)), (intent, slot)

    def test_tagged_verb_becomes_canonical(self):
        # v12 tags the word spoken; the plugin needs the value.
        assert canonicalize_choices(
            {"state": "raise"}, "raise the gear", "toggle_landing_gear"
        ) == {"state": "up"}
        assert canonicalize_choices({"position": "retract"}, "retract flaps", "set_flaps") == {
            "position": "up"
        }
        assert canonicalize_choices(
            {"state": "shut down"}, "shut down engine 2", "toggle_engine"
        ) == {"state": "off"}


def instance(text, intent="toggle_engine", slots=None):
    return resolve_instance(dict(slots or {}), text, intent)["instance"]


class TestResolveInstance:
    def test_named_number(self):
        assert instance("engine 2 off") == "2"
        assert instance("engage autopilot 1", "toggle_autopilot") == "1"
        assert instance("flight director 2 on", "toggle_flight_director") == "2"

    def test_tagged_value_wins(self):
        assert instance("start engine 2", slots={"instance": "2"}) == "2"
        assert instance("autopilot two on", "toggle_autopilot", {"instance": "two"}) == "2"

    def test_defaults_to_instance_1_when_none_named(self):
        assert instance("shut down the engine") == "1"
        assert instance("engage the autopilot", "toggle_autopilot") == "1"
        assert instance("flight director on", "toggle_flight_director") == "1"

    def test_ordinal_and_side_words_name_the_instance(self):
        assert instance("shut down right engine") == "2"
        assert instance("engage second autopilot", "toggle_autopilot") == "2"
        assert instance("start the left engine") == "1"

    def test_both(self):
        assert instance("shut down both engines") == "both"

    def test_disconnect_means_off_for_autopilot(self):
        slots = postprocess_slots({}, "disconnect autopilot", "toggle_autopilot")
        assert slots == {"state": "off", "instance": "1"}

    def test_not_applied_to_other_intents(self):
        assert resolve_instance({}, "gear 2 up", "toggle_landing_gear") == {}
        assert resolve_instance({}, "heading 2 7 0", "set_autopilot_heading") == {}

    def test_ignores_other_numbers(self):
        assert instance("engine 1 at 350") == "1"

    def test_handles_empty(self):
        assert resolve_instance({}, "engine 2", "") == {}
        assert instance("") == "1"
