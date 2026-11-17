"""vimaan_nlu.ambiguity: a target that depends on hearing "to" or "two"."""

import pytest
from vimaan_nlu.ambiguity import (
    MAX_TO_READINGS,
    bare_number,
    chosen,
    has_valid_target,
    to_reading,
    two_or_to,
)

HEADING, ALTITUDE, LEVEL = "set_autopilot_heading", "set_autopilot_altitude", "set_flight_level"


@pytest.mark.parametrize(
    ("heard", "intent", "as_two", "as_to"),
    [
        # The five wrong commands from the v17 test flight (session 20261030_215104).
        ("set altitude two one thousand feet", ALTITUDE, "21000", "1000"),  # c0028
        ("set heading two twenty degrees", HEADING, "220", "20"),  # c0030
        ("set heading two ten degrees", HEADING, "210", "10"),  # c0031, c0042
        ("set altitude two five thousand feet", ALTITUDE, "25000", "5000"),  # c0043
        # Valid commands that now ask, because the audio can't settle them.
        ("heading two seventy", HEADING, "270", "70"),
        ("flight level two forty", LEVEL, "240", "40"),
        ("set altitude two one thousand", ALTITUDE, "21000", "1000"),
        ("descend two three thousand feet", ALTITUDE, "23000", "3000"),
    ],
)
def test_two_valid_readings_are_ambiguous(heard, intent, as_two, as_to):
    assert two_or_to(intent, heard) == (as_two, as_to)


@pytest.mark.parametrize(
    ("heard", "intent"),
    [
        # Digit by digit: the "two" is a digit.
        ("heading two seven zero", HEADING),
        ("set heading two two zero", HEADING),
        ("fly heading two niner zero", HEADING),
        ("flight level two four zero", LEVEL),
        # Ways to give the value with no "two" at the start of it.
        ("altitude one thousand feet", ALTITUDE),
        ("set altitude twenty one thousand feet", ALTITUDE),
        ("set heading ten degrees", HEADING),
        ("heading zero one zero", HEADING),
        ("heading one eighty", HEADING),
        ("flight level three fifty", LEVEL),
        # A real "to" was heard.
        ("set altitude to one thousand feet", ALTITUDE),
        ("set heading to ten degrees", HEADING),
        # Only one reading works.
        ("set altitude two thousand", ALTITUDE),
        ("heading two hundred seventy", HEADING),
        ("heading two zero nine zero", HEADING),
        # "two" isn't the start of the value.
        ("heading one two zero", HEADING),
        ("climb two flight level three fifty", LEVEL),
    ],
)
def test_clear_values_are_not_ambiguous(heard, intent):
    assert two_or_to(intent, heard) is None


@pytest.mark.parametrize(
    ("intent", "heard"),
    [
        ("set_com_frequency", "tune com two to one two one decimal five"),
        ("set_com_frequency", "doing com one two one one eight decimal seven five"),
        ("toggle_engine", "engine two off"),
        ("toggle_autopilot", "autopilot two on"),
        ("toggle_landing_gear", "gear up"),
    ],
)
def test_other_commands_never_ask(intent, heard):
    assert two_or_to(intent, heard) is None


def test_low_altitude_reading_is_offered_even_below_the_minimum():
    # 2nd v17 flight c0013: "altitude to five hundred" heard as "two five
    # hundred" set 2,500 because 500 failed the 1,000 ft minimum. Now asked;
    # a reply of 500 is then refused as out of range, out loud.
    assert two_or_to(ALTITUDE, "altitude two five hundred") == ("2500", "500")
    assert two_or_to(LEVEL, "flight level two five") == ("25", "5")


def test_impossible_two_reading_is_not_offered():
    # 250,000 ft is no reading of anything: only "to fifty thousand" works.
    assert two_or_to(ALTITUDE, "altitude two fifty thousand") is None


@pytest.mark.parametrize(
    ("heard", "value"),
    [
        ("one thousand", "1000"),  # 2nd v17 flight c0004
        ("zero", "0"),  # c0010
        ("zero two zero", "020"),
        ("twenty one thousand feet", "21000"),
        ("two two zero degrees", "220"),
        ("uh five hundred", "500"),
    ],
)
def test_bare_number(heard, value):
    assert bare_number(heard) == value


@pytest.mark.parametrize(
    "heard",
    ["", "altitude one thousand", "one two one decimal five", "heading", "gear down", "70"],
)
def test_not_a_bare_number(heard):
    assert bare_number(heard) is None


@pytest.mark.parametrize(
    ("offered", "heard", "pick"),
    [
        (("1000", "21000"), "one thousand", "1000"),
        (("1000", "21000"), "twenty one thousand feet", "21000"),
        (("20", "220"), "zero two zero", "20"),
        (("20", "220"), "two two zero", "220"),
        (("20", "220"), "zero", None),  # c0010: neither value, never guessed
        (("1000", "21000"), "two thousand", None),
        (("1000", "21000"), "altitude one thousand", None),  # not bare: normal path
    ],
)
def test_chosen_only_names_an_offered_value(offered, heard, pick):
    assert chosen(offered, heard) == pick


SPEED, VS, FLAPS = "set_autopilot_speed", "set_vertical_speed", "set_flaps"


@pytest.mark.parametrize(
    ("heard", "intent", "as_two", "as_to"),
    [
        ("speed two eighty", SPEED, "280", "80"),  # the plugin drops 80 below Vso
        ("vertical speed two eight hundred feet per minute", VS, "2800", "800"),
        ("flaps two five", FLAPS, "25", "5"),
    ],
)
def test_tier1_two_readings(heard, intent, as_two, as_to):
    assert two_or_to(intent, heard) == (as_two, as_to)


@pytest.mark.parametrize(
    ("heard", "intent"),
    [
        ("speed two one zero", SPEED),  # three digits: the two is a digit
        ("set mach point seven eight", SPEED),
        ("vertical speed two thousand", VS),
        ("flaps two", FLAPS),
        ("flaps twenty five", FLAPS),
        ("flaps two fifteen", FLAPS),  # 215 is no flap setting
    ],
)
def test_tier1_clear_values(heard, intent):
    assert two_or_to(intent, heard) is None


# ---------------------------------------------------------------------------
# to_reading: impossible as heard, valid with one "two" read as "to"
# ---------------------------------------------------------------------------
def model(table):
    """A stand-in for the model: transcript -> (intent, slots); anything else
    is no command. Records what it was asked."""
    asked = []

    def predict(text):
        asked.append(text)
        intent, slots = table.get(text, ("None", {}))
        return {"intent": intent, "slots": dict(slots), "confidence": 0.99, "normalized_text": text}

    predict.asked = asked
    return predict


NAV, COM_SBY, BRAKE = "set_nav_frequency", "set_com_standby", "set_autobrake"


def test_the_three_refusals_from_the_owners_flight():
    # Session 20261116_205022, c0042, c0062, c0065: heard, as the model read it,
    # and the reading with the "two" that was "to".
    cases = [
        (
            "doing nav one two one one zero decimal five",
            NAV,
            {"nav_port": "121"},
            "doing nav one to one one zero decimal five",
            {"nav_port": "1", "nav_frequency": "110.5"},
        ),
        (
            "r lights status strobe two set the auto brake two three",
            BRAKE,
            {"autobrake_level": "23"},
            "r lights status strobe two set the auto brake to three",
            {"autobrake_level": "3"},
        ),
        (
            "set com one standby two one two two decimal eight",
            COM_SBY,
            {"com_port": "1", "frequency": "2122.8"},
            "set com one standby to one two two decimal eight",
            {"com_port": "1", "frequency": "122.8"},
        ),
    ]
    for heard, intent, slots, variant, want in cases:
        predict = model({variant: (intent, want)})
        got = to_reading(intent, slots, heard, predict)
        assert got is not None and got["text"] == variant
        assert got["intent"] == intent and got["slots"] == want


def test_a_valid_command_is_left_alone_without_asking_the_model():
    predict = model({})
    slots = {"com_port": "2", "frequency": "132.45"}
    heard = "com two standby one three two decimal four five"
    assert to_reading(COM_SBY, slots, heard, predict) is None
    assert to_reading(BRAKE, {"autobrake": "off"}, "auto brake off", predict) is None
    assert predict.asked == []


def test_commands_it_does_not_cover():
    predict = model({})
    assert (
        to_reading(
            "set_autopilot_heading", {"degrees": "2360"}, "heading two three six zero", predict
        )
        is None
    )
    assert to_reading("set_altimeter", {"pressure": "12110"}, "q now one two", predict) is None
    assert predict.asked == []


def test_a_reading_that_is_another_command_is_not_offered():
    heard = "nav one two one one zero decimal five"
    predict = model(
        {"nav one to one one zero decimal five": ("set_altimeter", {"pressure": "110"})}
    )
    assert to_reading(NAV, {"nav_port": "121"}, heard, predict) is None


def test_readings_that_disagree_are_not_offered():
    heard = "squawk two two one zero zero"  # 22100: neither "two" can be settled
    predict = model(
        {
            "squawk to two one zero zero": ("set_transponder_code", {"squawk": "2100"}),
            "squawk two to one zero zero": ("set_transponder_code", {"squawk": "2100"}),
        }
    )
    # The same value both ways is one reading.
    got = to_reading("set_transponder_code", {"squawk": "22100"}, heard, predict)
    assert got is not None and got["slots"] == {"squawk": "2100"}
    predict = model(
        {
            "squawk to two one zero zero": ("set_transponder_code", {"squawk": "2100"}),
            "squawk two to one zero zero": ("set_transponder_code", {"squawk": "1200"}),
        }
    )
    assert to_reading("set_transponder_code", {"squawk": "22100"}, heard, predict) is None


def test_a_reading_still_out_of_range_or_on_a_missing_radio_is_not_offered():
    heard = "com one standby two one two two decimal eight"
    for slots in (
        {"com_port": "1", "frequency": "212.28"},
        {"com_port": "5", "frequency": "122.8"},
    ):
        predict = model({"com one standby to one two two decimal eight": (COM_SBY, slots)})
        assert to_reading(COM_SBY, {"com_port": "1", "frequency": "2122.8"}, heard, predict) is None


def test_stray_slots_are_dropped_and_the_number_of_tries_is_capped():
    heard = "com one standby two one two two decimal eight"
    variant = "com one standby to one two two decimal eight"
    predict = model({variant: (COM_SBY, {"com_port": "1", "frequency": "122.8", "mach": "0"})})
    got = to_reading(COM_SBY, {"frequency": "2122.8"}, heard, predict)
    assert got["slots"] == {"com_port": "1", "frequency": "122.8"}
    predict = model({})
    many = "com one standby " + " ".join(["two"] * 9)
    assert to_reading(COM_SBY, {"frequency": "2"}, many, predict) is None
    assert len(predict.asked) == MAX_TO_READINGS


def test_has_valid_target():
    assert has_valid_target(BRAKE, {"autobrake_level": "3"})
    assert has_valid_target(BRAKE, {"autobrake": "max"})
    assert not has_valid_target(BRAKE, {"autobrake_level": "23"})
    assert not has_valid_target(NAV, {"nav_port": "1"})  # no frequency
    assert not has_valid_target(NAV, {"nav_port": "121", "nav_frequency": "110.5"})
