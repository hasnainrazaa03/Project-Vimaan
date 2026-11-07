"""vimaan_nlu.ambiguity: a target that depends on hearing "to" or "two"."""

import pytest
from vimaan_nlu.ambiguity import bare_number, chosen, two_or_to

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
