"""Phase L.1: relative commands (vimaan_nlu.relative), pure Python."""

import pytest
from vimaan_nlu import relative as rel


@pytest.mark.parametrize(
    ("heading", "side", "degrees", "target"),
    [
        (270, "left", 20, 250),
        (10, "left", 20, 350),
        (350, "right", 20, 10),
        (340, "right", 20, 360),  # never 000
        (183.4, "left", 3, 180),
        (0.4, "right", 90, 90),
    ],
)
def test_turned(heading, side, degrees, target):
    assert rel.turned(heading, side, degrees) == target


def test_altitude_is_measured_from_where_the_autopilot_is_going():
    # Climbing to 11,000: "climb another thousand" is 12,000.
    assert rel.altitude_reference(11000, 10500, 1500, moving=True) == 11000
    # Level, with the window preset elsewhere: from where we are.
    assert rel.altitude_reference(15000, 10012, 20, moving=False) == 10000
    assert rel.altitude_reference(15000, 10012, 20, moving=True) == 10000
    # Descending, but the window is above us: from where we are.
    assert rel.altitude_reference(12000, 8040, -1500, moving=True) == 8000
    # Climbing by hand (no autopilot mode): from where we are.
    assert rel.altitude_reference(11000, 10480, 1500, moving=False) == 10500


@pytest.mark.parametrize(
    ("direction", "feet", "reference", "current", "readings"),
    [
        # At 10,000 "descend two thousand feet": 2,000 or 8,000, so ask.
        ("down", 2000, 10000, 10000, [2000, 8000]),
        # "climb two thousand feet": 2,000 is below us, so only 12,000.
        ("up", 2000, 10000, 10000, [12000]),
        # At 2,500 "descend two thousand": 500 is too low, so only 2,000.
        ("down", 2000, 2500, 2500, [2000]),
        # "descend five hundred": 500 isn't an altitude, so 9,500.
        ("down", 500, 10000, 10000, [9500]),
        # At 5,000 "descend five thousand": both readings are 0 or 5,000.
        ("down", 5000, 5000, 5000, []),
        # Both readings the same: one value, nothing to ask.
        ("down", 5000, 10000, 10000, [5000]),
    ],
)
def test_altitude_readings(direction, feet, reference, current, readings):
    assert rel.altitude_readings(direction, feet, reference, current) == readings


def test_speed_is_measured_from_the_bug_only_when_it_is_flown():
    assert rel.speed_reference(250, 238.6, held=True) == 250
    assert rel.speed_reference(250, 238.6, held=False) == 240
    assert rel.speed_reference(None, 151, held=True) == 150


@pytest.mark.parametrize(
    ("text", "direction"),
    [
        ("descend two thousand feet", "down"),
        ("climb one thousand", "up"),
        ("please climb three thousand feet now", "up"),
        ("descend to two thousand", None),
        ("descend and maintain two thousand", None),
        ("climb another thousand", None),
        ("descend by two thousand feet", None),
        ("climb two thousand more", None),
        ("set altitude two thousand", None),
        ("climb at fifteen hundred feet per minute", None),
    ],
)
def test_bare_climb(text, direction):
    assert rel.bare_climb(text) == direction


@pytest.mark.parametrize(
    ("intent", "slots", "text", "reading"),
    [
        ("turn_by", {"turn": "left", "turn_degrees": "20"}, "turn left twenty", ("turn", ("left", 20))),
        ("turn_by", {"turn": "right", "turn_degrees": "270"}, "turn right two seven zero", ("heading", 270)),
        ("turn_by", {"turn": "left", "turn_degrees": "020"}, "turn left zero two zero", ("heading", 20)),
        ("turn_by", {"turn": "right", "turn_degrees": "120"}, "turn right one two zero", ("either", ("right", 120))),
        ("turn_by", {"turn": "right", "turn_degrees": "120"}, "turn right one twenty degrees", ("turn", ("right", 120))),
        ("turn_by", {"turn": "right", "turn_degrees": "120"}, "turn right by one twenty", ("turn", ("right", 120))),
        ("set_autopilot_heading", {"degrees": "120"}, "turn right one two zero", ("either", ("right", 120))),
        ("set_autopilot_heading", {"degrees": "120"}, "turn right heading one two zero", None),
        ("set_autopilot_heading", {"degrees": "270"}, "turn right two seven zero", None),
        ("set_autopilot_heading", {"degrees": "120"}, "set heading one two zero", None),
        ("turn_by", {"turn": "left", "turn_degrees": "400"}, "turn left four hundred", None),
    ],
)  # fmt: skip
def test_turn_reading(intent, slots, text, reading):
    assert rel.turn_reading(intent, slots, text) == reading


@pytest.mark.parametrize(
    ("intent", "slots", "text", "reading"),
    [
        ("set_autopilot_altitude", {"altitude": "2000"}, "descend two thousand feet", ("down", 2000)),
        ("change_altitude_by", {"direction": "down", "altitude_change": "2000"}, "descend two thousand feet", ("down", 2000)),
        ("set_autopilot_altitude", {"altitude": "8000"}, "descend eight thousand feet", None),
        ("set_autopilot_altitude", {"altitude": "2000"}, "descend to two thousand feet", None),
        ("change_altitude_by", {"direction": "down", "altitude_change": "2000"}, "descend another two thousand", None),
        ("set_autopilot_altitude", {"altitude": "2050"}, "climb two thousand fifty", None),
    ],
)  # fmt: skip
def test_climb_reading(intent, slots, text, reading):
    assert rel.climb_reading(intent, slots, text) == reading


@pytest.mark.parametrize(
    ("text", "conflict"),
    [
        ("turn north", False),
        ("heading south please", False),
        ("turn west twenty", True),
        ("turn left south", True),
        ("heading south one eight zero", True),
    ],
)
def test_compass_conflict(text, conflict):
    assert rel.compass_conflict(text) is conflict


def test_change_words():
    assert rel.says_change("up a thousand feet")
    assert rel.says_change("climb another thousand")
    assert not rel.says_change("climb a thousand feet")
