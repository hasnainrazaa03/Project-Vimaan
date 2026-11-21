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


ALT = "set_autopilot_altitude"


@pytest.mark.parametrize(
    ("intent", "text", "refused"),
    [
        # Words that say a change: v26 flew these to the number as an altitude.
        (ALT, "climb another two thousand", True),
        (ALT, "climb forty eight hundred feet more", True),
        (ALT, "please down by fifteen hundred feet", True),
        (ALT, "go ahead and one thousand feet higher please", True),
        (ALT, "descend a further two thousand", True),
        (ALT, "come down two thousand feet", True),
        (ALT, "go up one thousand", True),
        # The altitude: a word makes the number the target.
        (ALT, "climb up to eight thousand", False),
        (ALT, "come down to five thousand", False),
        (ALT, "descend and maintain four thousand", False),
        (ALT, "altitude eight thousand", False),
        # A bare climb is asked about (climb_reading), not refused.
        (ALT, "descend two thousand feet", False),
        # Only an altitude read so.
        ("change_altitude_by", "climb another two thousand", False),
        ("set_vertical_speed", "climb at five hundred feet per minute more", False),
    ],
)
def test_change_said_as_altitude(intent, text, refused):
    assert rel.change_said_as_altitude(intent, text) is refused


HDG = "set_autopilot_heading"


@pytest.mark.parametrize(
    ("intent", "text", "refused"),
    [
        # Words that say a turn: v26 set these as the heading.
        (HDG, "forty five degrees to the right", True),
        (HDG, "please turn right another one hundred twenty nine degrees", True),
        (HDG, "turn left one five one degrees", True),
        (HDG, "turn right by one oh four right away", True),
        (HDG, "give me one oh nine degrees right", True),
        # The heading: "heading", or "to" before the number.
        (HDG, "turn left heading two seven zero", False),
        (HDG, "turn left to one five zero degrees", False),
        (HDG, "turn left to 150", False),
        (HDG, "set heading to ten degrees", False),
        (HDG, "heading two seven zero degrees", False),
        # "Right away" names no side.
        (HDG, "three zero four degrees right away", False),
        # A bare turn of 100-179 is asked about (turn_reading), not refused.
        (HDG, "turn right one two zero", False),
        # Only a heading read so.
        ("turn_by", "forty five degrees to the right", False),
    ],
)
def test_change_said_as_heading(intent, text, refused):
    assert rel.change_said_as_heading(intent, text) is refused


@pytest.mark.parametrize(
    ("slots", "text", "refused"),
    [
        # "forward" heard as "four right" (Speech Commands, 2 of 300).
        ({"turn": "right", "turn_degrees": "4"}, "four right", True),
        ({"turn": "left", "turn_degrees": "5"}, "left five", True),
        ({"turn": "left", "turn_degrees": "9"}, "left nine", True),
        # A word that says it's a turn.
        ({"turn": "right", "turn_degrees": "4"}, "turn right four", False),
        ({"turn": "left", "turn_degrees": "5"}, "come left five", False),
        ({"turn": "right", "turn_degrees": "4"}, "right four degrees", False),
        ({"turn": "right", "turn_degrees": "1"}, "one degree right", False),
        # Ten and more turn as before.
        ({"turn": "left", "turn_degrees": "10"}, "left ten", False),
        ({"turn": "left", "turn_degrees": "20"}, "left twenty", False),
    ],
)
def test_a_small_turn_needs_a_word_that_says_turn(slots, text, refused):
    assert rel.small_turn_unsaid("turn_by", slots, text) is refused


def test_only_a_turn_is_held_to_the_small_turn_words():
    assert not rel.small_turn_unsaid("set_autopilot_heading", {"degrees": "4"}, "four right")
    assert not rel.small_turn_unsaid("turn_by", {"turn": "right"}, "right")


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


@pytest.mark.parametrize(
    ("text", "side"),
    [
        ("turn left 20", "left"),
        ("turn left to 148 right away", "left"),  # "right away" names no side
        ("turn right 20 right away", "right"),
        ("turn left right 20", None),
        ("heading 270", None),
    ],
)
def test_side_said(text, side):
    assert rel.side_said(text) == side


@pytest.mark.parametrize(
    ("text", "heading"),
    [
        ("turn left to 15", True),
        ("turn left heading 270", True),
        ("15 degrees to the left", False),  # "to the left" names the side
        ("i would like to turn left 20", False),  # "to" before a verb
        ("turn left 20", False),
    ],
)
def test_says_heading(text, heading):
    assert rel.says_heading(text) is heading


H, T = "set_autopilot_heading", "turn_by"


@pytest.mark.parametrize(
    ("intent", "slots", "text", "resolved"),
    [
        # v27's first candidate: the model's command, then the words'.
        (H, {"degrees": "90"}, "turn right 90", (T, {"turn": "right", "turn_degrees": "90"})),
        (T, {"turn": "left", "turn_degrees": "15"}, "turn left to 15", (H, {"degrees": "15"})),
        (H, {"turn_degrees": "42"}, "confirm turn left to 42", (H, {"degrees": "42"})),
        (T, {"turn": "left", "turn_degrees": "024"}, "turn left 024", (H, {"degrees": "024"})),
        (T, {"turn": "left", "turn_degrees": "244"}, "turn left 244", (H, {"degrees": "244"})),
        (
            T,
            {"turn": "right", "turn_degrees": "148"},
            "turn left to 148 right away",
            (H, {"degrees": "148"}),
        ),
        (
            T,
            {"turn": "right", "turn_degrees": "20"},
            "turn left 20 right away",
            (T, {"turn": "left", "turn_degrees": "20"}),
        ),
        # 100 to 179 with nothing saying which: a turn, which the plugin asks about.
        (H, {"degrees": "120"}, "turn right 120", (T, {"turn": "right", "turn_degrees": "120"})),
        (
            T,
            {"turn": "left", "turn_degrees": "120"},
            "turn left 120 degrees",
            (T, {"turn": "left", "turn_degrees": "120"}),
        ),
        (
            T,
            {"turn": "left", "turn_degrees": "15"},
            "15 degrees to the left",
            (T, {"turn": "left", "turn_degrees": "15"}),
        ),
        # No side, or both: a turn loses its side (refused), a heading stays.
        (
            T,
            {"turn": "left", "turn_degrees": "20"},
            "turn left right 20",
            (T, {"turn_degrees": "20"}),
        ),
        (H, {"degrees": "270"}, "fly 270 degrees", None),
        # Nothing to decide.
        (H, {}, "heading", None),
        ("set_autopilot_altitude", {"altitude": "5000"}, "climb to 5000", None),
    ],
)
def test_resolve_turn(intent, slots, text, resolved):
    assert rel.resolve_turn(intent, slots, text) == resolved


def test_resolve_turn_keeps_other_slots():
    out = rel.resolve_turn(H, {"degrees": "90", "instance": "1"}, "turn right 90")
    assert out == (T, {"instance": "1", "turn": "right", "turn_degrees": "90"})
