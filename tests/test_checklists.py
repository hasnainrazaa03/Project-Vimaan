"""Checklists (vimaan_nlu.checklists, phase L.4): the approved lists, the
check of each item and its words, and when a checklist is suggested."""

import pytest
from vimaan_nlu import checklists
from vimaan_nlu.checklists import CHECKLISTS, NAMES, Suggestions, check, summary
from vimaan_nlu.commands import BY_INTENT, required_slots
from vimaan_nlu.validators import validate_slot


def items(name):
    return [(i.name, i.only) for i in CHECKLISTS[name]]


def test_the_lists_are_the_approved_ones():
    # docs/PHASE_PLAN.md, "L.4 Checklists" (approved 2026-11-22), in order.
    assert NAMES == tuple(CHECKLISTS) == BY_INTENT["run_checklist"].slots["checklist"].values
    assert items("before takeoff") == [
        ("flaps", "jet"),
        ("flaps", "prop"),
        ("transponder", ""),
        ("landing lights", ""),
        ("strobes", ""),
        ("auto brake", "jet"),
        ("speed brakes", ""),
        ("fuel pumps", ""),
        ("seat belt sign", ""),
        ("yaw damper", ""),
        ("autopilot", ""),
    ]
    assert items("after takeoff") == [("gear", ""), ("flaps", ""), ("auto brake", "")]
    assert items("approach") == [
        ("seat belt sign", ""),
        ("landing lights", ""),
        ("auto brake", "jet"),
    ]
    assert items("before landing") == [
        ("gear", ""),
        ("flaps", ""),
        ("speed brakes", "jet"),
        ("landing lights", ""),
    ]
    assert items("after landing") == [
        ("speed brakes", ""),
        ("flaps", ""),
        ("auto brake", ""),
        ("strobes", ""),
        ("landing lights", ""),
        ("taxi lights", ""),
    ]
    assert items("shutdown") == [
        ("parking brake", ""),
        ("engines", ""),
        ("beacon", ""),
        ("fuel pumps", ""),
        ("seat belt sign", ""),
        ("taxi lights", ""),
        ("landing lights", ""),
    ]


def test_checked_only_items():
    # The table's "check only": props' takeoff flaps and the landing flaps.
    only = {(n, i.name) for n in NAMES for i in CHECKLISTS[n] if i.offer is None}
    assert only == {("before takeoff", "flaps"), ("before landing", "flaps")}
    assert all(i.hint for n in NAMES for i in CHECKLISTS[n] if i.offer is None)


@pytest.mark.parametrize(
    "item", [i for n in NAMES for i in CHECKLISTS[n] if i.offer is not None], ids=str
)
def test_every_offer_is_a_command_the_plugin_takes(item):
    intent, slots = item.offer
    assert intent in BY_INTENT and BY_INTENT[intent].handler
    assert set(required_slots().get(intent, ())) <= set(slots)
    for name, value in slots.items():
        assert name in BY_INTENT[intent].slots
        validator = {"instance": None}.get(name, name)  # "both" is the engines' own
        if validator:
            assert validate_slot(validator, value)[1], (name, value)
    assert item.ask and not item.ask.endswith("?")


def test_an_offer_sets_what_the_item_wants():
    # Each offer's own words name the state the item passes on.
    for name in NAMES:
        for i in CHECKLISTS[name]:
            if i.offer is None or i.reads != "switch":
                continue
            slots = i.offer[1]
            assert slots.get("switch", slots.get("state")) in i.want, (name, i.name)


OFFER = "Landing lights off. Set landing lights on? Say confirm, or cancel."


@pytest.mark.parametrize(
    ("state", "words", "said", "offered"),
    [
        ("on", "Landing lights on", "Landing lights on, checked", False),
        ("off", "Landing lights off", OFFER, True),
        ("mixed", "Landing lights partly on", OFFER.replace("off.", "partly on."), True),
        (None, "", "I can't read the landing lights", False),
    ],
)
def test_a_switch_checked(state, words, said, offered):
    result = check(checklists.LANDING_LIGHTS_ON, state, words)
    assert (result.said, result.ok, result.offer is not None) == (said, state == "on", offered)
    if offered:
        assert result.offer == ("toggle_lights", {"light": "landing", "switch": "on"})


def test_check_only_says_what_is_wrong():
    flaps = CHECKLISTS["before landing"][1]
    assert check(flaps, "up", "Flaps up").said == "Flaps up. Not set for landing"
    assert check(flaps, "first", "Flaps 5").said == "Flaps 5, checked"
    prop = CHECKLISTS["before takeoff"][1]
    assert check(prop, "out", "Flaps 30").said == "Flaps 30. Not set for takeoff"
    assert check(prop, "up", "Flaps up").ok and check(prop, "first", "Flaps 10").ok
    jet = CHECKLISTS["before takeoff"][0]
    assert not check(jet, "up", "Flaps up").ok and check(jet, "out", "Flaps 15").ok
    assert check(jet, "up", "Flaps up").offer == ("set_flaps", {"position": "down"})


def test_the_beacon_stays_on_until_the_engines_are_off():
    beacon = CHECKLISTS["shutdown"][2]
    on = check(beacon, "on", "Beacon on", engines_running=True)
    assert (on.ok, on.offer, on.said) == (False, None, "Beacon on, until the engines are off")
    assert check(beacon, "off", "Beacon off", engines_running=True).ok
    off = check(beacon, "on", "Beacon on", engines_running=False)
    assert off.said == "Beacon on. Set beacon off? Say confirm, or cancel."


def test_fuel_pumps_on_auto_pass():
    pumps = CHECKLISTS["before takeoff"][7]
    assert check(pumps, "auto", "Fuel pumps auto").ok
    assert not check(CHECKLISTS["shutdown"][3], "auto", "Fuel pumps auto").ok


def test_the_summary():
    assert summary("approach", []) == "Approach checklist complete"
    assert summary("shutdown", ["beacon"]) == "Shutdown checklist done, one item open: beacon"
    assert (
        summary("before takeoff", ["strobes", "landing lights", "strobes", "autopilot"])
        == "Before takeoff checklist done, 3 items open: strobes, landing lights and autopilot"
    )


def test_the_pace_follows_the_words():
    short, long = (
        checklists.speaking_sec("Gear down, checked"),
        checklists.speaking_sec(
            "Landing lights off. Set landing lights on? Say confirm, or cancel."
        ),
    )
    assert checklists.GAP_SEC < short < long < 8.0


def flight(s, steps):
    """Feed (phase, engines running, gear) samples; what was suggested."""
    return [n for n in (s.update(*step) for step in steps) if n is not None]


GROUND = [("parked", False, 1.0), ("parked", True, 1.0), ("taxi", True, 1.0)]
DEPART = [("takeoff_roll", True, 1.0), ("initial_climb", True, 0.5), ("climb", True, 0.0)]
ARRIVE = [
    ("cruise", True, 0.0),
    ("descent", True, 0.0),
    ("approach", True, 0.0),
    ("approach", True, 0.6),
    ("approach", True, 1.0),
    ("landing", True, 1.0),
    ("rollout", True, 1.0),
    ("taxi", True, 1.0),
    ("parked", True, 1.0),
]


def test_a_whole_flight_suggests_each_once_in_order():
    s = Suggestions()
    assert flight(s, GROUND + DEPART + ARRIVE) == list(NAMES)
    # Stopping and moving on again while taxiing out: not again.
    s = Suggestions()
    assert flight(s, [*GROUND, ("parked", True, 1.0), ("taxi", True, 1.0)]) == ["before takeoff"]


def test_a_go_around_brings_the_climb_and_the_approach_round_again():
    s = Suggestions()
    approach = [("descent", True, 0.0), ("approach", True, 0.0), ("approach", True, 1.0)]
    go_around = [("go_around", True, 1.0), ("initial_climb", True, 0.0), ("climb", True, 0.0)]
    said = flight(s, GROUND + DEPART + approach + go_around + approach)
    assert said == [
        "before takeoff",
        "after takeoff",
        "approach",
        "before landing",
        "after takeoff",
        "approach",
        "before landing",
    ]


def test_a_go_around_read_back_into_the_approach_is_not_a_new_approach():
    # vimaan_nlu.flight_phase (#99): a go-around below 1,000 ft, configured
    # and descending, is the approach again.
    s = Suggestions()
    steps = [("descent", True, 1.0), ("approach", True, 1.0), ("go_around", True, 1.0)]
    assert flight(s, [*steps, ("approach", True, 1.0)]) == ["approach"]


def test_not_after_the_pilot_ran_it_and_not_on_the_first_phase_seen():
    s = Suggestions()
    assert flight(s, GROUND[:2]) == []  # the engines started
    s.ran("before takeoff")
    assert flight(s, GROUND[2:]) == []
    # An air start on final: "approach" is the first phase known.
    s = Suggestions()
    assert flight(s, [("unknown", True, 1.0), ("approach", True, 1.0)]) == []
    s = Suggestions()
    assert flight(s, [("approach", True, 1.0)]) == []


def test_conditions():
    # Before takeoff needs the engines running; shutdown a landing first.
    assert flight(Suggestions(), [("parked", False, 1.0), ("taxi", False, 1.0)]) == []
    assert flight(Suggestions(), [("taxi", True, 1.0), ("parked", True, 1.0)]) == []
    # Fixed gear (None): no gear coming down, no "before landing".
    s = Suggestions()
    fixed = [("descent", True, None), ("approach", True, None), ("approach", True, None)]
    assert flight(s, fixed) == ["approach"]


def test_an_engine_start_after_landing_brings_before_takeoff_round():
    s = Suggestions()
    flight(s, GROUND + DEPART + ARRIVE)
    again = [("parked", False, 1.0), ("parked", True, 1.0), ("taxi", True, 1.0)]
    assert flight(s, again) == ["before takeoff"]
    s.reset()
    assert s.said == set() and not s.landed
