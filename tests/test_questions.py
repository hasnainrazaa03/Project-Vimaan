"""vimaan_nlu.questions: a question is never acted on as a command."""

import pytest
from vimaan_nlu.questions import bare_reading, looks_like_question, one_word


@pytest.mark.parametrize(
    "heard",
    [
        # How the owner's questions were heard (recordings of 2026-11-16).
        "is the autopilot off",
        "is the parking brakes set",
        "is the throttle throttle on",
        "r the landing lights on",  # "are the ..." (command grammar)
        "our the landing lights on",  # "are the ..." (question grammar)
        "r the strobes off",
        "now is the autopilot on",  # the recogniser's stray leading word
        "the is the gear down",
        "what the heading",
        "what r airspeed",
        "what are vertical speed",
        "oh what are ground speed",  # "uh what's our ground speed"
        "how much fuel degree have",
        # Other recognisers keep the full words.
        "what's the altimeter setting",
        "where are the flaps",
        "do we have the seatbelt sign on",
        "isn't the gear down",
        "Is The Gear Down",
    ],
)
def test_questions(heard):
    assert looks_like_question(heard)


@pytest.mark.parametrize(
    "heard",
    [
        # The owner's commands, and commands phrased politely.
        "autopilot off",
        "parking brakes set",
        "heading two seven zero",
        "can you turn on the logo lights",
        "can you please armed speed brakes",
        "oh s landing lights off",
        # "r" alone is a letter: "r t o" (auto brake), and a misheard start
        # of a command ("r lights status strobe two set the auto brake ...").
        "auto brake r t o",
        "r lights status strobe two set the auto brake two three",
        # Status questions without a question word stay with the model:
        # with no on/off they name no command it can act on.
        "autopilot status",
        "landing lights status",
        "",
        "uh",
    ],
)
def test_not_questions(heard):
    assert not looks_like_question(heard)


@pytest.mark.parametrize(
    ("heard", "want"),
    [
        ("heading", ("ask_flight_data", {"reading": "heading"})),
        ("what the heading", ("ask_flight_data", {"reading": "heading"})),
        ("what are altitude", ("ask_flight_data", {"reading": "altitude"})),
        ("what r airspeed", ("ask_flight_data", {"reading": "airspeed"})),
        ("what are vertical speed", ("ask_flight_data", {"reading": "vertical speed"})),
        ("say altitude", ("ask_flight_data", {"reading": "altitude"})),
        ("what are squawk", ("ask_transponder", {})),
        ("check the altimeter", ("ask_flight_data", {"reading": "altimeter"})),
    ],
)
def test_a_bare_reading(heard, want):
    assert bare_reading(heard) == want


@pytest.mark.parametrize(
    "heard",
    [
        "heading two seven zero",  # a value: a command
        "altitude hold",  # a mode
        "heading mode",
        "flaps",  # a switch: its command asks which way
        "gear",
        "fuel",
        "landing lights",
        "what com one",
        "",
    ],
)
def test_not_a_bare_reading(heard):
    assert bare_reading(heard) is None


def test_complete_command_is_what_the_command_model_would_act_on():
    from vimaan_nlu.questions import complete_command

    floor = 0.55
    on = {"light": "landing", "switch": "on"}
    assert complete_command("toggle_lights", on, 0.99, "landing lights on", floor)
    # No on/off: "landing lights status" is a question to the question model.
    assert not complete_command(
        "toggle_lights", {"light": "landing"}, 0.99, "landing lights status", floor
    )
    assert not complete_command("toggle_lights", on, 0.3, "landing lights on", floor)
    assert not complete_command("toggle_lights", on, 0.99, "seventy on", floor)  # no command word
    assert not complete_command("ask_status_generic", {}, 0.99, "status report", floor)
    assert not complete_command("set_flaps", {"position": "check"}, 0.99, "flaps check", floor)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("armed", True),
        ("uh torque", True),
        ("okay", True),
        ("", True),
        ("the armed", False),
        ("what's armed", False),
        ("oil pressure", False),
    ],
)
def test_one_word(text, want):
    assert one_word(text) is want
