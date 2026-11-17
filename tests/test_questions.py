"""vimaan_nlu.questions: a question is never acted on as a command."""

import pytest
from vimaan_nlu.questions import looks_like_question


@pytest.mark.parametrize(
    "heard",
    [
        # How the owner's questions were heard (recordings of 2026-11-16).
        "is the autopilot off",
        "is the parking brakes set",
        "is the throttle throttle on",
        "r the landing lights on",  # "are the ..."
        "r the strobes off",
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
