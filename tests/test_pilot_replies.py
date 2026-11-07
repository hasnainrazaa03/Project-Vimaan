"""vimaan_nlu.pilot_replies: "override" and "cancel", matched on the words."""

import pytest
from vimaan_nlu.pilot_replies import REPLY_WORDS, pilot_reply
from vimaan_nlu.stt_vocabulary import load_vocabulary


@pytest.mark.parametrize(
    ("heard", "reply"),
    [
        ("override", "override"),
        ("Override", "override"),
        ("confirm", "override"),
        ("affirmative please", "override"),
        ("uh override", "override"),
        ("confirm override", "override"),
        ("cancel", "cancel"),
        ("negative", "cancel"),
        ("disregard", "cancel"),
        ("please cancel", "cancel"),
    ],
)
def test_replies(heard, reply):
    assert pilot_reply(heard) == reply


@pytest.mark.parametrize(
    "heard",
    [
        "confirm gear up",  # a command: the model handles it
        "cancel the autopilot",
        "override engine two",
        "gear up",
        "",
        "please",
    ],
)
def test_anything_longer_is_not_a_reply(heard):
    assert pilot_reply(heard) is None


def test_reply_words_are_in_the_recogniser_grammar():
    assert REPLY_WORDS <= set(load_vocabulary())
