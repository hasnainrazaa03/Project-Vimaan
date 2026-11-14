"""Tests for vimaan_nlu.stt_vocabulary (the offline recogniser's grammar)."""

import os
import sys

from vimaan_nlu.stt_vocabulary import (
    UNKNOWN,
    build_vocabulary,
    entry_words,
    grammar,
    load_vocabulary,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def test_committed_vocabulary_matches_schema():
    import build_stt_vocabulary

    assert load_vocabulary() == build_stt_vocabulary.expected(), (
        "command_words.txt is stale: run python scripts/build_stt_vocabulary.py"
    )


def test_build_vocabulary():
    schema = {
        "toggle_x": {
            "templates": ["autopilot 1 {state}", "{state} co-pilot"],
            "slots": {
                "state": {
                    "type": "categorical",
                    "values": ["on", "off"],
                    "synonyms": {"on": ["engage"], "off": ["shut down"]},
                }
            },
        },
        "set_y": {
            "templates": ["heading {degrees}"],
            "slots": {"degrees": {"type": "numerical", "values": ["270"]}},
        },
    }
    words = build_vocabulary(schema, ["go ahead and"])
    for w in ("autopilot", "one", "on", "off", "engage", "shut", "down", "co", "pilot", "heading"):
        assert w in words
    assert "go" in words and "ahead" in words
    assert "270" not in words and "{state}" not in words  # numbers come from NUMBER_WORDS
    assert "niner" in words and "decimal" in words
    assert words == sorted(set(words))


def test_grammar_ends_with_unknown():
    g = grammar()
    assert g[-1] == UNKNOWN and "flaps" in g and "gear" in g
    assert grammar(["a"]) == ["a", UNKNOWN]


def test_number_soundalikes_excluded():
    # In-sim Vosk heard "two" as "do" every time while "do" was allowed.
    schema = {"ask_time": {"templates": ["do you have the time"], "slots": {}}}
    assert "do" not in build_vocabulary(schema)
    assert "do" not in load_vocabulary()
    assert "like" not in load_vocabulary()  # in-sim: "flight" heard as "like"


def test_a_word_said_only_inside_a_term_is_entered_as_the_term():
    schema = {
        "toggle_ice": {
            "templates": ["{ice} anti ice {switch}", "turn on the anti ice"],
            "slots": {},
        },
        "toggle_pump": {"templates": ["fuel pump {switch}", "boost pump {switch}"], "slots": {}},
    }
    words = build_vocabulary(schema, joined=("anti ice", "fuel pump"))
    assert "anti ice" in words and "anti" not in words and "ice" not in words
    assert "fuel pump" in words and "fuel" not in words
    assert "pump" in words  # also said without "fuel"
    loose = build_vocabulary(
        {"x": {"templates": ["ice {switch}", "anti ice {switch}"], "slots": {}}},
        joined=("anti ice",),
    )
    assert "ice" in loose and "anti ice" in loose and "anti" not in loose
    assert build_vocabulary(schema, joined=("never said",)) == build_vocabulary(schema, joined=())


def test_ice_and_fuel_only_inside_their_terms():
    # Vosk weighs a grammar line as a sentence: "ice" and "fuel" on lines of
    # their own were as likely as any word, and garbled first words in the
    # owner's recordings came back as them.
    g = load_vocabulary()
    assert {"ice", "anti", "fuel"}.isdisjoint(g)
    assert {"anti ice", "fuel pump", "fuel pumps"} <= set(g)


# Words of training phrasings the recogniser may never output, and why.
UNHEARD_OK = {
    # A contraction: the grammar builder keeps letters-only words. Unheard,
    # it decodes to [unk] and drops out ("let's get the flaps up" -> "get the
    # flaps up"), which the model handles.
    "let's",
}


def test_every_word_of_a_clean_command_can_be_heard():
    # The model is trained on these phrasings, so a word missing from the
    # grammar can never reach it: "oh" for zero was, and "course one oh nine"
    # could only be heard as "one nine" (course 19).
    import re

    from data import generate_v2
    from vimaan_nlu.commands import COMMANDS

    rows, _ = generate_v2.generate(per_intent=200, seed=11)
    actionable = {c.intent for c in COMMANDS if c.actionable}
    words = entry_words(load_vocabulary())
    missing = {}
    for r in rows:
        if r["intent"] in actionable and r["noise"] == "none":
            for w in re.findall(r"[a-z']+", r["text"].lower()):
                if w not in words and w not in UNHEARD_OK:
                    missing.setdefault(w, r["text"])
    assert missing == {}
