"""Tests for vimaan_nlu.stt_vocabulary (the offline recogniser's grammar)."""

import os
import sys

from vimaan_nlu.stt_vocabulary import UNKNOWN, build_vocabulary, grammar, load_vocabulary

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def test_committed_vocabulary_matches_schema():
    import build_stt_vocabulary

    assert load_vocabulary() == build_stt_vocabulary.expected(), (
        "command_words.txt is stale: run python scripts/build_stt_vocabulary.py"
    )


def test_generator_prefixes_match_dataset_generator():
    import build_stt_vocabulary

    src = open(os.path.join(ROOT, "ML", "data", "generate_slot_dataset.py")).read()
    for phrase in build_stt_vocabulary.PREFIXES + build_stt_vocabulary.SUFFIXES:
        assert f'"{phrase}"' in src


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
