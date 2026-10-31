"""Training data v2 (ML/data/generate_v2.py): properties every dataset must have."""

import dataclasses
import re
from collections import defaultdict

import pytest
from data import generate_v2
from vimaan_nlu.commands import COMMANDS
from vimaan_nlu.normalization import normalize_aviation_input


@pytest.fixture(scope="module")
def data():
    rows, stats = generate_v2.generate(per_intent=300, seed=7)
    return rows, stats


def test_every_slot_is_a_word_of_the_normalized_text(data):
    rows, _ = data
    for r in rows:
        norm = normalize_aviation_input(r["text"])
        for name, value in r["slots"].items():
            assert re.search(r"(?<![\w.])" + re.escape(value) + r"(?![\w.])", norm), (r, name)


def test_no_text_has_two_labels(data):
    rows, _ = data
    labels = defaultdict(set)
    for r in rows:
        labels[r["text"]].add(r["intent"])
    assert [t for t, i in labels.items() if len(i) > 1] == []


def test_held_out_templates_never_trained_on(data):
    rows, _ = data
    split_of = defaultdict(set)
    for r in rows:
        if not r["intent"] == "None":
            split_of[r["template"]].add(r["split"])
    actionable = {c.intent for c in COMMANDS if c.actionable}
    assert all(len(s) == 1 for t, s in split_of.items() if t.split("#")[0] in actionable)
    for intent in actionable:
        held = {t for t, s in split_of.items() if t.startswith(intent + "#") and s == {"test"}}
        assert len(held) >= 2, intent


def test_every_command_and_negatives_present(data):
    rows, stats = data
    intents = {r["intent"] for r in rows}
    assert {c.intent for c in COMMANDS} <= intents
    negatives = [r for r in rows if r["template"].startswith("None#neg")]
    actionable = {c.intent for c in COMMANDS if c.actionable}
    commands = sum(1 for r in rows if r["intent"] in actionable)
    assert len(negatives) >= 0.09 * commands
    assert not [k for k in stats if k.startswith("dropped:") and k != "dropped:negative_collides"]


def test_noise_is_applied_and_real(data):
    rows, _ = data
    actionable = [r for r in rows if r["intent"] not in ("None",) and r["slots"]]
    noisy = [r for r in actionable if r["noise"] != "none"]
    assert 0.1 < len(noisy) / len(actionable) < 0.3
    leads = [r for r in noisy if r["noise"] == "lead"]
    assert all(r["text"].split()[0] in generate_v2.LEADING_NOISE for r in leads)


def test_holdout_stable_when_phrasings_are_added():
    c = next(c for c in COMMANDS if c.intent == "toggle_engine")
    grown = dataclasses.replace(c, templates=(*c.templates, "a new phrasing {state}"))
    held, held_grown = generate_v2.holdout_templates(c), generate_v2.holdout_templates(grown)
    assert len(held & held_grown) >= len(held) - 1


def test_brake_heard_as_the_is_in_the_data(data):
    rows, _ = data
    misheard = [r for r in rows if r["noise"] == "mishear"]
    assert misheard and all(r["intent"] == "toggle_parking_brake" for r in misheard)
    assert all("parking the" in r["text"] for r in misheard)


def test_deterministic():
    a, _ = generate_v2.generate(per_intent=50, seed=3)
    b, _ = generate_v2.generate(per_intent=50, seed=3)
    assert a == b


def test_verbs_are_tagged_where_spoken(data):
    rows, _ = data
    gear = [r for r in rows if r["intent"] == "toggle_landing_gear"]
    assert any(r["slots"].get("state") in ("raise", "lower", "retract", "extend") for r in gear)
