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


def test_no_text_holds_a_placeholder(data):
    # Templates are filled, and the negatives' word salad is built from
    # template words with every placeholder removed, "{slot=word}" included.
    rows, _ = data
    assert [r["text"] for r in rows if "{" in r["text"] or "}" in r["text"]] == []
    filler, _, _ = generate_v2._vocab()
    assert [w for w in filler if "{" in w or "}" in w or "=" in w] == []


def test_same_seed_same_data_in_any_process():
    # Python orders sets differently in every process (hash seed); the data
    # must not depend on that (data v11 differed in 774 bare-number negatives).
    import os
    import subprocess
    import sys
    from pathlib import Path

    ml = Path(generate_v2.__file__).resolve().parents[1]
    code = (
        "import hashlib, json; from data import generate_v2; "
        "rows, _ = generate_v2.generate(per_intent=40, seed=3); "
        "print(hashlib.sha256(json.dumps(rows).encode()).hexdigest())"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(ml)}
        out = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, "generate_v2 depends on set order"


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
    # A stray word fused onto a value ("eighty fifty degrees" -> 8050) can't be
    # labelled and is dropped; that must stay rare.
    for key, count in stats.items():
        if key.startswith("dropped:") and key != "dropped:negative_collides":
            assert count < 0.01 * 300, key


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


def test_two_digit_headings_beside_three_digit_ones(data):
    rows, _ = data
    values = [r["slots"]["degrees"] for r in rows if r["intent"] == "set_autopilot_heading"]
    two = [v for v in values if len(v) == 2]
    three = [v for v in values if len(v) == 3]
    assert len(two) > 0.1 * len(values) and len(three) > 0.6 * len(values)
    assert all(len(v) in (2, 3) for v in values)


def test_deterministic():
    a, _ = generate_v2.generate(per_intent=50, seed=3)
    b, _ = generate_v2.generate(per_intent=50, seed=3)
    assert a == b


def test_verbs_are_tagged_where_spoken(data):
    rows, _ = data
    gear = [r for r in rows if r["intent"] == "toggle_landing_gear"]
    assert any(r["slots"].get("state") in ("raise", "lower", "retract", "extend") for r in gear)
