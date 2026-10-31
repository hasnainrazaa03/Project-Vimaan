"""The command table (vimaan_nlu.commands) is the single source of truth:
these tests hold it to that and to what it replaced."""

import json
import os
import sys

import pytest
from fake_xplane import DEFAULT_DATAREFS, KNOWN_COMMANDS
from vimaan_nlu import commands
from vimaan_nlu.commands import COMMANDS, SLOT_VALIDATORS
from vimaan_nlu.safety import RULES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def test_training_schema_unchanged_from_v11():
    # E0 is a refactor: the generator must see exactly the schema v11 was trained on.
    with open(os.path.join(ROOT, "tests", "fixtures", "schema_v11.json")) as fh:
        assert commands.training_schema() == json.load(fh)


def test_intents_unique():
    names = [c.intent for c in COMMANDS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("c", commands.actionable(), ids=lambda c: c.intent)
def test_actionable_command_is_complete(c):
    assert c.command_words, "fragment guard needs command words"
    assert c.bindings, "every action touches X-Plane"
    for slot in c.required:
        assert slot in c.slots
    for slot in c.slots:
        assert slot in SLOT_VALIDATORS, f"{slot} has no validator"
    for rule in c.safety:
        assert rule in RULES
    for template in c.templates:
        assert any("{" + s + "}" in template for s in c.slots), template


def test_non_actionable_commands_have_no_handler_bits():
    for c in COMMANDS:
        if not c.actionable:
            assert not (c.slots or c.bindings or c.safety or c.self_test), c.intent


def test_every_binding_exists_in_the_fake_sim():
    # The fake only knows names verified against X-Plane 12.4's lists.
    known = set(DEFAULT_DATAREFS) | KNOWN_COMMANDS
    assert commands.all_bindings() <= known, commands.all_bindings() - known


def test_self_test_probes_cover_each_family():
    probed = {intent for _, intent in commands.self_test_probes()}
    families = {c.intent.rstrip("_12") for c in commands.actionable()}
    assert {i.rstrip("_12") for i in probed} == families


def test_generated_files_are_current():
    import build_command_docs
    import build_stt_vocabulary

    assert build_command_docs.main(["--check"]) == 0, "run scripts/build_command_docs.py"
    assert build_stt_vocabulary.main(["--check"]) == 0, "run scripts/build_stt_vocabulary.py"


def test_dataref_checker_sees_table_bindings():
    import check_xplane_refs

    assert commands.all_bindings() <= check_xplane_refs.table_names()
    assert "sim/autopilot/servos2_toggle" in check_xplane_refs.table_names()  # from params
