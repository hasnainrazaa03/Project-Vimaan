"""The command table (vimaan_nlu.commands) is the single source of truth:
these tests hold it to that and to what it replaced."""

import json
import os
import re
import sys

import pytest
from fake_xplane import DEFAULT_DATAREFS, KNOWN_COMMANDS
from vimaan_nlu import commands
from vimaan_nlu.commands import COMMANDS, SLOT_VALIDATORS
from vimaan_nlu.safety import RULES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def test_every_v11_intent_still_understood():
    # Older models (v11) emit pre-family names; the legacy map must cover them
    # so the plugin can still run such a model.
    with open(os.path.join(ROOT, "tests", "fixtures", "schema_v11.json")) as fh:
        v11_intents = json.load(fh)
    for intent in v11_intents:
        family = commands.LEGACY_INTENTS[intent][0] if intent in commands.LEGACY_INTENTS else intent
        assert family in commands.BY_INTENT, intent


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
    assert len(c.templates) >= 15, "E1: at least 15 phrasings per command"
    for template in c.templates:
        used = re.findall(r"\{(\w+)(?:=[^}]+)?\}", template)
        bases = {re.sub(r"_(verb|ord)$", "", u) for u in used}
        assert bases, f"no slot in {template!r}"
        assert bases <= set(c.slots), f"unknown placeholder in {template!r}"


def test_non_actionable_commands_have_no_handler_bits():
    for c in COMMANDS:
        if not c.actionable and not c.is_question:
            assert not (c.slots or c.bindings or c.safety or c.self_test), c.intent


QUESTIONS = [c for c in COMMANDS if c.is_question]


@pytest.mark.parametrize("c", QUESTIONS, ids=lambda c: c.intent)
def test_a_question_reads_and_never_acts(c):
    # Tier 4: a question has an answer and no handler, so nothing that treats
    # a command as acting (the gate, safety rules, stray words) sees it as one.
    assert c.handler is None and not c.actionable
    assert not c.safety and c.reply is None
    for slot in c.slots:
        assert slot in SLOT_VALIDATORS, f"{slot} has no validator"
    assert len(c.templates) >= 15, "at least 15 phrasings per question"
    for template in c.templates:
        used = re.findall(r"\{(\w+)(?:=[^}]+)?\}", template)
        bases = {re.sub(r"_(verb|ord)$", "", u) for u in used}
        assert bases <= set(c.slots), f"unknown placeholder in {template!r}"


def test_a_yes_no_question_names_its_question_word():
    # "is the gear down" without "is" is the command "gear down": a question
    # template that holds a state word must start like a question.
    from vimaan_nlu.questions import looks_like_question

    states = {"on", "off", "up", "down", "set", "armed", "running", "engaged"}
    for c in QUESTIONS:
        for template in c.templates:
            words = re.sub(r"\{[^}]*\}", "X", template).split()
            if words[-1] in states:  # "... gear down", not "time remaining on fuel"
                assert looks_like_question(template), (c.intent, template)


def test_every_binding_exists_in_the_fake_sim():
    # The fake only knows names verified against X-Plane 12.4's lists.
    known = set(DEFAULT_DATAREFS) | KNOWN_COMMANDS
    assert commands.all_bindings() <= known, commands.all_bindings() - known


def test_self_test_probes_cover_each_family():
    # Questions too: Ctrl-Z skips the intents the loaded model doesn't know,
    # so v26 isn't failed for questions it was never trained on.
    probed = {intent for _, intent in commands.self_test_probes()}
    families = {c.intent.rstrip("_12") for c in commands.actionable()}
    questions = {c.intent for c in COMMANDS if c.is_question}
    assert {i.rstrip("_12") for i in probed} == families | questions


def test_generated_files_are_current():
    import build_command_docs
    import build_stt_vocabulary

    assert build_command_docs.main(["--check"]) == 0, "run scripts/build_command_docs.py"
    assert build_stt_vocabulary.main(["--check"]) == 0, "run scripts/build_stt_vocabulary.py"


def test_dataref_checker_sees_table_bindings():
    import check_xplane_refs

    assert commands.all_bindings() <= check_xplane_refs.table_names()
    assert "sim/autopilot/servos2_toggle" in check_xplane_refs.table_names()  # from params
