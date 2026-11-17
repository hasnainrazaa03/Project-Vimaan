"""The real-voice set must be what the shipped grammar hears.

tests/fixtures/real_voice.jsonl stores Vosk's transcripts of real recordings,
and the gate grades models on them. A grammar change alters those transcripts,
so the set records the grammar it was heard with (real_voice.meta.json). Once
it stayed frozen at E1 while Tier 1 and 2 grew the grammar: 12 of 118 clips
were heard differently ("autopilot two on" as "autopilot new on").
The held-out set (real_voice_heldout.jsonl, the owner's 2026-11-16 voice
check) is kept the same way. The gate only measures it.
Pure Python: runs in CI without Vosk or the audio.
"""

import json
from pathlib import Path

import pytest
from vimaan_nlu.stt_vocabulary import grammar_sha256

FIXTURES = Path(__file__).parent / "fixtures"
SETS = {"real_voice": "", "real_voice_heldout": " --set heldout"}


def rows(name):
    return [json.loads(line) for line in (FIXTURES / f"{name}.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("name", sorted(SETS))
def test_heard_with_the_shipped_grammar(name):
    meta = json.loads((FIXTURES / f"{name}.meta.json").read_text())
    assert meta["grammar_sha256"] == grammar_sha256(), (
        f"the grammar changed since {name} was heard: run "
        f"python scripts/build_real_voice_set.py{SETS[name]} (needs the recordings)"
    )


@pytest.mark.parametrize("name", sorted(SETS))
def test_meta_counts_the_clips(name):
    meta = json.loads((FIXTURES / f"{name}.meta.json").read_text())
    assert meta["clips"] == len(rows(name))


def test_no_clip_is_in_both_sets():
    gate = {r["clip"] for r in rows("real_voice")}
    assert not gate & {r["clip"] for r in rows("real_voice_heldout")}


@pytest.mark.parametrize("name", sorted(SETS))
def test_every_clip_names_a_known_command(name):
    from vimaan_nlu.commands import command

    for r in rows(name):
        cmd = command(r["intent"])
        assert cmd is not None and cmd.actionable, r["clip"]
        assert set(r["slots"]) <= set(cmd.slots), r["clip"]


def test_fingerprint_follows_the_words():
    assert grammar_sha256(["a", "b"]) != grammar_sha256(["a", "c"])
    assert grammar_sha256(["a", "b"]) == grammar_sha256(["a", "b"])
