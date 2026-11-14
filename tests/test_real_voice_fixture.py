"""The real-voice set must be what the shipped grammar hears.

tests/fixtures/real_voice.jsonl stores Vosk's transcripts of real recordings,
and the gate grades models on them. A grammar change alters those transcripts,
so the set records the grammar it was heard with (real_voice.meta.json). Once
it stayed frozen at E1 while Tier 1 and 2 grew the grammar: 12 of 118 clips
were heard differently ("autopilot two on" as "autopilot new on").
Pure Python: runs in CI without Vosk or the audio.
"""

import json
from pathlib import Path

from vimaan_nlu.stt_vocabulary import grammar_sha256

FIXTURES = Path(__file__).parent / "fixtures"


def test_heard_with_the_shipped_grammar():
    meta = json.loads((FIXTURES / "real_voice.meta.json").read_text())
    assert meta["grammar_sha256"] == grammar_sha256(), (
        "the grammar changed since the real-voice set was heard: "
        "run python scripts/build_real_voice_set.py (needs the recordings)"
    )


def test_meta_counts_the_clips():
    meta = json.loads((FIXTURES / "real_voice.meta.json").read_text())
    rows = (FIXTURES / "real_voice.jsonl").read_text().splitlines()
    assert meta["clips"] == len(rows)


def test_fingerprint_follows_the_words():
    assert grammar_sha256(["a", "b"]) != grammar_sha256(["a", "c"])
    assert grammar_sha256(["a", "b"]) == grammar_sha256(["a", "b"])
