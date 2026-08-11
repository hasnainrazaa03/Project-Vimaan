"""Gold AUDIO end-to-end regression (Phase 6E).

Feeds recorded wavs through offline STT (whisper, deterministic) -> the real
model, and asserts intent. This is the full mic-to-command path minus the mic.

Activates only when you provide fixtures: drop wavs in ``tests/audio_fixtures/``
and list them in ``tests/audio_fixtures/manifest.jsonl`` (see
``manifest.example.jsonl`` and the README). Without a manifest there are no
cases and the module is a no-op. Needs torch + whisper + speech_recognition, so
it is skipped in CI.
"""

import json
import os

import pytest

requires_torch = pytest.mark.requires_torch

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "audio_fixtures")
MANIFEST = os.path.join(FIXTURES_DIR, "manifest.jsonl")


def _load_manifest():
    if not os.path.isfile(MANIFEST):
        return []
    with open(MANIFEST) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="session")
def gold_audio_model():
    from utils import get_latest_model_path

    if not get_latest_model_path():
        pytest.skip("no trained model present")
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader()
    loader.load_all()
    return loader


@requires_torch
@pytest.mark.parametrize("entry", _load_manifest(), ids=lambda e: e.get("wav", "?"))
def test_gold_audio(entry, gold_audio_model):
    sr = pytest.importorskip("speech_recognition")
    pytest.importorskip("whisper")  # deterministic offline transcription
    from vimaan_nlu import predict
    from vimaan_nlu.stt import transcribe

    wav = os.path.join(FIXTURES_DIR, entry["wav"])
    if not os.path.isfile(wav):
        pytest.skip(f"missing wav: {entry['wav']}")

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav) as source:
        audio = recognizer.record(source)
    text = transcribe(recognizer, audio, "whisper", whisper_model="base.en")

    r = predict(
        text,
        gold_audio_model.model,
        gold_audio_model.tokenizer,
        gold_audio_model.device,
        gold_audio_model.intent_map_rev,
        gold_audio_model.slot_map_rev,
    )
    assert r["intent"] == entry["intent"], f"{entry['wav']}: {text!r} -> {r['intent']}"
    for key, val in (entry.get("slots") or {}).items():
        assert r["slots"].get(key) == val, f"{entry['wav']}: slot {key}={r['slots'].get(key)!r}"
