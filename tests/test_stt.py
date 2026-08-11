"""Tests for vimaan_nlu.stt (Phase 7) — STT backend dispatch.

Uses a fake recognizer so no speech_recognition / model / audio is needed.
"""

import pytest
from vimaan_nlu.stt import SUPPORTED_BACKENDS, transcribe


class FakeRecognizer:
    def __init__(self, vosk_return='{"text": "vosk heard this"}'):
        self.calls = []
        self._vosk_return = vosk_return

    def recognize_google(self, audio, language="en-US"):
        self.calls.append(("google", language))
        return "google heard this"

    def recognize_vosk(self, audio):
        self.calls.append(("vosk",))
        return self._vosk_return

    def recognize_whisper(self, audio, model="base.en"):
        self.calls.append(("whisper", model))
        return "whisper heard this"

    def recognize_sphinx(self, audio):
        self.calls.append(("sphinx",))
        return "sphinx heard this"


class TestTranscribe:
    def test_default_is_google(self):
        r = FakeRecognizer()
        assert transcribe(r, object()) == "google heard this"
        assert r.calls[0][0] == "google"

    def test_none_backend_falls_back_to_google(self):
        r = FakeRecognizer()
        assert transcribe(r, object(), None) == "google heard this"

    def test_backend_is_case_insensitive(self):
        r = FakeRecognizer()
        assert transcribe(r, object(), "GOOGLE") == "google heard this"

    def test_vosk_parses_json_text(self):
        r = FakeRecognizer(vosk_return='{"text": "gear up"}')
        assert transcribe(r, object(), "vosk") == "gear up"

    def test_vosk_accepts_dict(self):
        r = FakeRecognizer(vosk_return={"text": "flaps down"})
        assert transcribe(r, object(), "vosk") == "flaps down"

    def test_vosk_malformed_falls_back_to_str(self):
        r = FakeRecognizer(vosk_return="not json at all")
        assert transcribe(r, object(), "vosk") == "not json at all"

    def test_whisper_passes_model(self):
        r = FakeRecognizer()
        assert transcribe(r, object(), "whisper", whisper_model="small.en") == "whisper heard this"
        assert ("whisper", "small.en") in r.calls

    def test_sphinx(self):
        r = FakeRecognizer()
        assert transcribe(r, object(), "sphinx") == "sphinx heard this"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            transcribe(FakeRecognizer(), object(), "telepathy")

    def test_supported_backends_listed(self):
        assert set(SUPPORTED_BACKENDS) == {"google", "vosk", "whisper", "sphinx"}
