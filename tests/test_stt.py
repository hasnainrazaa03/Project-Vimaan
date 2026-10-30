"""Tests for vimaan_nlu.stt (Phase 7) — STT backend dispatch.

Uses a fake recognizer so no speech_recognition / model / audio is needed.
"""

import os

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


class _Audio:
    def __init__(self, frame_data, sample_rate=16000, sample_width=2):
        self.frame_data, self.sample_rate, self.sample_width = frame_data, sample_rate, sample_width


def test_flac_encoded_in_process_round_trips():
    sf = pytest.importorskip("soundfile")
    import io
    import struct

    from vimaan_nlu.stt import flac_bytes

    pcm = struct.pack("<4h", 0, 1000, -1000, 32767) * 400
    data = flac_bytes(pcm, 16000)
    assert data[:4] == b"fLaC"
    back, rate = sf.read(io.BytesIO(data), dtype="int16")
    assert rate == 16000 and back.tobytes() == pcm


def test_google_uses_in_process_flac_not_bundled_binary():
    # SpeechRecognition's bundled flac-mac is x86_64-only ("Bad CPU type" on
    # Apple Silicon without Rosetta); the Google path must not shell out.
    pytest.importorskip("soundfile")
    from vimaan_nlu.stt import use_inprocess_flac

    audio = _Audio(b"\x00\x00" * 1600)
    assert use_inprocess_flac(audio) is True
    assert audio.get_flac_data(convert_rate=None, convert_width=2)[:4] == b"fLaC"


def test_in_process_flac_skipped_for_non_16bit_audio():
    from vimaan_nlu.stt import use_inprocess_flac

    assert use_inprocess_flac(_Audio(b"\x00" * 100, sample_width=1)) is False


# ---------------------------------------------------------------------------
# VoskStream (streaming offline recognition) with a fake vosk module
# ---------------------------------------------------------------------------
class FakeKaldi:
    def __init__(self, model, rate, grammar=None):
        self.model, self.rate, self.grammar = model, rate, grammar
        self.fed = b""
        self.final = '{"text": "gear up"}'

    def Reset(self):
        self.fed = b""

    def AcceptWaveform(self, pcm):
        self.fed += pcm

    def FinalResult(self):
        return self.final


class FakeVosk:
    def __init__(self):
        self.Model = lambda path: ("model", path)
        self.KaldiRecognizer = FakeKaldi


class TestVoskStream:
    def make(self, words=("gear", "up", "[unk]")):
        from vimaan_nlu.stt import VoskStream

        return VoskStream(
            "/models/x", words=None if words is None else list(words), vosk=FakeVosk()
        )

    def test_grammar_passed_as_json(self):
        import json

        s = self.make()
        assert s.grammar and json.loads(s._rec.grammar) == ["gear", "up", "[unk]"]
        assert s._rec.model == ("model", "/models/x") and s._rec.rate == 16000

    def test_full_vocabulary_without_grammar(self):
        s = self.make(words=None)
        assert not s.grammar and s._rec.grammar is None

    def test_feed_and_finish(self):
        s = self.make()
        s.begin()
        s.feed(b"\x01\x00" * 10)
        s.feed(b"")
        assert s.fed_bytes == 20 and s._rec.fed == b"\x01\x00" * 10
        assert s.finish() == "gear up"
        s.begin()
        assert s.fed_bytes == 0 and s._rec.fed == b""

    def test_unknown_words_dropped(self):
        s = self.make()
        s._rec.final = '{"text": "[unk] gear [unk] up"}'
        assert s.finish() == "gear up"

    @pytest.mark.parametrize("final", ['{"text": ""}', '{"text": "[unk] [unk]"}'])
    def test_nothing_recognised_is_not_understood(self, final):
        from vimaan_nlu.stt import UnknownValueError
        from vimaan_nlu.voice_worker import _error_outcome

        s = self.make()
        s._rec.final = final
        with pytest.raises(UnknownValueError) as exc:
            s.finish()
        assert _error_outcome(exc.value) == "not_understood"


def _real_vosk_ready():
    import shutil

    from vimaan_nlu.stt import DEFAULT_VOSK_MODEL

    try:
        import vosk  # noqa: F401
    except Exception:
        return False
    return os.path.isdir(DEFAULT_VOSK_MODEL) and shutil.which("say") is not None


@pytest.mark.skipif(not _real_vosk_ready(), reason="needs vosk, its model and macOS `say`")
def test_real_vosk_grammar_on_synthesized_speech(tmp_path):
    import subprocess
    import wave

    from vimaan_nlu.stt import DEFAULT_VOSK_MODEL, VoskStream
    from vimaan_nlu.stt_vocabulary import grammar

    stream = VoskStream(DEFAULT_VOSK_MODEL, words=grammar())
    heard = {}
    for phrase in ("set heading two seven zero", "parking brake on", "flaps up"):
        path = tmp_path / "p.wav"
        subprocess.run(
            ["say", "-v", "Samantha", "-o", str(path), "--data-format=LEI16@16000", phrase],
            check=True,
        )
        pcm = wave.open(str(path)).readframes(10**7)
        stream.begin()
        for i in range(0, len(pcm), 640):  # 20 ms chunks, like the live mic
            stream.feed(pcm[i : i + 640])
        heard[phrase] = stream.finish()
    assert heard == {p: p for p in heard}
