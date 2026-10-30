"""Speech-to-text backend dispatch (Phase 7 / R-09).

Lets the plugin transcribe captured audio with a selectable backend without
rewriting audio capture. ``google`` is the default (online); ``vosk`` and
``whisper`` are offline (no network). The recognizer (a
``speech_recognition.Recognizer``) is injected, so this dispatch is
unit-testable with a fake and imports without pulling ``speech_recognition``
or any model.

Offline backends need their package + a model installed; see
``docs/OFFLINE_STT.md``. Selection is via the plugin's ``VIMAAN_STT_BACKEND``
environment variable.
"""

from __future__ import annotations

import json
import os
from typing import Any

SUPPORTED_BACKENDS: tuple[str, ...] = ("google", "vosk", "whisper", "sphinx")
# Shipped by scripts/install_plugin.py next to the NLU model.
DEFAULT_VOSK_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "vosk",
    "vosk-model-small-en-us-0.15",
)


def transcribe(
    recognizer: Any,
    audio: Any,
    backend: str = "google",
    *,
    whisper_model: str = "base.en",
    language: str = "en-US",
) -> str:
    """Transcribe ``audio`` to a plain string using ``backend``.

    Raises ``ValueError`` for an unknown backend. Backend-specific recognition
    errors (no speech understood, network failure) propagate from the
    recognizer for the caller to handle.
    """
    backend = (backend or "google").lower()
    if backend == "google":
        use_inprocess_flac(audio)
        return recognizer.recognize_google(audio, language=language)
    if backend == "vosk":
        # recognize_vosk returns a JSON string, e.g. '{"text": "gear up"}'.
        return _vosk_text(recognizer.recognize_vosk(audio))
    if backend == "whisper":
        return recognizer.recognize_whisper(audio, model=whisper_model)
    if backend == "sphinx":
        return recognizer.recognize_sphinx(audio)
    raise ValueError(f"unknown STT backend: {backend!r} (supported: {SUPPORTED_BACKENDS})")


def flac_bytes(frame_data: bytes, sample_rate: int) -> bytes:
    """Encode 16-bit mono PCM as FLAC in-process (``soundfile``/libsndfile)."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    pcm = np.frombuffer(frame_data, dtype="<i2")
    sf.write(buf, pcm, sample_rate, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def use_inprocess_flac(audio: Any) -> bool:
    """Make ``audio.get_flac_data`` encode in-process instead of shelling out.

    SpeechRecognition's Google backend converts audio by running a ``flac``
    executable; the one it bundles for macOS is x86_64-only, so on Apple
    Silicon without Rosetta every request fails with "Bad CPU type". Returns
    False (leaving SpeechRecognition's converter in place) when soundfile is
    unavailable or the audio isn't 16-bit.
    """
    if getattr(audio, "sample_width", None) != 2:
        return False
    try:
        import soundfile  # noqa: F401
    except Exception:
        return False

    def get_flac_data(convert_rate=None, convert_width=None):
        # Google asks for >= 8 kHz / 16-bit; capture is already 16 kHz / 16-bit.
        return flac_bytes(audio.frame_data, audio.sample_rate)

    audio.get_flac_data = get_flac_data
    return True


class UnknownValueError(Exception):
    """Nothing intelligible was said. Same name as SpeechRecognition's, so the
    voice worker classifies both as ``not_understood``."""


class VoskStream:
    """Streaming offline recognition: audio is decoded *while* the pilot
    talks, so after key-up only the last few hundred ms remain (measured
    1-2 ms to finish vs ~460 ms for a Google round trip).

    ``words`` restricts output to a grammar (see ``stt_vocabulary``); None
    means Vosk's full vocabulary. ``model`` is a path or a loaded
    ``vosk.Model``; ``vosk`` is injectable for tests.
    """

    def __init__(
        self,
        model: Any,
        sample_rate: int = 16_000,
        words: list[str] | None = None,
        vosk: Any = None,
    ) -> None:
        if vosk is None:
            import vosk as vosk_module  # deferred: optional dependency

            vosk_module.SetLogLevel(-1)
            vosk = vosk_module
        if isinstance(model, str):
            model = vosk.Model(model)
        args = (model, sample_rate) if words is None else (model, sample_rate, json.dumps(words))
        self._rec = vosk.KaldiRecognizer(*args)
        self.sample_rate = sample_rate
        self.grammar = words is not None
        self.fed_bytes = 0

    def begin(self) -> None:
        self._rec.Reset()
        self.fed_bytes = 0

    def feed(self, pcm: bytes) -> None:
        if pcm:
            self._rec.AcceptWaveform(pcm)
            self.fed_bytes += len(pcm)

    def partial(self) -> str:
        """Best guess so far (shown live in the overlay while talking)."""
        try:
            raw = json.loads(self._rec.PartialResult()).get("partial", "")
        except (ValueError, TypeError, AttributeError):
            return ""
        return " ".join(w for w in raw.split() if w != "[unk]")

    def finish(self) -> str:
        """Final transcript of everything fed since :meth:`begin`."""
        text = _vosk_text(self._rec.FinalResult())
        text = " ".join(w for w in text.split() if w != "[unk]")
        if not text:
            raise UnknownValueError("no command words recognised")
        return text


def _vosk_text(raw: Any) -> str:
    """Pull the transcript out of vosk's JSON (or dict/plain) output."""
    if isinstance(raw, dict):
        return (raw.get("text") or "").strip()
    try:
        return (json.loads(raw).get("text") or "").strip()
    except (ValueError, TypeError, AttributeError):
        return str(raw).strip()
