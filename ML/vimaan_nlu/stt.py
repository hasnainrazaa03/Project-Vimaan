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
from typing import Any

SUPPORTED_BACKENDS: tuple[str, ...] = ("google", "vosk", "whisper", "sphinx")


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
        return recognizer.recognize_google(audio, language=language)
    if backend == "vosk":
        # recognize_vosk returns a JSON string, e.g. '{"text": "gear up"}'.
        return _vosk_text(recognizer.recognize_vosk(audio))
    if backend == "whisper":
        return recognizer.recognize_whisper(audio, model=whisper_model)
    if backend == "sphinx":
        return recognizer.recognize_sphinx(audio)
    raise ValueError(f"unknown STT backend: {backend!r} (supported: {SUPPORTED_BACKENDS})")


def _vosk_text(raw: Any) -> str:
    """Pull the transcript out of vosk's JSON (or dict/plain) output."""
    if isinstance(raw, dict):
        return (raw.get("text") or "").strip()
    try:
        return (json.loads(raw).get("text") or "").strip()
    except (ValueError, TypeError, AttributeError):
        return str(raw).strip()
