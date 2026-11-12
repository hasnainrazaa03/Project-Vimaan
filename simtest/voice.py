"""Spoken commands without a pilot: speech generated with macOS ``say``, as
16 kHz mono 16-bit WAV (what Vimaan's recogniser takes), and an offline check
of what Vosk hears in it, so a test only uses voices it can understand."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .bridge import default_log_dir

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "ML") not in sys.path:
    sys.path.insert(0, str(_REPO / "ML"))

# Clear system voices on a stock macOS install ("say -v ?"). Offline, on ten
# Tier 1 phrases (2026-11-12), Vosk heard Daniel right 9/10, Samantha, Karen
# and Moira 7/10, Fred 4/10, Ralph 3/10.
DEFAULT_VOICE = "Daniel"
CANDIDATE_VOICES = ("Daniel", "Samantha", "Karen", "Moira", "Fred", "Ralph")
SAMPLE_RATE = 16_000
_VOICE_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2}_[A-Z]{2})\s+#")
XPLANE_ML = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common/X-Plane 12"
    / "Resources/plugins/PythonPlugins/ML"
)


def available() -> bool:
    return shutil.which("say") is not None


def voices() -> list[str]:
    """Installed English voices, by the name ``say -v`` takes."""
    if not available():
        return []
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    names: list[str] = []
    for line in out.splitlines():
        m = _VOICE_LINE.match(line)
        if m and m["locale"].startswith("en_"):
            name = m["name"].split(" (")[0]
            if name not in names:
                names.append(name)
    return names


def vosk_model_path() -> Path:
    """Vimaan's Vosk model: $VIMAAN_VOSK_MODEL, this checkout, the main
    checkout (from a git worktree), or the copy installed in X-Plane."""
    from vimaan_nlu import stt

    name = Path(stt.DEFAULT_VOSK_MODEL).name
    candidates = (
        [Path(os.environ["VIMAAN_VOSK_MODEL"])] if os.environ.get("VIMAAN_VOSK_MODEL") else []
    )
    candidates.append(_REPO / "ML" / "models" / "vosk" / name)
    git = _REPO / ".git"
    if git.is_file():  # a worktree: ".git" names the main repository's git folder
        common = Path(git.read_text().split("gitdir:", 1)[1].strip()).parents[2]
        candidates.append(common / "ML" / "models" / "vosk" / name)
    candidates.append(XPLANE_ML / "models" / "vosk" / name)
    for c in candidates:
        if (c / "am").is_dir():
            return c
    raise FileNotFoundError(f"no Vosk model {name}; set VIMAAN_VOSK_MODEL")


def synthesize(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate_wpm: int | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """A WAV of ``text`` in ``voice`` (cached by text, voice and rate)."""
    if not available():
        raise RuntimeError("macOS 'say' is not available here")
    folder = cache_dir or default_log_dir() / "test" / "voice"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{rate_wpm}|{text}".encode()).hexdigest()[:16]
    path = folder / f"{voice.split()[0].lower()}_{key}.wav"
    if not path.exists():
        cmd = ["say", "-v", voice, "-o", str(path), "--file-format=WAVE"]
        cmd += [f"--data-format=LEI16@{SAMPLE_RATE}"]
        if rate_wpm:
            cmd += ["-r", str(rate_wpm)]
        subprocess.run([*cmd, text], check=True, capture_output=True)
    return path


@lru_cache(maxsize=1)
def _recogniser_factory() -> Any:
    import vosk
    from vimaan_nlu import stt, stt_vocabulary

    vosk.SetLogLevel(-2)  # errors only: the grammar's "vimaan" warns on every recogniser
    model = vosk.Model(str(vosk_model_path()))
    words = stt_vocabulary.grammar()
    return lambda: stt.VoskStream(model, SAMPLE_RATE, words, vosk=vosk)


def hear(path: Path) -> str:
    """What Vimaan's recogniser (Vosk, with its grammar) hears in a WAV."""
    import wave

    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    stream = _recogniser_factory()()
    stream.begin()
    stream.feed(raw)
    return stream.finish()
