"""Your own voice for a checklist: each phrase it says, recorded once, for
``python -m simtest run <checklist> <aircraft> --recordings <folder>``.

    python -m simtest record relative all
    python -m simtest record checklists-ground all   # the six names, confirm, cancel
    python -m simtest record wake all                # "hey copilot ..." ( ... a pause)

Each phrase is shown in turn: Enter starts, Enter stops. The audio is 16 kHz
mono 16-bit WAV, what Vimaan's recogniser takes, captured by the plugin's own
recorder with its pre-roll (``VIMAAN_MIC_PREROLL_MS``, 300 ms by default). Run
it again with the same ``--out`` to go on where you stopped.

The recordings stay on this Mac (``~/Vimaan_Logs/recordings``). They measure
how Vimaan hears you, in the sim; nothing is trained or tuned on them.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import wave
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import checklists, relative, tier1, tier2, tier3, tier4, wake
from .bridge import default_log_dir
from .scenario import Step

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "ML") not in sys.path:
    sys.path.insert(0, str(_REPO / "ML"))

SAMPLE_RATE = 16_000
MANIFEST = "manifest.json"
# A take shorter than this, or quieter, didn't catch the phrase: asked again.
MIN_SEC = 0.4
MIN_PEAK = 500  # of 32767
CHECKLISTS: dict[str, Callable[[str], list[Step]]] = {
    "tier1": tier1.steps,
    "tier2": tier2.steps,
    "tier3": tier3.steps,
    "tier4": tier4.steps,
    "relative": relative.steps,
    "checklists-ground": checklists.ground,
    "checklists-air": checklists.air,
    "wake": wake.steps,
}
# A checklist's offers are answered aloud (scenario.Follow), and a hold the
# confirm meets is cancelled: these are said too.
CHECKLIST_ANSWERS = ("confirm", "cancel")


def phrases(checklist: str, aircraft: Iterable[str]) -> list[tuple[str, bool]]:
    """Every phrase the checklist says on these aircraft, once each, in the
    order it first says them: (phrase, said on the question key). A step's
    fallbacks (``abort``) are said only if it fails, but are recorded too, and
    so are a checklist's answers ("confirm", "cancel")."""
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()

    def walk(steps: Iterable[Step]) -> None:
        for step in steps:
            said = [step.say] if step.say else []
            if step.follow is not None:
                said += CHECKLIST_ANSWERS
            for phrase in said:
                if phrase not in seen:
                    seen.add(phrase)
                    out.append((phrase, step.ask))
            if step.abort:
                walk(step.abort)

    for name in aircraft:
        walk(CHECKLISTS[checklist](name))
    return out


def default_folder(checklist: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return default_log_dir() / "recordings" / f"{checklist}-{stamp}"


def load_manifest(folder: Path) -> dict[str, Any]:
    path = Path(folder) / MANIFEST
    if not path.exists():
        return {"recordings": {}}
    return json.loads(path.read_text())


def latest_folder(checklist: str, root: Path | None = None) -> Path:
    """Your newest recordings for ``checklist`` (``run --recordings latest``):
    the folder whose manifest is this checklist's and was saved last."""
    root = root or default_log_dir() / "recordings"
    found = []
    for path in root.glob(f"*/{MANIFEST}"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("checklist") == checklist and manifest.get("recordings"):
            found.append((path.stat().st_mtime, path.parent))
    if not found:
        raise FileNotFoundError(
            f"no recordings for {checklist} in {root} yet: python -m simtest record {checklist} all"
        )
    return max(found)[1]


def load_recordings(folder: str | os.PathLike[str]) -> dict[str, Path]:
    """phrase -> its recording, for the runner. Only files that exist."""
    folder = Path(folder).expanduser()
    if not (folder / MANIFEST).exists():
        raise FileNotFoundError(f"no {MANIFEST} in {folder}: record with python -m simtest record")
    out = {}
    for phrase, entry in load_manifest(folder)["recordings"].items():
        path = folder / entry["file"]
        if path.exists():
            out[phrase] = path
    return out


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def _peak(pcm: bytes) -> int:
    import array

    samples = array.array("h", pcm[: len(pcm) - len(pcm) % 2])
    return max((abs(s) for s in samples), default=0)


def take_problem(pcm: bytes) -> str | None:
    """Why a take can't be used (nothing heard), or None. Says nothing about
    what Vimaan will make of it."""
    seconds = len(pcm) / (2 * SAMPLE_RATE)
    if seconds < MIN_SEC:
        return f"only {seconds:.1f} s"
    if _peak(pcm) < MIN_PEAK:
        return "too quiet (is the right microphone selected?)"
    return None


def record(
    checklist: str,
    aircraft: list[str],
    folder: Path,
    recorder: Any = None,
    ask: Callable[[str], str] = input,
    log: Callable[[str], None] = print,
) -> int:
    """Record each phrase not recorded yet in ``folder``. Returns how many
    phrases are still missing."""
    from vimaan_nlu.audio_capture import PushToTalkRecorder

    folder.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(folder)
    manifest.update(checklist=checklist, aircraft=aircraft, sample_rate=SAMPLE_RATE)
    done: dict[str, Any] = manifest["recordings"]
    todo = phrases(checklist, aircraft)
    if recorder is None:
        preroll = float(os.environ.get("VIMAAN_MIC_PREROLL_MS", "300")) / 1000
        recorder = PushToTalkRecorder(preroll_sec=preroll)
        recorder.listen()
        manifest["device"] = recorder.describe().get("using")
        log(f"Microphone: {manifest['device']} (the system's input; use the one you fly with)")
    log(f"{len(todo)} phrases for {checklist} on {', '.join(aircraft)}; {len(done)} recorded.")
    log("Say each one as you would in flight. Enter starts, Enter stops.")
    if any(" ... " in phrase for phrase, _ in todo):
        log('" ... " means pause for a second there, as you would after "hey copilot".')
    log("At the prompt: r = redo the one before, s = skip, q = stop for now.\n")

    def save() -> None:
        (folder / MANIFEST).write_text(json.dumps(manifest, indent=1, ensure_ascii=False))

    i = 0
    while i < len(todo):
        phrase, on_question_key = todo[i]
        if phrase in done and not done[phrase].get("redo"):
            i += 1
            continue
        key = "  (the question key)" if on_question_key else ""
        log(f"[{i + 1}/{len(todo)}]  “{phrase}”{key}")
        answer = _ask_while_listening(recorder, ask, "  Enter to start: ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            i += 1
            continue
        if answer == "r":
            back = max(0, i - 1)
            prev = todo[back][0]
            if prev in done:
                done[prev]["redo"] = True
            i = back
            continue
        pcm = _take(recorder, ask)
        problem = take_problem(pcm) or (str(recorder.error) if recorder.error else None)
        if problem:
            log(f"  Not kept: {problem}. Once more.")
            continue
        entry = done.get(phrase, {})
        name = entry.get("file") or f"{i + 1:03d}.wav"
        write_wav(folder / name, pcm)
        done[phrase] = {
            "file": name,
            "seconds": round(len(pcm) / (2 * SAMPLE_RATE), 2),
            "takes": entry.get("takes", 0) + 1,
            "question_key": on_question_key,
        }
        save()
        i += 1
    save()
    missing = sum(phrase not in done for phrase, _ in todo)
    log(f"\nSaved in {folder}: {len(todo) - missing} of {len(todo)} phrases.")
    if missing:
        log("Run the same command with --out to record the rest.")
    else:
        log(f"Next: python -m simtest run {checklist} <aircraft> --recordings {folder}")
    return missing


def _ask_while_listening(recorder: Any, ask: Callable[[str], str], prompt: str) -> str:
    """``ask(prompt)``, reading the open microphone meanwhile. Between takes
    that keeps the pre-roll the latest sound (unread, PortAudio's buffer
    overflows and the recorder drops it), as the worker does between presses."""
    answer: list[str] = []
    done = threading.Event()

    def wait() -> None:
        try:
            answer.append(ask(prompt))
        finally:
            done.set()

    threading.Thread(target=wait, daemon=True).start()
    while not done.is_set():
        recorder.poll()
        time.sleep(0.02)
    return answer[0] if answer else "q"


def _take(recorder: Any, ask: Callable[[str], str]) -> bytes:
    """One recording, from Enter to Enter."""
    recorder.start()
    _ask_while_listening(recorder, ask, "  ● recording, Enter to stop ")
    return recorder.stop()
