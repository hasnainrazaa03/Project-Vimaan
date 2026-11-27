"""The wake word (phase L.5): "hey copilot", heard on the open microphone
between presses, then the command that follows, in one breath or after a
pause (docs/PHASE_PLAN.md, "L.5 Wake word").

Pure Python: the recogniser is made by a factory (a Vosk KaldiRecognizer with
the grammar ``GRAMMAR``, words on), so this is tested with a fake.

**The wake.** Vosk decodes the stream with a grammar of the phrase or anything
else ("[unk]"). A result is a wake when it holds "hey" and then "copilot",
next to each other, each at ``MIN_CONF`` or more: the grammar lets "copilot"
come alone ("autopilot on" is heard as "[unk] copilot ..."), and in L.5.1
every real detection scored 1.000. Vosk ends a result at a pause, so in one
breath ("hey copilot, gear down") the result holds the command too.

**The command** follows "copilot" (its word time), at once or after a pause.
In one breath it is handed on from the start of the wake phrase, to be
decoded with the phrase's words in the grammar and the phrase then taken off
the front (:func:`strip_wake`): cut at the word time instead, one breath lost
the command's first word ("down" for "gear down", "pilot on" for "autopilot
on", L.5.2), as Vosk's end of "copilot" runs into the next word. After a
pause (``PAUSE_SEC`` of quiet, then speech) the cut is clean: the command is
handed on alone, from just before it starts, for the command grammar as on
the talk key. Decoded from the phrase through the pause instead, the generated
voice's "heading one eight zero" was heard "heading one h zero" (L.5.3).

Vosk can't say when the command is over: its partial results come about once a
second, and it didn't end an utterance of "[unk]" in 1.5 s of quiet. So it ends
by loudness: a 20 ms frame is speech when it is ``SPEECH_OVER_FLOOR`` times the
background (measured in the seconds before the wake phrase) and at least
``SPEECH_RMS_MIN`` (the owner's clips: quiet frames about 5, speech about
5,000). Speech 10 dB over noise is about 3 times its level, so the factor is 2.
It ends once ``QUIET_END_SEC`` pass with no speech; nothing said within
``COMMAND_START_SEC`` is no command at all.

Times are counted in samples fed, not by the clock, so a file played through
it (the test bridge) behaves as the microphone does.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PHRASE = ("hey", "copilot")
GRAMMAR = [" ".join(PHRASE), "[unk]"]
# Each wake word's confidence (L.5.1: true 1.000, false 0.64 / 0.87).
MIN_CONF = 0.95
# After the wake: the command must start within this...
COMMAND_START_SEC = 4.0
# ...ends once nothing more is said for this long...
QUIET_END_SEC = 0.7
# ...and is cut off after this, as a held key is at 12 s.
COMMAND_MAX_SEC = 8.0
# Sound kept to cut the command out of: a wake found at the end of a long
# result (one breath) reaches back to where "copilot" ended.
KEEP_SEC = 15.0
# Speech by loudness, in 20 ms frames.
FRAME_SEC = 0.02
SPEECH_RMS_MIN = 300.0
SPEECH_OVER_FLOOR = 2.0
FLOOR_SEC = 3.0  # the background: this much sound before the wake phrase
FLOOR_PERCENTILE = 0.5
# Loudness is judged over this many frames (100 ms): speech is sustained,
# a clatter in the background is brief.
SMOOTH_FRAMES = 5
# The command keeps this much after its last speech (the recogniser wants
# the word's tail).
TAIL_SEC = 0.3
# Quiet this long after "copilot", then speech: a pause. The command is then
# handed on alone, from this much before its speech is first seen (the
# loudness is judged over 100 ms, so it is seen a little late).
PAUSE_SEC = 0.4
LEAD_SEC = 0.5
SAMPLE_WIDTH = 2


@dataclass(frozen=True)
class Wake:
    """The wake phrase heard: its words' lowest confidence, and whether the
    command came straight after it in the same breath."""

    conf: float
    one_breath: bool


@dataclass(frozen=True)
class Command:
    """The wake phrase and the command after it, to the end of what was said
    (decoded with :func:`wake_grammar`, then :func:`strip_wake`)."""

    pcm: bytes
    seconds: float
    cut_off: bool  # stopped at COMMAND_MAX_SEC, not by a pause
    # The wake phrase is in front (one breath): decode with wake_grammar,
    # then strip_wake. False: the command alone, after a pause.
    with_phrase: bool = True


NO_COMMAND = "no_command"  # nothing said within COMMAND_START_SEC


def find_wake(words: list[dict[str, Any]], min_conf: float = MIN_CONF) -> tuple[int, float] | None:
    """In a Vosk result's words, the index of the first wake word and the
    phrase's lowest confidence, or None."""
    n = len(PHRASE)
    for i in range(len(words) - n + 1):
        run = words[i : i + n]
        if tuple(w.get("word") for w in run) == PHRASE:
            conf = min(float(w.get("conf", 0.0)) for w in run)
            if conf >= min_conf:
                return i, conf
    return None


def wake_grammar(words: list[str]) -> list[str]:
    """A grammar (Vosk word list) with the wake phrase's words added: a wake's
    command is decoded with the phrase still in front of it."""
    return [*words, *(w for w in PHRASE if w not in words)]


def strip_wake(text: str) -> str:
    """The transcript with the wake phrase (or what was heard of it) taken off
    the front: "hey copilot gear down" -> "gear down"."""
    words = text.split()
    i = 0
    while i < min(len(words), len(PHRASE)) and words[i] in PHRASE:
        i += 1
    return " ".join(words[i:])


def frame_rms(pcm: bytes) -> float:
    n = len(pcm) // SAMPLE_WIDTH
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm[: n * SAMPLE_WIDTH])
    return math.sqrt(sum(s * s for s in samples) / n)


class WakeSpotter:
    """Feed it the open microphone's sound between presses; it returns what
    happened: a :class:`Wake`, then a :class:`Command` or ``NO_COMMAND``."""

    def __init__(
        self,
        make_recognizer: Callable[[], Any],
        sample_rate: int = 16_000,
        min_conf: float = MIN_CONF,
    ) -> None:
        self.make_recognizer = make_recognizer
        self.sample_rate = sample_rate
        self.min_conf = min_conf
        self.frame = int(FRAME_SEC * sample_rate)
        self.reset()

    def reset(self) -> None:
        """Start afresh: a key pressed, Vimaan speaking, a gap in the sound."""
        self._rec = self.make_recognizer()
        self._fed = 0  # samples fed since the reset
        self._buf = bytearray()  # the last KEEP_SEC of them
        self._buf_start = 0  # the sample _buf starts at
        self._command: dict[str, Any] | None = None

    @property
    def in_command(self) -> bool:
        return self._command is not None

    def feed(self, pcm: bytes) -> list[Wake | Command | str]:
        """New 16-bit mono sound; returns the events it completed, in order."""
        out: list[Wake | Command | str] = []
        if not pcm:
            return out
        self._buf += pcm
        self._fed += len(pcm) // SAMPLE_WIDTH
        excess = len(self._buf) // SAMPLE_WIDTH - int(KEEP_SEC * self.sample_rate)
        if excess > 0:
            del self._buf[: excess * SAMPLE_WIDTH]
            self._buf_start += excess
        # A command in hand ends first: a result in the same sound may be the
        # next wake.
        if self._command is not None:
            self._scan()
            self._check_end(out)
        if self._rec.AcceptWaveform(pcm) and self._command is None:
            words = json.loads(self._rec.Result()).get("result") or []
            self._heard(words, out)
            if self._command is not None:  # one breath: its command may be over
                self._scan()
                self._check_end(out)
        return out

    def _heard(self, words: list[dict[str, Any]], out: list[Wake | Command | str]) -> None:
        found = find_wake(words, self.min_conf)
        if found is None:
            return
        first, conf = found
        start = self._sample(words[first].get("start", 0.0))
        end = self._sample(words[first + len(PHRASE) - 1].get("end", 0.0))
        self._command = {
            "phrase": start,  # handed on from here, the phrase and all
            "from": end,  # the command starts where "copilot" ends
            "woke": self._fed,
            "spoke": None,  # the end of its last speech frame
            "scanned": end,
            "quiet": 0,  # samples of quiet since the last speech (or "copilot")
            "after_pause": None,  # where speech first came after a pause
            "threshold": self._threshold(start),
        }
        out.append(Wake(round(conf, 3), one_breath=len(words) > first + len(PHRASE)))

    def _threshold(self, phrase_start: int) -> float:
        """Speech loudness: well over the background before the wake phrase."""
        first = max(self._buf_start, phrase_start - int(FLOOR_SEC * self.sample_rate))
        rms = [
            frame_rms(self._slice(s, s + self.frame))
            for s in range(first, phrase_start - self.frame + 1, self.frame)
        ]
        floor = sorted(rms)[int(len(rms) * FLOOR_PERCENTILE)] if rms else 0.0
        return max(SPEECH_RMS_MIN, SPEECH_OVER_FLOOR * floor)

    def _scan(self) -> None:
        """Each whole 20 ms frame since the last look: speech, or not, by its
        loudness over the last ``SMOOTH_FRAMES`` frames."""
        cmd = self._command
        assert cmd is not None
        s = cmd["scanned"]
        recent = cmd.setdefault("recent", [])
        while s + self.frame <= self._fed:
            recent.append(frame_rms(self._slice(s, s + self.frame)))
            del recent[:-SMOOTH_FRAMES]
            if sum(recent) / len(recent) >= cmd["threshold"]:
                pause = cmd["quiet"] >= PAUSE_SEC * self.sample_rate
                if cmd["after_pause"] is None and cmd["spoke"] is None and pause:
                    cmd["after_pause"] = s
                cmd["spoke"] = s + self.frame
                cmd["quiet"] = 0
            else:
                cmd["quiet"] += self.frame
            s += self.frame
        cmd["scanned"] = s

    def _check_end(self, out: list[Wake | Command | str]) -> None:
        cmd = self._command
        assert cmd is not None
        rate = self.sample_rate
        if cmd["spoke"] is None:
            if self._fed - cmd["woke"] >= COMMAND_START_SEC * rate:
                self._command = None
                out.append(NO_COMMAND)
            return
        cut_off = self._fed - cmd["from"] >= COMMAND_MAX_SEC * rate
        if cut_off or self._fed - cmd["spoke"] >= QUIET_END_SEC * rate:
            end = self._fed if cut_off else min(self._fed, cmd["spoke"] + int(TAIL_SEC * rate))
            after = cmd["after_pause"]
            start = (
                cmd["phrase"] if after is None else max(cmd["from"], after - int(LEAD_SEC * rate))
            )
            pcm = self._slice(start, end)
            self._command = None
            seconds = round(len(pcm) / (SAMPLE_WIDTH * rate), 2)
            out.append(Command(pcm, seconds, cut_off, with_phrase=after is None))

    def _sample(self, sec: Any) -> int:
        return min(self._fed, max(0, int(round(float(sec) * self.sample_rate))))

    def _slice(self, start: int, end: int) -> bytes:
        a = max(0, start - self._buf_start) * SAMPLE_WIDTH
        b = max(0, end - self._buf_start) * SAMPLE_WIDTH
        return bytes(self._buf[a:b])


def vosk_spotter(model: Any, sample_rate: int = 16_000, vosk: Any = None) -> WakeSpotter:
    """A spotter on a loaded ``vosk.Model`` (the worker's own, shared)."""
    if vosk is None:
        import vosk as vosk_module  # deferred: optional dependency

        vosk = vosk_module

    def make() -> Any:
        rec = vosk.KaldiRecognizer(model, sample_rate, json.dumps(GRAMMAR))
        rec.SetWords(True)
        return rec

    return WakeSpotter(make, sample_rate)


def chime_pcm(sample_rate: int = 16_000) -> bytes:
    """A short rising two-note chime (16-bit mono): the wake word was heard,
    say the command. Made here, so nothing is shipped or loaded."""
    out = []
    for freq, sec in ((880.0, 0.07), (1320.0, 0.09)):
        n = int(sample_rate * sec)
        for i in range(n):
            fade = min(1.0, i / (0.01 * sample_rate), (n - i) / (0.02 * sample_rate))
            out.append(int(9000 * fade * math.sin(2 * math.pi * freq * i / sample_rate)))
    return struct.pack(f"<{len(out)}h", *out)
