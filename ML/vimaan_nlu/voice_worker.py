"""Out-of-process voice pipeline: microphone -> speech-to-text -> NLU.

Runs as a child process of X-Plane, started by the plugin with XPPython3's
``xp.pythonExecutable``. Inside X-Plane, Python background threads only run
when the simulator's main thread lets go of the GIL, so audio capture, STT and
the model all competed with rendering: in-sim that cost a ~118 ms frame hitch
per command and several ms per frame while transcribing. Here they have their
own interpreter and their own cores; the plugin only exchanges small messages
and pays microseconds per frame. torch is never imported into X-Plane at all.

Protocol: one JSON object per line. The plugin writes requests to our stdin;
we write replies to stdout (the real stdout is reserved for the protocol —
stray prints are redirected to stderr, which the plugin sends to a log file).

Requests (``op``):
    start {cmd, ask}     open the mic and record (ask: the question key)
    stop {cmd}           key released: keep recording a short tail, then
                         transcribe + predict and reply ``result``
    text {cmd, text, ask}  NLU only (no audio)
    selftest {probes}    NLU on [text, expected] pairs; reply ``selftest``
    ping                 reply ``pong``
    quit                 exit
Replies (``type``):
    ready, fatal, recording, partial, captured, result, error, selftest, pong, log,
    usage (every --usage-every seconds: Mac CPU / memory / GPU, vimaan_nlu.usage)
    (``partial`` = live transcript while recording, streaming backends only;
    ``result.to_reading`` = the command with one "two" read as "to", when the
    number is impossible as heard: see vimaan_nlu.ambiguity.to_reading)

EOF on stdin (the plugin or X-Plane went away) makes the worker exit.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import select
import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

from vimaan_nlu.ambiguity import to_reading
from vimaan_nlu.audio_health import is_silent, rms_from_pcm16
from vimaan_nlu.commands import command
from vimaan_nlu.multi import two_commands
from vimaan_nlu.questions import looks_like_question
from vimaan_nlu.usage import USAGE_SEC, UsageSampler

MAX_RECORD_SEC = 12.0  # recording is cut off after this even if still held
MIN_RECORD_SEC = 0.3  # shorter than this is an accidental tap, not a command
# Google STT has no timeout by default: one stalled request would block the
# worker forever and every later command would time out. Fail it instead.
STT_TIMEOUT_SEC = 8.0
# Keep recording this long after the key is released: PortAudio hands audio
# over ~100-160 ms late, and pilots let go as they finish the last word. Without
# it, in-sim Google heard "gear" / "parking brake" with the "up" / "on" cut off.
RELEASE_TAIL_SEC = 0.3
# Each command's audio is saved (16 kHz mono WAV, ~32 KB/s) so real flights
# become a test set for comparing recognisers (scripts/stt_benchmark.py).
# Saving stops once the folder holds this much.
MAX_AUDIO_DIR_BYTES = 500 * 1024 * 1024
# Live transcript for the overlay: at most this often, only when it changed.
PARTIAL_INTERVAL_SEC = 0.1
# The mic stays open between presses and each recording starts with this much
# sound from before the key went down: opening the mic takes ~85 ms, and
# pilots start the first word as they press. 0 opens the mic on each press.
DEFAULT_PREROLL_MS = 300
# Between presses the open mic is drained this often (PortAudio's buffer holds
# 0.2 s, audio_capture.RING_BUFFER_SEC).
LISTEN_POLL_SEC = 0.05
# A recording played from a file (the test bridge) when there's no microphone
# to take the rate from: what every recogniser here wants.
FILE_SAMPLE_RATE = 16_000


class WavRecorder:
    """Stands in for the microphone: "records" a 16-bit mono WAV in real time
    (debug / regression runs via ``--audio-file`` or VIMAAN_WORKER_AUDIO_FILE).
    Same interface as ``PushToTalkRecorder``."""

    def __init__(self, path: str) -> None:
        import wave

        with wave.open(path) as w:
            if w.getsampwidth() != 2 or w.getnchannels() != 1:
                raise ValueError(f"{path}: need 16-bit mono PCM")
            self.sample_rate = w.getframerate()
            self._pcm = w.readframes(w.getnframes())
        self.path = path
        self.error: BaseException | None = None
        self.overflows, self.truncated, self.busy, self.is_open = 0, False, False, False
        self._t0: float | None = None
        self._taken = 0
        self._streamed = 0  # frames handed out by take_new()

    def describe(self) -> dict[str, Any]:
        return {"using": f"WAV file {self.path}", "sample_rate": self.sample_rate}

    def start(self) -> None:
        self._t0, self._taken, self._streamed, self.is_open = time.perf_counter(), 0, 0, True

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    def poll(self) -> int:
        if self._t0 is None:
            return 0
        due = min(len(self._pcm) // 2, int(self.elapsed() * self.sample_rate))
        n, self._taken = due - self._taken, due
        return n

    def take_new(self) -> bytes:
        new = self._pcm[self._streamed * 2 : self._taken * 2]
        self._streamed = self._taken
        return new

    def stop(self) -> bytes:
        self.poll()
        raw = self._pcm[: self._taken * 2]
        self._t0, self.is_open = None, False
        return raw

    def wait_idle(self, timeout: float = 2.0) -> None:
        pass

    def seconds(self, raw: bytes) -> float:
        return len(raw) / (2 * self.sample_rate)


class UnknownValueError(Exception):
    """Nothing heard that the question grammar knows (named as the
    recognisers' own, so it is classified the same: "not_understood")."""


def _error_outcome(exc: BaseException) -> str:
    # Matched by name so this module (and its tests) don't import
    # speech_recognition just to classify its exceptions.
    name = type(exc).__name__
    if name == "UnknownValueError":
        return "not_understood"
    if name in ("RequestError", "URLError", "TimeoutError", "timeout"):
        return "stt_service_error"
    return "worker_error"


class VoiceWorker:
    """Protocol state machine. Dependencies are injected so the whole
    request/reply flow is testable in-process with fakes."""

    def __init__(
        self,
        send: Callable[[dict[str, Any]], None],
        recorder: Any,
        transcribe_audio: Callable[[bytes, int], str],
        predict_text: Callable[[str], dict[str, Any]],
        *,
        release_tail_sec: float = RELEASE_TAIL_SEC,
        clock: Callable[[], float] = time.perf_counter,
        stream_stt: Any = None,
        question_stt: Any = None,
        predict_question: Callable[[str], dict[str, Any]] | None = None,
        audio_dir: str | None = None,
        audio_prefix: str = "",
    ) -> None:
        self.send = send
        self.recorder = recorder
        self.transcribe_audio = transcribe_audio
        self.predict_text = predict_text
        self.release_tail_sec = release_tail_sec
        self.clock = clock
        # Optional streaming recogniser (stt.VoskStream): fed while recording,
        # used instead of transcribe_audio.
        self.stream_stt = stream_stt
        # Tier 4: the question grammar (stt.VoskStream). The question key's
        # recording, and one that starts like a question on the talk key, are
        # decoded again with it: question words in the command grammar made
        # commands worse ("flight level" heard as "light level").
        self.question_stt = question_stt
        # Tier 4: the question model (qN), beside the command model. It knows
        # only questions, so whatever it reads, the plugin can only answer.
        self.predict_question = predict_question
        self.audio_dir = audio_dir
        self.audio_prefix = audio_prefix
        self._audio_dir_bytes = _dir_bytes(audio_dir) if audio_dir else 0
        self.rec: dict[str, Any] | None = None  # the recording in progress

    # ------------------------------------------------------------------
    def handle(self, msg: dict[str, Any]) -> bool:
        """Process one request. Returns False when the worker should exit."""
        op = msg.get("op")
        cmd = msg.get("cmd")
        if op == "start":
            if self.rec is not None:
                return True  # key auto-repeat
            if self.recorder is None:
                self.send(
                    {"type": "error", "cmd": cmd, "outcome": "mic_error", "error": "no microphone"}
                )
                return True
            self.recorder.start()
            if self.stream_stt is not None:
                self.stream_stt.begin()
            self.rec = {
                "cmd": cmd,
                "ask": bool(msg.get("ask")),
                "partial": "",
                "partial_at": 0.0,
                "t_start": self.clock(),
                "released_at": None,
                "announced": False,
            }
        elif op == "stop":
            if self.rec is not None and self.rec["cmd"] == cmd:
                self.rec["released_at"] = self.clock()
                if self.release_tail_sec <= 0:
                    self._finish()
        elif op == "text":
            self._predict_and_reply(
                cmd, msg.get("text", ""), stt_ms=None, audio=None, ask=bool(msg.get("ask"))
            )
        elif op == "wav":
            self._play_file(cmd, msg.get("path") or "", ask=bool(msg.get("ask")))
        elif op == "selftest":
            results = []
            for text, expected in msg.get("probes", []):
                asked = command(expected)
                on_question_model = (
                    self.predict_question is not None and asked is not None and asked.is_question
                )
                predict = self.predict_question if on_question_model else self.predict_text
                try:
                    assert predict is not None
                    p = predict(text)
                    results.append(
                        {
                            "text": text,
                            "expected": expected,
                            "intent": p["intent"],
                            "confidence": p["confidence"],
                            "slots": p["slots"],
                        }
                    )
                except Exception as exc:
                    results.append({"text": text, "expected": expected, "error": repr(exc)})
            self.send({"type": "selftest", "results": results})
        elif op == "ping":
            self.send({"type": "pong"})
        elif op == "quit":
            return False
        return True

    def tick(self) -> None:
        """Advance the recording in progress, or keep the open mic's pre-roll
        fresh between presses. Call often (every ~10 ms while recording)."""
        rec = self.rec
        if rec is None:
            self._listen()
            return
        if self.recorder.error is not None:
            err, self.recorder.error = self.recorder.error, None
            self.rec = None
            self.recorder.stop()
            self.send(
                {
                    "type": "error",
                    "cmd": rec["cmd"],
                    "outcome": "mic_error",
                    "error": f"{type(err).__name__}: {err}",
                }
            )
            return
        if not rec["announced"] and self.recorder.is_open:
            rec["announced"] = True
            self.send(
                {
                    "type": "recording",
                    "cmd": rec["cmd"],
                    "open_ms": round((self.clock() - rec["t_start"]) * 1000, 1),
                }
            )
        self.recorder.poll()
        if self.stream_stt is not None:
            self.stream_stt.feed(self.recorder.take_new())
            self._send_partial(rec)
        released = rec["released_at"]
        if released is not None:
            if self.clock() - released >= self.release_tail_sec:
                self._finish()
        elif self.recorder.elapsed() > MAX_RECORD_SEC + 0.5:
            rec["released_at"] = self.clock() - self.release_tail_sec  # missed key-up
            self._finish(auto_stop=True)

    @property
    def idle(self) -> bool:
        return self.rec is None

    @property
    def listening(self) -> bool:
        return self.recorder is not None and bool(getattr(self.recorder, "is_listening", False))

    def _listen(self) -> None:
        if self.recorder is None:
            return
        if self.listening:
            self.recorder.poll()
        err = getattr(self.recorder, "listen_error", None)
        if err is not None:
            self.recorder.listen_error = None
            self.send(
                {
                    "type": "log",
                    "level": "warning",
                    "msg": f"mic closed between presses ({type(err).__name__}: {err}); "
                    "the next press opens it again",
                }
            )

    # ------------------------------------------------------------------
    def _finish(self, auto_stop: bool = False) -> None:
        rec, self.rec = self.rec, None
        if rec is None:
            return
        preroll_frames = getattr(self.recorder, "preroll_frames", 0)
        raw = self.recorder.stop()
        self._recognise(
            rec["cmd"],
            raw,
            self.recorder.sample_rate,
            live=True,
            ask=rec.get("ask", False),
            overflows=self.recorder.overflows,
            truncated=self.recorder.truncated,
            auto_stop=auto_stop,
            preroll_sec=round(preroll_frames / self.recorder.sample_rate, 3),
        )

    def _play_file(self, cmd: Any, path: str, ask: bool = False) -> None:
        """A recording from a file, recognised exactly like a push-to-talk one
        (the automated-test bridge, Phase A). 16-bit mono PCM at the
        recogniser's rate."""
        if self.rec is not None:
            self.send({"type": "error", "cmd": cmd, "outcome": "worker_error", "error": "busy"})
            return
        import wave

        rate = int(getattr(self.recorder, "sample_rate", 0) or FILE_SAMPLE_RATE)
        try:
            with wave.open(path) as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    raise ValueError("need 16-bit mono PCM")
                if w.getframerate() != rate:
                    raise ValueError(f"sample rate {w.getframerate()}, need {rate}")
                raw = w.readframes(w.getnframes())
        except (OSError, EOFError, ValueError, wave.Error) as exc:
            self.send(
                {
                    "type": "error",
                    "cmd": cmd,
                    "outcome": "worker_error",
                    "stage": "capture",
                    "error": f"{path}: {exc}",
                }
            )
            return
        self._recognise(cmd, raw, rate, live=False, ask=ask, source_file=path)

    def _recognise(
        self, cmd: Any, raw: bytes, rate: int, *, live: bool, ask: bool = False, **captured: Any
    ) -> None:
        """Recognise one recording, then predict and reply. ``live``: it came
        from the microphone, and a streaming recogniser has been fed already."""
        audio_sec = len(raw) / (2 * rate)
        rms = rms_from_pcm16(raw)
        self.send(
            {
                "type": "captured",
                "cmd": cmd,
                "audio_sec": round(audio_sec, 2),
                "rms": round(rms, 1),
                **captured,
                "audio_file": self._save_audio(cmd, raw, rate),
            }
        )
        # The pre-roll is from before the key went down: not part of the press.
        if audio_sec - captured.get("preroll_sec", 0.0) < MIN_RECORD_SEC:
            self.send({"type": "error", "cmd": cmd, "outcome": "too_short"})
            return
        if is_silent(rms):
            self.send({"type": "error", "cmd": cmd, "outcome": "silent_mic"})
            return
        t = self.clock()
        try:
            if self.stream_stt is not None:
                if live:
                    self.stream_stt.feed(raw[self.stream_stt.fed_bytes :])  # the release tail
                else:
                    self.stream_stt.begin()
                    self.stream_stt.feed(raw)
                text = self.stream_stt.finish()
            else:
                text = self.transcribe_audio(raw, rate)
        except Exception as exc:
            if ask and self.question_stt is not None:
                text = ""  # the question grammar may still hear it
            else:
                self._stt_error(cmd, exc, t)
                return
        heard = None
        if self.question_stt is not None and (ask or looks_like_question(text)):
            asked = self._hear_question(raw)
            if asked:
                heard, text = text, asked
            elif not text:
                self._stt_error(cmd, UnknownValueError("no question words recognised"), t)
                return
        self._predict_and_reply(
            cmd,
            text,
            stt_ms=round((self.clock() - t) * 1000, 1),
            audio=audio_sec,
            ask=ask,
            heard=heard,
        )

    def _hear_question(self, raw: bytes) -> str:
        """The recording decoded with the question grammar, or "" when it
        heard nothing it knows."""
        try:
            self.question_stt.begin()
            self.question_stt.feed(raw)
            return str(self.question_stt.finish())
        except Exception:
            return ""

    def _stt_error(self, cmd: Any, exc: BaseException, t: float) -> None:
        self.send(
            {
                "type": "error",
                "cmd": cmd,
                "outcome": _error_outcome(exc),
                "stage": "stt",
                "error": f"{type(exc).__name__}: {exc}",
                "stt_ms": round((self.clock() - t) * 1000, 1),
            }
        )

    def _send_partial(self, rec: dict[str, Any]) -> None:
        now = self.clock()
        if now - rec["partial_at"] < PARTIAL_INTERVAL_SEC:
            return
        rec["partial_at"] = now
        text = self.stream_stt.partial() if hasattr(self.stream_stt, "partial") else ""
        if text and text != rec["partial"]:
            rec["partial"] = text
            self.send({"type": "partial", "cmd": rec["cmd"], "text": text})

    def _save_audio(self, cmd: Any, raw: bytes, rate: int) -> str | None:
        if not self.audio_dir or not raw:
            return None
        if self._audio_dir_bytes + len(raw) > MAX_AUDIO_DIR_BYTES:
            return None
        import wave

        try:
            os.makedirs(self.audio_dir, exist_ok=True)
            path = os.path.join(self.audio_dir, f"{self.audio_prefix}{cmd}.wav")
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(raw)
        except OSError as exc:
            self.send({"type": "log", "level": "warning", "msg": f"audio not saved: {exc}"})
            return None
        self._audio_dir_bytes += len(raw) + 44
        return path

    def _predict_and_reply(
        self,
        cmd: Any,
        text: str,
        stt_ms: float | None,
        audio: float | None,
        ask: bool = False,
        heard: str | None = None,
    ) -> None:
        t = self.clock()
        try:
            p = self.predict_text(text)
        except Exception as exc:
            self.send(
                {
                    "type": "error",
                    "cmd": cmd,
                    "outcome": "worker_error",
                    "stage": "nlu",
                    "error": f"{type(exc).__name__}: {exc}",
                    "text": text,
                    "traceback": traceback.format_exc(),
                }
            )
            return
        # A number impossible as heard: try "two" as "to" (vimaan_nlu.ambiguity).
        # A second look only: if it fails, the command still goes as heard.
        try:
            meant = to_reading(p["intent"], p["slots"], text, self.predict_text)
        except Exception as exc:
            meant = None
            self.send({"type": "log", "msg": f"to_reading failed: {type(exc).__name__}: {exc}"})
        # Two commands in one breath (phase L.3.1, vimaan_nlu.multi): the
        # plugin refuses them. A second look only: if it fails, the command
        # goes as heard.
        try:
            two = two_commands(text, self.predict_text, whole_reading=p)
        except Exception as exc:
            two = None
            self.send({"type": "log", "msg": f"two_commands failed: {type(exc).__name__}: {exc}"})
        question = None
        if self.predict_question is not None:
            # A second look only: if it fails, the command model's reading stands.
            try:
                q = self.predict_question(text)
                question = {
                    "intent": q["intent"],
                    "confidence": q["confidence"],
                    "slots": q["slots"],
                    "normalized": q.get("normalized_text"),
                }
            except Exception as exc:
                self.send(
                    {"type": "log", "msg": f"question model failed: {type(exc).__name__}: {exc}"}
                )
        self.send(
            {
                "type": "result",
                "cmd": cmd,
                "text": text,
                "stt_ms": stt_ms,
                "nlu_ms": round((self.clock() - t) * 1000, 1),
                "normalized": p.get("normalized_text"),
                "intent": p["intent"],
                "confidence": p["confidence"],
                "slots": p["slots"],
                "to_reading": meant,
                "two_commands": two,
                "ask": ask,
                # What the command grammar heard, when the question grammar's
                # transcript replaced it.
                "heard": heard,
                # The question model's reading (None without one).
                "question": question,
            }
        )

    # ------------------------------------------------------------------
    def run(self, stream: Any) -> None:
        """Serve requests from ``stream`` (a text file on a pipe) until quit/EOF."""
        fd = stream.fileno()
        buf = b""
        while True:
            # Poll fast while recording (drain the mic, time the release tail);
            # often enough between presses to keep the pre-roll; otherwise just
            # wait for the next request.
            if self.rec is not None:
                timeout = 0.01
            elif self.listening:
                timeout = LISTEN_POLL_SEC
            else:
                timeout = 0.5
            readable, _, _ = select.select([fd], [], [], timeout)
            if readable:
                chunk = os.read(fd, 65536)
                if not chunk:
                    return  # EOF: the plugin (or X-Plane) is gone
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            self.send(
                                {
                                    "type": "log",
                                    "level": "warning",
                                    "msg": f"bad request: {line[:200]!r}",
                                }
                            )
                            continue
                        if not self.handle(msg):
                            return
            self.tick()


def _dir_bytes(path: str) -> int:
    try:
        return sum(e.stat().st_size for e in os.scandir(path) if e.is_file())
    except OSError:
        return 0


def _protocol_writer() -> Callable[[dict[str, Any]], None]:
    """Reserve the real stdout for protocol lines; send stray prints (and
    C-level writes, e.g. library warnings) to stderr instead."""
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    lock = threading.Lock()  # the usage thread writes too

    def send(msg: dict[str, Any]) -> None:
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        with lock:
            proto.write(line)
            proto.flush()

    return send


def start_usage_reports(
    send: Callable[[dict[str, Any]], None], every: float, sampler: Any = None
) -> threading.Thread | None:
    """Report the Mac's usage every ``every`` seconds from a daemon thread
    (vimaan_nlu.usage), off X-Plane and off the request loop. 0: never.
    The parent process is X-Plane when the plugin starts the worker."""
    if every <= 0:
        return None
    sampler = sampler or UsageSampler({"xplane": os.getppid(), "worker": os.getpid()})

    def loop() -> None:
        sampler.sample()  # the first sample only sets the CPU baseline
        while True:
            time.sleep(every)
            try:
                send({"type": "usage", "every_sec": every, **sampler.sample()})
            except Exception as exc:  # never let a usage probe stop the worker
                send({"type": "log", "msg": f"usage sample failed: {type(exc).__name__}: {exc}"})

    thread = threading.Thread(target=loop, name="vimaan-usage", daemon=True)
    thread.start()
    return thread


def _make_stream_stt(args: argparse.Namespace) -> tuple[Any, Any, str, dict[str, Any]]:
    """(streaming recogniser or None, question recogniser or None, backend
    actually used, details to log).
    Vosk that can't load (package or model missing) falls back to Google so
    the copilot still works; the reason is reported in ``ready``."""
    if args.stt_backend != "vosk":
        return None, None, args.stt_backend, {}
    from vimaan_nlu.stt import DEFAULT_VOSK_MODEL, VoskStream
    from vimaan_nlu.stt_vocabulary import grammar, question_grammar

    model = args.vosk_model or DEFAULT_VOSK_MODEL
    t = time.perf_counter()
    try:
        import vosk

        vosk.SetLogLevel(-1)
        loaded = vosk.Model(model)
        words = None if args.no_grammar else grammar()
        stream = VoskStream(loaded, words=words)
    except Exception as exc:
        return (
            None,
            None,
            "google",
            {"fallback_from": "vosk", "error": f"{type(exc).__name__}: {exc}"},
        )
    try:
        question: Any = None if args.no_grammar else VoskStream(loaded, words=question_grammar())
    except Exception:
        question = None  # questions go without their own grammar; commands are unaffected
    return (
        stream,
        question,
        "vosk",
        {
            "model": model,
            "grammar_words": None if words is None else len(words),
            "question_grammar_words": None if question is None else len(question_grammar()),
            "load_ms": round((time.perf_counter() - t) * 1000, 1),
        },
    )


def _load_question_model(args: Any, model_loader: Any, predict: Callable[..., dict[str, Any]]):
    """(predict_question, loader, path) for the question model, or Nones
    when there is none: then the command model's own question intents answer,
    as before Tier 4's question model."""
    import torch
    from utils import get_latest_question_model_path

    path = args.question_model_path or get_latest_question_model_path()
    if not path:
        return None, None, None
    loader = model_loader(torch.device("cpu"))
    loader.load_all(
        model_path=path,
        quantize=not args.no_quantize and args.backend != "onnx",
        backend=args.backend,
    )

    def predict_question(text: str) -> dict[str, Any]:
        return predict(
            text,
            loader.model,
            loader.tokenizer,
            loader.device,
            loader.intent_map_rev,
            loader.slot_map_rev,
        )

    predict_question("what's our altitude")  # warm-up
    return predict_question, loader, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vimaan voice worker (spawned by the plugin)")
    parser.add_argument("--stt-backend", default="vosk", help="vosk | google | whisper | sphinx")
    parser.add_argument("--vosk-model", default=None, help="Vosk model directory")
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="vosk: decode with the full vocabulary instead of the command words",
    )
    parser.add_argument(
        "--save-audio-dir", default=None, help="save each command's audio here as WAV"
    )
    parser.add_argument("--audio-prefix", default="", help="file-name prefix for saved audio")
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument(
        "--preroll-ms",
        type=int,
        default=DEFAULT_PREROLL_MS,
        help="keep the mic open and start each recording with this much sound "
        "from before the key went down (0: open the mic on each press)",
    )
    parser.add_argument("--backend", default="torch", choices=("torch", "onnx"))
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--question-model-path",
        default=None,
        help="the question model (default: the newest qN; none: questions go to the command model)",
    )
    parser.add_argument(
        "--usage-every",
        type=float,
        default=USAGE_SEC,
        help="seconds between Mac usage reports (CPU, memory, GPU); 0: none",
    )
    parser.add_argument(
        "--audio-file",
        default=os.environ.get("VIMAAN_WORKER_AUDIO_FILE"),
        help="debug: 'record' this 16-bit mono WAV instead of the microphone",
    )
    args = parser.parse_args(argv)

    send = _protocol_writer()
    faulthandler.enable(sys.stderr, all_threads=True)
    t0 = time.perf_counter()
    try:
        import speech_recognition as sr
        import torch

        from vimaan_nlu.audio_capture import PushToTalkRecorder
        from vimaan_nlu.inference import predict
        from vimaan_nlu.model_loader import ModelLoader
        from vimaan_nlu.stt import transcribe

        import_ms = round((time.perf_counter() - t0) * 1000, 1)
        loader = ModelLoader(torch.device("cpu"))
        loaded = loader.load_all(
            model_path=args.model_path,
            quantize=not args.no_quantize and args.backend != "onnx",
            backend=args.backend,
        )

        def predict_text(text: str) -> dict[str, Any]:
            return predict(
                text,
                loader.model,
                loader.tokenizer,
                loader.device,
                loader.intent_map_rev,
                loader.slot_map_rev,
            )

        predict_text("set heading two seven zero")  # warm-up: first pass is slow
        predict_question, question_loader, question_path = _load_question_model(
            args, ModelLoader, predict
        )
        stream_stt, question_stt, stt_backend, stt_info = _make_stream_stt(args)
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = STT_TIMEOUT_SEC

        def transcribe_audio(raw: bytes, rate: int) -> str:
            return transcribe(
                recognizer,
                sr.AudioData(raw, rate, 2),
                stt_backend,
                whisper_model=args.whisper_model,
            )

        try:
            if args.audio_file:
                recorder: Any = WavRecorder(args.audio_file)
            else:
                recorder = PushToTalkRecorder(
                    max_sec=MAX_RECORD_SEC, preroll_sec=max(0, args.preroll_ms) / 1000
                )
            mic: dict[str, Any] = recorder.describe()
        except Exception as exc:
            recorder, mic = None, {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        send(
            {
                "type": "fatal",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    import transformers

    send(
        {
            "type": "ready",
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "import_ms": import_ms,
            "load_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model_path": loaded["model"]["model_path"],
            # What this model can say: the plugin self-tests only these, so a
            # model older than the command table isn't failed for commands
            # it was never trained on.
            "intents": sorted(loader.intent_map),
            "question_model_path": question_path,
            "question_intents": sorted(question_loader.intent_map) if question_loader else None,
            "quantized": loaded.get("quantized"),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "stt_backend": stt_backend,
            "stt": stt_info,
            "mic": mic,
            "audio_dir": args.save_audio_dir,
        }
    )
    if recorder is not None and hasattr(recorder, "listen"):
        recorder.listen()  # pre-roll: the mic stays open from now on
    start_usage_reports(send, args.usage_every)
    try:
        VoiceWorker(
            send,
            recorder,
            transcribe_audio,
            predict_text,
            stream_stt=stream_stt,
            question_stt=question_stt,
            predict_question=predict_question,
            audio_dir=args.save_audio_dir,
            audio_prefix=args.audio_prefix,
        ).run(sys.stdin)
    finally:
        if recorder is not None and hasattr(recorder, "close"):
            recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
