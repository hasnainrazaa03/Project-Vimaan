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
    start {cmd}          open the mic and record
    stop {cmd}           key released: keep recording a short tail, then
                         transcribe + predict and reply ``result``
    text {cmd, text}     NLU only (no audio)
    selftest {probes}    NLU on [text, expected] pairs; reply ``selftest``
    ping                 reply ``pong``
    quit                 exit
Replies (``type``):
    ready, fatal, recording, captured, result, error, selftest, pong, log

EOF on stdin (the plugin or X-Plane went away) makes the worker exit.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import select
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

from vimaan_nlu.audio_health import is_silent, rms_from_pcm16

MAX_RECORD_SEC = 12.0  # recording is cut off after this even if still held
MIN_RECORD_SEC = 0.3  # shorter than this is an accidental tap, not a command
# Google STT has no timeout by default: one stalled request would block the
# worker forever and every later command would time out. Fail it instead.
STT_TIMEOUT_SEC = 8.0
# Keep recording this long after the key is released: PortAudio hands audio
# over ~100-160 ms late, and pilots let go as they finish the last word. Without
# it, in-sim Google heard "gear" / "parking brake" with the "up" / "on" cut off.
RELEASE_TAIL_SEC = 0.3


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

    def describe(self) -> dict[str, Any]:
        return {"using": f"WAV file {self.path}", "sample_rate": self.sample_rate}

    def start(self) -> None:
        self._t0, self._taken, self.is_open = time.perf_counter(), 0, True

    def elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.perf_counter() - self._t0

    def poll(self) -> int:
        if self._t0 is None:
            return 0
        due = min(len(self._pcm) // 2, int(self.elapsed() * self.sample_rate))
        n, self._taken = due - self._taken, due
        return n

    def stop(self) -> bytes:
        self.poll()
        raw = self._pcm[: self._taken * 2]
        self._t0, self.is_open = None, False
        return raw

    def wait_idle(self, timeout: float = 2.0) -> None:
        pass

    def seconds(self, raw: bytes) -> float:
        return len(raw) / (2 * self.sample_rate)


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
    ) -> None:
        self.send = send
        self.recorder = recorder
        self.transcribe_audio = transcribe_audio
        self.predict_text = predict_text
        self.release_tail_sec = release_tail_sec
        self.clock = clock
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
            self.rec = {
                "cmd": cmd,
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
            self._predict_and_reply(cmd, msg.get("text", ""), stt_ms=None, audio=None)
        elif op == "selftest":
            results = []
            for text, expected in msg.get("probes", []):
                try:
                    p = self.predict_text(text)
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
        """Advance the recording in progress. Call often (every ~10 ms)."""
        rec = self.rec
        if rec is None:
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

    # ------------------------------------------------------------------
    def _finish(self, auto_stop: bool = False) -> None:
        rec, self.rec = self.rec, None
        if rec is None:
            return
        cmd = rec["cmd"]
        raw = self.recorder.stop()
        audio_sec = self.recorder.seconds(raw)
        rms = rms_from_pcm16(raw)
        self.send(
            {
                "type": "captured",
                "cmd": cmd,
                "audio_sec": round(audio_sec, 2),
                "rms": round(rms, 1),
                "overflows": self.recorder.overflows,
                "truncated": self.recorder.truncated,
                "auto_stop": auto_stop,
            }
        )
        if audio_sec < MIN_RECORD_SEC:
            self.send({"type": "error", "cmd": cmd, "outcome": "too_short"})
            return
        if is_silent(rms):
            self.send({"type": "error", "cmd": cmd, "outcome": "silent_mic"})
            return
        t = self.clock()
        try:
            text = self.transcribe_audio(raw, self.recorder.sample_rate)
        except Exception as exc:
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
            return
        self._predict_and_reply(
            cmd, text, stt_ms=round((self.clock() - t) * 1000, 1), audio=audio_sec
        )

    def _predict_and_reply(
        self, cmd: Any, text: str, stt_ms: float | None, audio: float | None
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
            }
        )

    # ------------------------------------------------------------------
    def run(self, stream: Any) -> None:
        """Serve requests from ``stream`` (a text file on a pipe) until quit/EOF."""
        fd = stream.fileno()
        buf = b""
        while True:
            # Poll fast while recording (drain the mic, time the release tail);
            # otherwise just wait for the next request.
            timeout = 0.01 if self.rec is not None else 0.5
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


def _protocol_writer() -> Callable[[dict[str, Any]], None]:
    """Reserve the real stdout for protocol lines; send stray prints (and
    C-level writes, e.g. library warnings) to stderr instead."""
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(msg: dict[str, Any]) -> None:
        proto.write(json.dumps(msg, ensure_ascii=False) + "\n")
        proto.flush()

    return send


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vimaan voice worker (spawned by the plugin)")
    parser.add_argument("--stt-backend", default="google")
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--backend", default="torch", choices=("torch", "onnx"))
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--model-path", default=None)
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
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = STT_TIMEOUT_SEC

        def transcribe_audio(raw: bytes, rate: int) -> str:
            return transcribe(
                recognizer,
                sr.AudioData(raw, rate, 2),
                args.stt_backend,
                whisper_model=args.whisper_model,
            )

        try:
            if args.audio_file:
                recorder: Any = WavRecorder(args.audio_file)
            else:
                recorder = PushToTalkRecorder(max_sec=MAX_RECORD_SEC)
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
            "quantized": loaded.get("quantized"),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "stt_backend": args.stt_backend,
            "mic": mic,
        }
    )
    VoiceWorker(send, recorder, transcribe_audio, predict_text).run(sys.stdin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
