"""Hold-to-talk microphone capture via ``sounddevice``.

Replaces SpeechRecognition's ``Microphone`` (which needs PyAudio — no macOS
wheel, must be compiled against XPPython3's bundled Python). ``sounddevice``
wheels ship PortAudio, so a plain ``pip install sounddevice`` is enough.

Recording runs from :meth:`start` (button down) to :meth:`stop` (button up):
the pilot decides where the phrase ends, so no voice-activity guessing.
Output is 16-bit mono PCM, the format ``speech_recognition.AudioData`` and the
STT backends expect.

Why no audio callback: inside X-Plane, background Python threads barely get
the GIL (measured in-sim: a 9 s press captured 0.03 s through a Python
callback, with no overflow reported). So the stream runs in PortAudio's
*blocking* mode — PortAudio's own C thread fills a ring buffer, which needs no
GIL — and the owner calls :meth:`poll` from the thread that does hold the GIL
(the plugin's flight loop, every frame while recording) to drain it.

Opening and closing a CoreAudio stream take ~85 ms and ~118 ms (a visible
frame hitch per command in-sim), so both run on a short-lived thread;
:attr:`busy` tells the owner to give that thread GIL time.

``sd`` is injectable so this is testable without audio hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Any

DEFAULT_SAMPLE_RATE = 16_000  # what every STT backend we support wants
SAMPLE_WIDTH = 2  # int16
# PortAudio sizes its blocking-mode ring buffer from the suggested latency,
# and the same value sets how late audio is handed over. Measured on an M5
# MacBook mic: 1.0 -> 230-360 ms late (cut the last word off every command);
# 0.1 -> 22 ms late but overflows if polls are >0.25 s apart; 0.2 survives
# 0.25 s gaps. The flight loop polls every frame while recording.
RING_BUFFER_SEC = 0.2


class PushToTalkRecorder:
    def __init__(
        self,
        sd: Any = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_sec: float = 12.0,
        device: int | str | None = None,
    ) -> None:
        if sd is None:
            import sounddevice  # deferred: optional at import time

            sd = sounddevice
        self._sd = sd
        self.sample_rate = sample_rate
        self.max_sec = max_sec
        self.device = device
        self._lock = threading.Lock()
        self._chunks: list[bytes] = []
        self._frames = 0
        self._stream: Any = None
        self._recording = False
        self._started_at: float | None = None
        self._workers: list[threading.Thread] = []
        self.truncated = False
        self.overflows = 0
        self.error: BaseException | None = None  # set if opening the mic failed

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Input devices + which one we'll use, for the startup log."""
        devices = self._sd.query_devices()
        inputs = [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]
        default = self._sd.query_devices(self.device, "input")
        return {"input_devices": inputs, "using": default["name"], "sample_rate": self.sample_rate}

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def busy(self) -> bool:
        """True while the mic is being opened or closed in the background."""
        self._workers = [t for t in self._workers if t.is_alive()]
        return bool(self._workers)

    def wait_idle(self, timeout: float = 2.0) -> None:
        """Block until background open/close work is done (shutdown, tests)."""
        for t in list(self._workers):
            t.join(timeout)

    def _run_async(self, fn, name: str) -> None:
        t = threading.Thread(target=fn, name=f"mic-{name}", daemon=True)
        self._workers.append(t)
        t.start()

    @staticmethod
    def _close(stream: Any) -> None:
        try:
            stream.stop()
        finally:
            stream.close()

    def elapsed(self) -> float:
        return 0.0 if self._started_at is None else time.perf_counter() - self._started_at

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Begin a recording. The stream opens in the background (~85 ms);
        audio is captured from the moment it is open."""
        if self._recording:
            return
        with self._lock:
            self._chunks, self._frames = [], 0
        self.truncated = False
        self.overflows = 0
        self.error = None
        self._recording = True
        self._started_at = time.perf_counter()
        self._run_async(self._open, "open")

    def _open(self) -> None:
        try:
            stream = self._sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
                latency=RING_BUFFER_SEC,
            )
            stream.start()
        except BaseException as exc:  # reported to the owner via .error
            self.error = exc
            return
        with self._lock:
            if self._recording and self._stream is None:
                self._stream = stream
                return
        self._close(stream)  # released before it finished opening

    def poll(self) -> int:
        """Move whatever PortAudio has buffered into memory; returns frames read.

        Call often while recording (every frame from the flight loop). Never
        blocks: it only reads what is already available.
        """
        stream = self._stream
        if stream is None:
            return 0
        available = stream.read_available
        if available <= 0:
            return 0
        data, overflowed = stream.read(available)
        if overflowed:
            self.overflows += 1
        with self._lock:
            room = int(self.max_sec * self.sample_rate) - self._frames
            if room <= 0:
                self.truncated = True
                return available
            if available > room:
                self.truncated = True
            take = min(available, room)
            self._chunks.append(bytes(data)[: take * SAMPLE_WIDTH])
            self._frames += take
        return available

    def stop(self) -> bytes:
        """End the recording and return the captured 16-bit mono PCM.

        Drains what PortAudio has buffered, then closes the stream on a
        background thread so the caller (X-Plane's main thread) doesn't stall.
        """
        self._recording = False
        self.poll()
        with self._lock:
            stream, self._stream = self._stream, None
            raw = b"".join(self._chunks)
            self._chunks, self._frames = [], 0
        if stream is not None:
            self._run_async(lambda: self._close(stream), "close")
        return raw

    def seconds(self, raw: bytes) -> float:
        return len(raw) / (SAMPLE_WIDTH * self.sample_rate)
