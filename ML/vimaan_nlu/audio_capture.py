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

``sd`` is injectable so this is testable without audio hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Any

DEFAULT_SAMPLE_RATE = 16_000  # what every STT backend we support wants
SAMPLE_WIDTH = 2  # int16
# PortAudio sizes its blocking-mode ring buffer from the suggested latency.
# Generous, so a long sim frame (scenery load) between polls loses nothing.
RING_BUFFER_SEC = 1.0


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
        self._started_at: float | None = None
        self.truncated = False
        self.overflows = 0

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Input devices + which one we'll use, for the startup log."""
        devices = self._sd.query_devices()
        inputs = [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]
        default = self._sd.query_devices(self.device, "input")
        return {"input_devices": inputs, "using": default["name"], "sample_rate": self.sample_rate}

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def elapsed(self) -> float:
        return 0.0 if self._started_at is None else time.perf_counter() - self._started_at

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._chunks, self._frames = [], 0
        self.truncated = False
        self.overflows = 0
        stream = self._sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
            latency=RING_BUFFER_SEC,
        )
        stream.start()
        self._stream = stream
        self._started_at = time.perf_counter()

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
        """Drain, stop recording and return the captured 16-bit mono PCM."""
        try:
            self.poll()
        finally:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        with self._lock:
            raw = b"".join(self._chunks)
            self._chunks, self._frames = [], 0
        return raw

    def seconds(self, raw: bytes) -> float:
        return len(raw) / (SAMPLE_WIDTH * self.sample_rate)
