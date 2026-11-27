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

Pre-roll (``preroll_sec`` > 0): the stream stays open between presses and
:meth:`poll` keeps only the last ``preroll_sec`` of sound, in memory. A press
starts with that sound, so the first word isn't cut by the ~85 ms open or by
speaking as the key goes down. Nothing between presses is kept longer or
saved. With 0 the stream opens on each press, as before.

The wake word (phase L.5, ``hear_between``): the sound between presses is
also handed to the owner (:meth:`take_heard`), which listens for "hey
copilot" in it. It is held in memory only until taken (at most
``HEARD_MAX_SEC``), and never saved.

``sd`` is injectable so this is testable without audio hardware.
"""

from __future__ import annotations

import collections
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
# Sound heard between presses waits at most this long to be taken.
HEARD_MAX_SEC = 2.0


class PushToTalkRecorder:
    def __init__(
        self,
        sd: Any = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_sec: float = 12.0,
        device: int | str | None = None,
        preroll_sec: float = 0.0,
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
        self._taken = 0  # chunks already handed out by take_new()
        self._stream: Any = None
        self._recording = False
        self._started_at: float | None = None
        self._workers: list[threading.Thread] = []
        self.truncated = False
        self.overflows = 0
        self.error: BaseException | None = None  # set if opening the mic failed
        self.preroll_sec = max(0.0, float(preroll_sec))
        self._ring: collections.deque[bytes] = collections.deque()  # sound before a press
        self._ring_frames = 0
        self.preroll_frames = 0  # how much of the recording came from before the press
        self._closed = False
        # The mic failed between presses (device gone). The next press opens
        # it again, and it stays open from then on.
        self.listen_error: BaseException | None = None
        # Phase L.5: the sound between presses, for the wake word.
        self.hear_between = False
        self._heard: collections.deque[bytes] = collections.deque()
        self._heard_frames = 0
        self._heard_gap = False  # some of it was lost (an overflow, a press)

    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Input devices + which one we'll use, for the startup log."""
        devices = self._sd.query_devices()
        inputs = [d["name"] for d in devices if d.get("max_input_channels", 0) > 0]
        default = self._sd.query_devices(self.device, "input")
        return {
            "input_devices": inputs,
            "using": default["name"],
            "sample_rate": self.sample_rate,
            "preroll_ms": round(self.preroll_sec * 1000),
        }

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def keeps_open(self) -> bool:
        """The stream stays open between presses (pre-roll on)."""
        return self.preroll_sec > 0 and not self._closed

    @property
    def is_listening(self) -> bool:
        """Open between presses, keeping the last ``preroll_sec`` of sound.
        :meth:`poll` must then be called at least every 0.2 s."""
        return self.keeps_open and self._stream is not None and not self._recording

    @property
    def is_open(self) -> bool:
        """True once the background open has finished and audio is flowing."""
        return self._stream is not None

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

    @classmethod
    def _close_quietly(cls, stream: Any) -> None:
        try:
            cls._close(stream)
        except Exception:
            pass  # the device is already gone

    def elapsed(self) -> float:
        return 0.0 if self._started_at is None else time.perf_counter() - self._started_at

    # ------------------------------------------------------------------
    def listen(self) -> None:
        """Open the mic now and keep it open between presses (pre-roll).
        Does nothing when pre-roll is off or the mic is already open."""
        if self.keeps_open and self._stream is None and not self.busy:
            self._run_async(self._open, "open")

    def start(self) -> None:
        """Begin a recording. With the mic open between presses, it starts
        with the last ``preroll_sec`` of sound; otherwise the stream opens in
        the background (~85 ms) and audio is captured from then."""
        if self._recording:
            return
        self.poll()  # the newest sound into the pre-roll
        self._drop_heard()  # a press: what the listener had is cut off
        with self._lock:
            self._chunks, self._frames, self._taken = [], 0, 0
            preroll = self._take_preroll()
            if preroll:
                self._chunks.append(preroll)
                self._frames = len(preroll) // SAMPLE_WIDTH
            self.preroll_frames = self._frames
        self.truncated = False
        self.overflows = 0
        self.error = None
        self._recording = True
        self._started_at = time.perf_counter()
        if self._stream is None:
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
            if self._recording:
                self.error = exc
            else:
                self.listen_error = exc
            return
        with self._lock:
            if (self._recording or self.keeps_open) and self._stream is None:
                self._stream = stream
                return
        self._close(stream)  # released before it finished opening

    def _keep_preroll(self, data: bytes) -> None:
        """Between presses: keep only the last ``preroll_sec`` of sound."""
        self._ring.append(data)
        self._ring_frames += len(data) // SAMPLE_WIDTH
        keep = int(self.preroll_sec * self.sample_rate)
        while self._ring and self._ring_frames - len(self._ring[0]) // SAMPLE_WIDTH >= keep:
            self._ring_frames -= len(self._ring.popleft()) // SAMPLE_WIDTH

    def _keep_heard(self, data: bytes) -> None:
        self._heard.append(data)
        self._heard_frames += len(data) // SAMPLE_WIDTH
        keep = int(HEARD_MAX_SEC * self.sample_rate)
        if self._heard_frames > keep:  # not taken in time: only the newest waits
            newest = b"".join(self._heard)[-keep * SAMPLE_WIDTH :]
            self._heard.clear()
            self._heard.append(newest)
            self._heard_frames = keep
            self._heard_gap = True

    def _drop_heard(self) -> None:
        if self._heard or self.hear_between:
            self._heard_gap = True
        self._heard.clear()
        self._heard_frames = 0

    def take_heard(self) -> tuple[bytes, bool]:
        """The sound heard between presses since the last call, and whether
        any was lost before it (an overflow, a press, or not taken in time):
        the listener then starts afresh."""
        raw, gap = b"".join(self._heard), self._heard_gap
        self._heard.clear()
        self._heard_frames = 0
        self._heard_gap = False
        return raw, gap

    def _take_preroll(self) -> bytes:
        keep = int(self.preroll_sec * self.sample_rate) * SAMPLE_WIDTH
        raw = b"".join(self._ring)[-keep:] if keep and self._ring else b""
        self._ring.clear()
        self._ring_frames = 0
        return raw

    def _drop_stream(self, exc: BaseException) -> None:
        """The stream failed (device unplugged): let it go and say so."""
        with self._lock:
            stream, self._stream = self._stream, None
        self._ring.clear()
        self._ring_frames = 0
        self._drop_heard()
        if self._recording:
            self.error = exc
        else:
            self.listen_error = exc
        if stream is not None:
            self._run_async(lambda: self._close_quietly(stream), "close")

    def poll(self) -> int:
        """Move whatever PortAudio has buffered into memory; returns frames read.

        Call often while recording (every frame from the flight loop). Never
        blocks: it only reads what is already available.
        """
        stream = self._stream
        if stream is None:
            return 0
        try:
            available = stream.read_available
            if available <= 0:
                return 0
            data, overflowed = stream.read(available)
        except Exception as exc:  # the device went away
            self._drop_stream(exc)
            return 0
        if not self._recording:
            if overflowed:
                # Not drained in time: this is older sound, with a gap after
                # it. Better no pre-roll than stale sound before the press.
                self._ring.clear()
                self._ring_frames = 0
                self._drop_heard()
            elif self.keeps_open:
                self._keep_preroll(bytes(data))
                if self.hear_between:
                    self._keep_heard(bytes(data))
            return available
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

    def take_new(self) -> bytes:
        """PCM captured since the previous call (for streaming recognition).
        Doesn't affect what :meth:`stop` returns."""
        with self._lock:
            new = self._chunks[self._taken :]
            self._taken = len(self._chunks)
        return b"".join(new)

    def stop(self) -> bytes:
        """End the recording and return the captured 16-bit mono PCM.

        Drains what PortAudio has buffered. With pre-roll the stream stays
        open; otherwise it closes on a background thread so the caller
        (X-Plane's main thread) doesn't stall.
        """
        if self._recording:
            self.poll()
        self._recording = False
        with self._lock:
            if self.keeps_open:
                stream = None
            else:
                stream, self._stream = self._stream, None
            raw = b"".join(self._chunks)
            self._chunks, self._frames, self._taken = [], 0, 0
        if stream is not None:
            self._run_async(lambda: self._close(stream), "close")
        return raw

    def close(self) -> None:
        """Release the mic for good (the worker is exiting)."""
        self._closed = True
        self._recording = False
        with self._lock:
            stream, self._stream = self._stream, None
        self._ring.clear()
        self._ring_frames = 0
        if stream is not None:
            self._close_quietly(stream)

    def seconds(self, raw: bytes) -> float:
        return len(raw) / (SAMPLE_WIDTH * self.sample_rate)
