"""Tests for vimaan_nlu.voice_worker (protocol state machine) and
vimaan_nlu.worker_client (the plugin's non-blocking handle on the process)."""

import json
import os
import struct
import sys
import time

import pytest
from fake_xplane import make_sounddevice
from vimaan_nlu import voice_worker
from vimaan_nlu.audio_capture import PushToTalkRecorder
from vimaan_nlu.voice_worker import VoiceWorker
from vimaan_nlu.worker_client import WorkerClient

LOUD = struct.pack("<2h", 8000, -8000) * 8000  # 1 s of 16 kHz


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class UnknownValueError(Exception):
    pass


class RequestError(Exception):
    pass


def make_worker(transcript="gear up", tail=0.3):
    sd = make_sounddevice()
    out: list[dict] = []
    rec = PushToTalkRecorder(sd=sd)
    clock = Clock()

    def transcribe(raw, rate):
        if isinstance(transcript, Exception):
            raise transcript
        return transcript

    def predict(text):
        return {
            "intent": "toggle_landing_gear",
            "slots": {"state": "up"},
            "confidence": 0.98,
            "normalized_text": text,
        }

    w = VoiceWorker(out.append, rec, transcribe, predict, release_tail_sec=tail, clock=clock)
    return w, out, sd, clock


def press_and_feed(w, sd, audio=LOUD, cmd="c1"):
    w.handle({"op": "start", "cmd": cmd})
    w.recorder.wait_idle()
    w.tick()
    sd.streams[-1].feed(audio)


def types_of(out):
    return [m["type"] for m in out]


def test_full_command_flow():
    w, out, sd, clock = make_worker()
    press_and_feed(w, sd)
    w.handle({"op": "stop", "cmd": "c1"})
    w.tick()
    assert types_of(out) == ["recording"]  # still inside the release tail
    clock.now += 0.31
    w.tick()
    assert types_of(out) == ["recording", "captured", "result"]
    result = out[-1]
    assert result["cmd"] == "c1" and result["text"] == "gear up"
    assert result["intent"] == "toggle_landing_gear" and result["slots"] == {"state": "up"}
    assert result["stt_ms"] is not None and result["nlu_ms"] is not None
    assert w.idle


def test_release_tail_captures_late_audio():
    w, out, sd, clock = make_worker()
    press_and_feed(w, sd)
    w.handle({"op": "stop", "cmd": "c1"})
    sd.streams[-1].feed(LOUD[:3200])  # last word arrives after key-up
    clock.now += 0.31
    w.tick()
    assert [m for m in out if m["type"] == "captured"][0]["audio_sec"] == 1.1


def test_stop_for_other_command_ignored():
    w, out, sd, clock = make_worker()
    press_and_feed(w, sd)
    w.handle({"op": "stop", "cmd": "c999"})
    clock.now += 1
    w.tick()
    assert "captured" not in types_of(out)


def test_missed_key_up_auto_stops(monkeypatch):
    monkeypatch.setattr(voice_worker, "MAX_RECORD_SEC", -1.0)
    w, out, sd, clock = make_worker()
    press_and_feed(w, sd)
    w.tick()
    captured = [m for m in out if m["type"] == "captured"][0]
    assert captured["auto_stop"] is True


@pytest.mark.parametrize(
    ("exc", "outcome"),
    [
        (UnknownValueError(), "not_understood"),
        (RequestError("offline"), "stt_service_error"),
        (TimeoutError("read timed out"), "stt_service_error"),
        (RuntimeError("boom"), "worker_error"),
    ],
)
def test_stt_errors_classified(exc, outcome):
    w, out, sd, clock = make_worker(transcript=exc, tail=0)
    press_and_feed(w, sd)
    w.handle({"op": "stop", "cmd": "c1"})
    err = out[-1]
    assert err["type"] == "error" and err["outcome"] == outcome and err["stage"] == "stt"


def test_short_and_silent_audio():
    w, out, sd, _ = make_worker(tail=0)
    press_and_feed(w, sd, LOUD[:3200])
    w.handle({"op": "stop", "cmd": "c1"})
    assert out[-1]["outcome"] == "too_short"
    press_and_feed(w, sd, b"\x00\x00" * 16000, cmd="c2")
    w.handle({"op": "stop", "cmd": "c2"})
    assert out[-1]["outcome"] == "silent_mic"


def test_mic_open_failure_reported():
    w, out, sd, _ = make_worker()

    def broken(**kw):
        raise OSError("no input device")

    sd.RawInputStream = broken
    w.handle({"op": "start", "cmd": "c1"})
    w.recorder.wait_idle()
    w.tick()
    assert out[-1]["outcome"] == "mic_error" and "no input device" in out[-1]["error"]
    assert w.idle


def test_no_microphone_at_all():
    out: list[dict] = []
    w = VoiceWorker(out.append, None, lambda r, s: "", lambda t: {})
    w.handle({"op": "start", "cmd": "c1"})
    assert out[-1]["outcome"] == "mic_error"


def test_text_selftest_ping_quit():
    w, out, _, _ = make_worker()
    w.handle({"op": "text", "cmd": "c7", "text": "gear up"})
    assert out[-1]["type"] == "result" and out[-1]["stt_ms"] is None
    w.handle({"op": "selftest", "probes": [["gear up", "toggle_landing_gear"]]})
    assert out[-1]["results"][0]["intent"] == "toggle_landing_gear"
    w.handle({"op": "ping"})
    assert out[-1] == {"type": "pong"}
    assert w.handle({"op": "quit"}) is False


def test_run_serves_a_pipe_and_exits_on_eof():
    w, out, _, _ = make_worker()
    r, wfd = os.pipe()
    os.write(wfd, b'{"op": "ping"}\nnot json\n{"op": "text", "cmd": "c1", "text": "x"}\n')
    os.close(wfd)  # EOF after the requests
    with os.fdopen(r) as stream:
        w.run(stream)
    assert types_of(out) == ["pong", "log", "result"]


class FakeStream:
    """Stands in for stt.VoskStream."""

    def __init__(self, text="gear up"):
        self.text, self.fed, self.fed_bytes, self.begun = text, b"", 0, 0

    def begin(self):
        self.fed, self.fed_bytes, self.begun = b"", 0, self.begun + 1

    def feed(self, pcm):
        self.fed += pcm
        self.fed_bytes += len(pcm)

    def finish(self):
        return self.text


def test_streaming_stt_fed_while_recording(tmp_path):
    w, out, sd, clock = make_worker(transcript=RuntimeError("batch STT must not run"))
    w.stream_stt = FakeStream()
    press_and_feed(w, sd, LOUD[:16000])
    w.tick()
    assert w.stream_stt.fed_bytes == 16000  # decoded while the key is held
    w.handle({"op": "stop", "cmd": "c1"})
    sd.streams[-1].feed(LOUD[16000:32000])  # arrives during the release tail
    clock.now += 0.31
    w.tick()
    assert w.stream_stt.fed == LOUD[:32000]  # every byte exactly once
    assert out[-1]["type"] == "result" and out[-1]["text"] == "gear up"
    assert w.stream_stt.begun == 1


def test_audio_saved_per_command(tmp_path):
    import wave

    w, out, sd, clock = make_worker(tail=0)
    w.audio_dir, w.audio_prefix = str(tmp_path / "audio"), "20261030_120000_"
    press_and_feed(w, sd, LOUD[:32000])
    w.handle({"op": "stop", "cmd": "c1"})
    captured = next(m for m in out if m["type"] == "captured")
    path = captured["audio_file"]
    assert path == str(tmp_path / "audio" / "20261030_120000_c1.wav")
    with wave.open(path) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2)
        assert f.readframes(10**6) == LOUD[:32000]


def test_audio_not_saved_when_off_or_full(tmp_path, monkeypatch):
    w, out, sd, clock = make_worker(tail=0)
    press_and_feed(w, sd)
    w.handle({"op": "stop", "cmd": "c1"})
    assert next(m for m in out if m["type"] == "captured")["audio_file"] is None
    monkeypatch.setattr(voice_worker, "MAX_AUDIO_DIR_BYTES", 1000)
    w.audio_dir = str(tmp_path)
    out.clear()
    press_and_feed(w, sd, cmd="c2")
    w.handle({"op": "stop", "cmd": "c2"})
    assert next(m for m in out if m["type"] == "captured")["audio_file"] is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# WorkerClient against a real (tiny, torch-free) child process
# ---------------------------------------------------------------------------
ECHO_WORKER = r"""
import json, sys
print("starting up", file=sys.stderr, flush=True)
print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("op") == "quit":
        break
    if msg.get("op") == "crash":
        print("Fatal Python error: Segmentation fault", file=sys.stderr, flush=True)
        sys.exit(-11 % 256)
    print(json.dumps({"type": "echo", "msg": msg}), flush=True)
"""


@pytest.fixture
def echo_client(tmp_path):
    script = tmp_path / "echo_worker.py"
    script.write_text(ECHO_WORKER)
    client = WorkerClient([sys.executable, "-u", str(script)], tmp_path / "worker.log")
    client.start()
    yield client
    client.stop()


def poll_until(client, n, timeout=5.0):
    got: list[dict] = []
    deadline = time.time() + timeout
    while len(got) < n and time.time() < deadline:
        got += client.poll()
        time.sleep(0.01)
    return got


def test_client_poll_never_blocks(echo_client):
    t = time.perf_counter()
    echo_client.poll()  # may or may not have "ready" yet; must return at once
    assert time.perf_counter() - t < 0.05


def test_client_round_trip(echo_client):
    assert poll_until(echo_client, 1)[0] == {"type": "ready"}
    assert echo_client.send({"op": "start", "cmd": "c1"})
    assert poll_until(echo_client, 1)[0]["msg"] == {"op": "start", "cmd": "c1"}
    assert echo_client.alive


def test_client_detects_crash_and_keeps_log(echo_client):
    poll_until(echo_client, 1)
    echo_client.send({"op": "crash"})
    deadline = time.time() + 5
    while echo_client.alive and time.time() < deadline:
        time.sleep(0.01)
    assert not echo_client.alive and echo_client.returncode not in (None, 0)
    assert any("Segmentation fault" in line for line in echo_client.log_tail())
    assert echo_client.send({"op": "ping"}) is False


def test_client_stop_ends_process(echo_client):
    echo_client.stop()
    assert not echo_client.alive


# ---------------------------------------------------------------------------
# The real worker process (needs torch + the trained model)
# ---------------------------------------------------------------------------
@pytest.mark.requires_torch
def test_real_worker_process(tmp_path):
    ml = os.path.join(os.path.dirname(__file__), "..", "ML")
    if not os.path.isdir(os.path.join(ml, "models", "vimaan_nlu_model_best")):
        pytest.skip("no trained model checkout")
    env = dict(os.environ, PYTHONPATH=os.path.abspath(ml), HF_HUB_OFFLINE="1")
    client = WorkerClient(
        [sys.executable, "-u", "-m", "vimaan_nlu.voice_worker"],
        tmp_path / "worker.log",
        env=env,
        cwd=os.path.abspath(ml),
    )
    client.start()
    try:
        ready = poll_until(client, 1, timeout=60)
        assert ready and ready[0]["type"] == "ready", client.log_tail()
        client.send({"op": "text", "cmd": "c1", "text": "Set altitude to 7,500."})
        result = poll_until(client, 1, timeout=10)[0]
        assert result["intent"] == "set_autopilot_altitude"
        assert result["slots"]["altitude"] == "7500"
        assert json.dumps(result)  # protocol stays JSON-clean
    finally:
        client.stop()
