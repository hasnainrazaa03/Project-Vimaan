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


def make_listening_worker(tail=0.0):
    """A worker whose mic stays open between presses (300 ms pre-roll)."""
    w, out, sd, clock = make_worker(tail=tail)
    w.recorder = PushToTalkRecorder(sd=sd, preroll_sec=0.3)
    w.recorder.listen()
    w.recorder.wait_idle()
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


def test_the_open_mic_is_drained_between_presses():
    w, out, sd, _ = make_listening_worker()
    assert w.listening
    sd.streams[-1].feed(LOUD)
    w.tick()
    assert sd.streams[-1].read_available == 0 and out == []


def test_the_preroll_is_in_the_recording_but_not_in_the_tap_check():
    w, out, sd, _ = make_listening_worker()
    sd.streams[-1].feed(LOUD)  # a second of sound before the press
    w.tick()
    press_and_feed(w, sd, LOUD[:3200])  # a 0.1 s tap
    w.handle({"op": "stop", "cmd": "c1"})
    captured = [m for m in out if m["type"] == "captured"][-1]
    assert captured["audio_sec"] == 0.4 and captured["preroll_sec"] == 0.3
    assert out[-1]["outcome"] == "too_short"
    press_and_feed(w, sd, LOUD[:16000], cmd="c2")  # 0.5 s: a command
    w.handle({"op": "stop", "cmd": "c2"})
    assert out[-1]["type"] == "result"
    assert out[0]["type"] == "recording" and out[0]["open_ms"] == 0  # already open


def test_a_mic_lost_between_presses_is_logged_once():
    w, out, sd, _ = make_listening_worker()

    def unplugged(frames):
        raise OSError("device unavailable")

    sd.streams[-1].feed(LOUD[:3200])
    sd.streams[-1].read = unplugged
    w.tick()
    w.tick()
    w.recorder.wait_idle()
    logs = [m for m in out if m["type"] == "log"]
    assert len(logs) == 1 and "device unavailable" in logs[0]["msg"]
    assert not w.listening


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


def test_an_impossible_number_comes_with_its_to_reading():
    heard = "set com one standby two one two two decimal eight"
    table = {
        heard: ("set_com_standby", {"com_port": "1", "frequency": "2122.8"}),
        "set com one standby to one two two decimal eight": (
            "set_com_standby",
            {"com_port": "1", "frequency": "122.8"},
        ),
        "gear up": ("toggle_landing_gear", {"state": "up"}),
    }
    asked = []

    def predict(text):
        asked.append(text)
        intent, slots = table.get(text, ("None", {}))
        return {"intent": intent, "slots": slots, "confidence": 1.0, "normalized_text": text}

    out: list[dict] = []
    w = VoiceWorker(out.append, None, lambda r, s: "", predict)
    w.handle({"op": "text", "cmd": "c1", "text": heard})
    assert out[-1]["slots"]["frequency"] == "2122.8"  # what was heard, unchanged
    assert out[-1]["to_reading"]["slots"] == {"com_port": "1", "frequency": "122.8"}
    asked.clear()
    w.handle({"op": "text", "cmd": "c2", "text": "gear up"})
    assert out[-1]["to_reading"] is None and asked == ["gear up"]  # no extra passes


def test_a_failing_second_look_still_sends_the_command():
    calls = []

    def predict(text):
        calls.append(text)
        if len(calls) > 1:
            raise RuntimeError("model went away")
        return {
            "intent": "set_transponder_code",
            "slots": {"squawk": "21200"},
            "confidence": 1.0,
            "normalized_text": text,
        }

    out: list[dict] = []
    w = VoiceWorker(out.append, None, lambda r, s: "", predict)
    w.handle({"op": "text", "cmd": "c1", "text": "squawk two one two zero zero"})
    assert out[-2]["type"] == "log" and "to_reading failed" in out[-2]["msg"]
    assert out[-1]["type"] == "result" and out[-1]["to_reading"] is None
    assert out[-1]["slots"] == {"squawk": "21200"}


class Stream:
    """A stand-in for stt.VoskStream with a fixed transcript (or error)."""

    def __init__(self, text):
        self.text, self.fed, self.fed_bytes = text, b"", 0

    def begin(self):
        self.fed, self.fed_bytes = b"", 0

    def feed(self, pcm):
        self.fed += pcm
        self.fed_bytes += len(pcm)

    def finish(self):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text

    def partial(self):
        return ""


def question_worker(commands_hear, questions_hear):
    out: list[dict] = []

    def predict(text):
        return {"intent": "None", "slots": {}, "confidence": 1.0, "normalized_text": text}

    stream, question = Stream(commands_hear), Stream(questions_hear)
    w = VoiceWorker(
        out.append, None, lambda r, s: "", predict, stream_stt=stream, question_stt=question
    )
    return w, out, question


def test_a_question_is_decoded_again_with_the_question_grammar():
    w, out, question = question_worker("what the window", "what the wind")
    w._recognise("c1", LOUD, 16000, live=False)
    assert out[-1]["type"] == "result"
    assert out[-1]["text"] == "what the wind" and out[-1]["heard"] == "what the window"
    assert question.fed == LOUD


def test_a_command_is_not_decoded_again():
    w, out, question = question_worker("landing lights on", "landing light on")
    w._recognise("c1", LOUD, 16000, live=False)
    assert out[-1]["text"] == "landing lights on" and out[-1]["heard"] is None
    assert question.fed == b""


def test_the_question_key_always_uses_the_question_grammar():
    w, out, _ = question_worker("vertical flaps", "where our the flaps")
    w._recognise("c1", LOUD, 16000, live=False, ask=True)
    assert out[-1]["text"] == "where our the flaps" and out[-1]["ask"] is True
    # Nothing the command grammar knows: the question grammar may still hear it.
    w, out, _ = question_worker(
        UnknownValueError("no command words"), "how far two the next way point"
    )
    w._recognise("c1", LOUD, 16000, live=False, ask=True)
    assert out[-1]["type"] == "result" and out[-1]["text"] == "how far two the next way point"
    w, out, _ = question_worker(UnknownValueError("no command words"), "")
    w._recognise("c1", LOUD, 16000, live=False, ask=True)
    assert out[-1]["type"] == "error" and out[-1]["outcome"] == "not_understood"


def test_the_ask_flag_reaches_the_result():
    w, out, _, _ = make_worker()
    w.handle({"op": "text", "cmd": "c1", "text": "gear up", "ask": True})
    assert out[-1]["ask"] is True
    w.handle({"op": "text", "cmd": "c2", "text": "gear up"})
    assert out[-1]["ask"] is False


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

    def partial(self):
        return "gear" if self.fed_bytes else ""


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


def test_live_partials_sent_while_recording():
    w, out, sd, clock = make_worker()
    w.stream_stt = FakeStream()
    press_and_feed(w, sd, LOUD[:3200])
    w.tick()
    w.tick()  # same text again within the interval: not resent
    clock.now += 0.2
    w.tick()  # unchanged text: not resent either
    partials = [m for m in out if m["type"] == "partial"]
    assert partials == [{"type": "partial", "cmd": "c1", "text": "gear"}]


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
        # No pre-roll: the test mustn't open the real microphone.
        [sys.executable, "-u", "-m", "vimaan_nlu.voice_worker", "--preroll-ms", "0"],
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


# ---------------------------------------------------------------------------
# A recording from a file (the automated-test bridge, Phase A)
# ---------------------------------------------------------------------------
def write_wav(path, pcm, rate=16000, channels=1):
    import wave

    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm)
    return str(path)


def test_a_wav_is_recognised_like_a_recording(tmp_path):
    w, out, sd, clock = make_worker()
    w.audio_dir, w.audio_prefix = str(tmp_path / "audio"), "20261111_120000_"
    path = write_wav(tmp_path / "gear.wav", LOUD)
    w.handle({"op": "wav", "cmd": "c7", "path": path})
    assert types_of(out) == ["captured", "result"]
    captured, result = out
    assert captured["source_file"] == path and captured["audio_sec"] == 1.0
    assert captured["audio_file"].endswith("20261111_120000_c7.wav")  # kept for review
    assert result["cmd"] == "c7" and result["text"] == "gear up" and result["stt_ms"] is not None


def test_a_wav_goes_through_the_streaming_recogniser_from_the_start(tmp_path):
    w, out, sd, clock = make_worker(transcript=RuntimeError("batch STT must not run"))
    w.stream_stt = FakeStream("autopilot on")
    w.stream_stt.feed(b"left over from an earlier recording")
    w.handle({"op": "wav", "cmd": "c8", "path": write_wav(tmp_path / "a.wav", LOUD)})
    assert w.stream_stt.begun == 1 and w.stream_stt.fed == LOUD
    assert out[-1]["text"] == "autopilot on"


@pytest.mark.parametrize(
    ("make", "why"),
    [
        (lambda p: write_wav(p, LOUD, rate=44100), "sample rate 44100"),
        (lambda p: write_wav(p, LOUD, channels=2), "16-bit mono"),
        (lambda p: str(p) + ".missing", "No such file"),
    ],
)
def test_unusable_wavs_are_errors_not_crashes(tmp_path, make, why):
    w, out, sd, clock = make_worker()
    w.handle({"op": "wav", "cmd": "c9", "path": make(tmp_path / "x.wav")})
    (e,) = out
    assert e["type"] == "error" and e["outcome"] == "worker_error" and why in e["error"]


def test_short_and_silent_wavs_are_reported_as_such(tmp_path):
    w, out, sd, clock = make_worker()
    w.handle({"op": "wav", "cmd": "c1", "path": write_wav(tmp_path / "s.wav", LOUD[:3200])})
    w.handle({"op": "wav", "cmd": "c2", "path": write_wav(tmp_path / "q.wav", b"\0\0" * 16000)})
    assert [m.get("outcome") for m in out if m["type"] == "error"] == ["too_short", "silent_mic"]


def _p(intent, text, slots=None):
    return {"intent": intent, "slots": slots or {}, "confidence": 0.9, "normalized_text": text}


def test_the_question_models_reading_goes_with_the_result():
    out: list[dict] = []
    w = VoiceWorker(
        out.append,
        None,
        lambda r, s: "",
        lambda t: _p("toggle_lights", t, {"light": "landing"}),
        predict_question=lambda t: _p("ask_switch", t, {"light": "landing"}),
    )
    w.handle({"op": "text", "cmd": "c1", "text": "landing lights status"})
    assert out[-1]["intent"] == "toggle_lights"
    assert out[-1]["question"] == {
        "intent": "ask_switch",
        "slots": {"light": "landing"},
        "confidence": 0.9,
        "normalized": "landing lights status",
    }


def test_without_a_question_model_there_is_no_question_reading():
    out: list[dict] = []
    w = VoiceWorker(out.append, None, lambda r, s: "", lambda t: _p("None", t))
    w.handle({"op": "text", "cmd": "c1", "text": "hello"})
    assert out[-1]["question"] is None


def test_a_failing_question_model_leaves_the_command_reading():
    def broken(text):
        raise RuntimeError("question model went away")

    out: list[dict] = []
    w = VoiceWorker(
        out.append, None, lambda r, s: "", lambda t: _p("set_flaps", t), predict_question=broken
    )
    w.handle({"op": "text", "cmd": "c1", "text": "flaps five"})
    result = [m for m in out if m["type"] == "result"][-1]
    assert result["intent"] == "set_flaps" and result["question"] is None
    assert any(m["type"] == "log" and "question model failed" in m["msg"] for m in out)


def test_self_test_asks_each_model_its_own_probes():
    asked = []
    w = VoiceWorker(
        (out := []).append,
        None,
        lambda r, s: "",
        lambda t: asked.append(("command", t)) or _p("set_flaps", t),
        predict_question=lambda t: asked.append(("question", t)) or _p("ask_fuel", t),
    )
    w.handle(
        {"op": "selftest", "probes": [["flaps five", "set_flaps"], ["fuel check", "ask_fuel"]]}
    )
    assert asked == [("command", "flaps five"), ("question", "fuel check")]
    assert [r["intent"] for r in out[-1]["results"]] == ["set_flaps", "ask_fuel"]
