"""Phase L.5: the wake word ("hey copilot", vimaan_nlu.wake_word), in the
spotter, the voice worker and the recorder. Vosk is faked: each result is
scripted at a time, with its words' times and confidences."""

from __future__ import annotations

import json
import struct
import wave

import pytest
from fake_xplane import make_sounddevice
from vimaan_nlu import voice_worker
from vimaan_nlu import wake_word as ww
from vimaan_nlu.audio_capture import PushToTalkRecorder
from vimaan_nlu.voice_worker import VoiceWorker

RATE = 16_000


def quiet(sec, level=5):
    return struct.pack("<h", level) * int(RATE * sec)


def speech(sec, level=6000):
    return struct.pack("<2h", level, -level) * int(RATE * sec / 2)


def w(word, start, end, conf=1.0):
    return {"word": word, "start": start, "end": end, "conf": conf}


WAKE = [w("hey", 1.0, 1.2), w("copilot", 1.2, 1.6)]


class FakeRec:
    """Vosk's recogniser: each result is given once the stream reaches its time
    (counted from this recogniser's start). ``results`` is shared by the
    recognisers a spotter makes: a result heard once isn't heard again after a
    reset, as real sound wouldn't be."""

    def __init__(self, results):
        self.results = results
        self.fed = 0
        self.pending = None

    def AcceptWaveform(self, pcm):
        self.fed += len(pcm) // 2
        if self.results and self.fed >= self.results[0][0] * RATE:
            self.pending = self.results.pop(0)[1]
            return True
        return False

    def Result(self):
        return json.dumps({"result": self.pending or []})


def spotter(*results):
    shared = sorted(results, key=lambda r: r[0])
    return ww.WakeSpotter(lambda: FakeRec(shared), RATE)


def run(s, *pieces, step=0.05):
    """Feed in 50 ms polls, as the worker does; (seconds, event) in order."""
    pcm = b"".join(pieces)
    n = int(step * RATE) * 2
    out = []
    for i in range(0, len(pcm), n):
        for ev in s.feed(pcm[i : i + n]):
            out.append((round((i + n) / (2 * RATE), 2), ev))
    return out


def test_in_one_breath_the_phrase_comes_with_it():
    # No pause after "copilot": decoded with the phrase, which comes off after.
    s = spotter((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    events = run(s, quiet(1.0), speech(1.4), quiet(2.0))
    assert events[1][1].with_phrase is True


def test_one_breath():
    # "hey copilot gear down": Vosk's result, at the pause, holds the command too.
    s = spotter((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    events = run(s, quiet(1.0), speech(1.4), quiet(2.0))
    assert [type(e).__name__ for _, e in events] == ["Wake", "Command"]
    (t_wake, wake), (t_cmd, cmd) = events
    assert wake == ww.Wake(1.0, one_breath=True) and t_wake == 2.6
    # From the wake phrase (1.0 s) to the last speech (2.4 s) + 0.3 s, handed
    # on 0.7 s after it: the phrase is decoded with it and taken off after.
    assert cmd.seconds == pytest.approx(1.7, abs=0.1) and not cmd.cut_off
    assert t_cmd == pytest.approx(3.1, abs=0.15)


def test_a_pause_then_the_command():
    s = spotter((2.4, WAKE))
    events = run(s, quiet(1.0), speech(0.6), quiet(1.0), speech(1.5), quiet(2.0))
    assert [type(e).__name__ for _, e in events] == ["Wake", "Command"]
    assert events[0][1].one_breath is False
    # After the pause the command comes alone: from 0.5 s before its speech
    # (2.6 s) to the end of it (4.1 s) + 0.3 s.
    cmd = events[1][1]
    assert cmd.with_phrase is False and cmd.seconds == pytest.approx(2.3, abs=0.1)


def test_nothing_said_after_it_is_no_command():
    s = spotter((2.4, WAKE))
    events = run(s, quiet(1.0), speech(0.6), quiet(6.0))
    assert [e for _, e in events][1:] == [ww.NO_COMMAND]
    assert events[1][0] == pytest.approx(2.4 + ww.COMMAND_START_SEC, abs=0.06)
    assert not s.in_command


@pytest.mark.parametrize(
    "words",
    [
        [w("[unk]", 1.0, 1.2), w("copilot", 1.2, 1.6)],  # "autopilot on"
        [w("hey", 1.0, 1.2), w("copilot", 1.2, 1.6, conf=0.87)],  # L.5.1's false ones
        [w("copilot", 1.0, 1.2), w("hey", 1.2, 1.6)],
        [w("hey", 1.0, 1.2), w("[unk]", 1.2, 1.4), w("copilot", 1.4, 1.6)],
    ],
)
def test_not_a_wake(words):
    s = spotter((2.4, words))
    assert run(s, quiet(1.0), speech(1.0), quiet(4.0)) == []


def test_a_long_command_is_cut_off():
    s = spotter((2.4, WAKE))
    events = run(s, quiet(1.0), speech(0.6), quiet(0.8), speech(10.0), quiet(1.0))
    cmd = events[1][1]
    # 8 s after "copilot" ends (1.6 s), from 0.5 s before the speech after the pause (2.4 s).
    assert cmd.cut_off and cmd.seconds == pytest.approx(1.6 + ww.COMMAND_MAX_SEC - 1.9, abs=0.06)


def test_speech_is_loud_against_the_background():
    # A noisy cockpit (2,000 before the phrase): the same noise after the wake
    # is not a command; a voice well over it is.
    s = spotter((4.4, [w("hey", 3.0, 3.2), w("copilot", 3.2, 3.6)]))
    events = run(s, quiet(3.0, level=2000), speech(0.6), quiet(5.0, level=2000))
    assert [e for _, e in events][1:] == [ww.NO_COMMAND]
    s = spotter((4.4, [w("hey", 3.0, 3.2), w("copilot", 3.2, 3.6)]))
    events = run(s, quiet(3.0, level=2000), speech(0.6), speech(1.0, 12_000), quiet(2.0, 2000))
    assert isinstance(events[1][1], ww.Command)


def test_reset_forgets_a_wake_in_hand():
    s = spotter((2.4, WAKE))
    run(s, quiet(1.0), speech(0.6), quiet(1.0))
    assert s.in_command
    s.reset()
    assert not s.in_command and run(s, quiet(5.0)) == []


def test_a_clatter_in_the_background_is_not_speech():
    # Loudness over 100 ms: 20 ms clicks, each over the speech level on its
    # own (1,200 against 300), don't keep the command going; it ends 0.7 s
    # after the voice.
    s = spotter((2.4, WAKE))
    knock = speech(0.02, 1200)
    gap = quiet(0.2)
    events = run(s, quiet(1.0), speech(0.6), quiet(1.0), speech(1.0), (knock + gap) * 10)
    cmd = events[1][1]
    assert isinstance(cmd, ww.Command) and not cmd.cut_off
    assert cmd.seconds == pytest.approx(3.6 + 0.3 - 2.1, abs=0.15)  # from 0.5 s before 2.6 s


def test_the_phrase_comes_off_the_front():
    assert ww.strip_wake("hey copilot gear down") == "gear down"
    assert ww.strip_wake("copilot set heading two seven zero") == "set heading two seven zero"
    assert ww.strip_wake("hey copilot") == ""
    assert ww.strip_wake("gear down") == "gear down"
    assert ww.strip_wake("flaps up hey copilot") == "flaps up hey copilot"
    assert ww.wake_grammar(["a", "hey", "[unk]"]) == ["a", "hey", "[unk]", "copilot"]


def test_find_wake():
    assert ww.find_wake([*WAKE, w("[unk]", 1.6, 2.0)]) == (0, 1.0)
    assert ww.find_wake([w("[unk]", 0, 1), *WAKE]) == (1, 1.0)
    assert ww.find_wake([w("hey", 0, 1, 0.99), w("copilot", 1, 2, 0.96)]) == (0, 0.96)
    assert ww.find_wake([]) is None


def test_the_chime():
    pcm = ww.chime_pcm()
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    assert len(samples) == int(0.07 * RATE) + int(0.09 * RATE)
    assert max(abs(s) for s in samples) > 5000
    assert abs(samples[0]) < 100 and abs(samples[-1]) < 1000  # faded in and out


def test_the_vosk_spotter_listens_for_the_phrase():
    made = []

    class Rec:
        def __init__(self, model, rate, grammar):
            made.append((model, rate, json.loads(grammar)))
            self.words = None

        def SetWords(self, on):
            self.words = on

    vosk = type("Vosk", (), {"KaldiRecognizer": Rec})
    s = ww.vosk_spotter("the model", 16_000, vosk=vosk)
    assert made == [("the model", 16_000, ["hey copilot", "[unk]"])]
    assert s._rec.words is True


# ---- the voice worker ----------------------------------------------------
def make_worker(*results, transcript="gear down"):
    sd = make_sounddevice()
    out: list[dict] = []
    rec = PushToTalkRecorder(sd=sd, preroll_sec=0.3)
    rec.listen()
    rec.wait_idle()
    clock = [100.0]

    def predict(text):
        return {"intent": "toggle_landing_gear", "slots": {"state": "down"}, "confidence": 0.99}

    worker = VoiceWorker(
        out.append,
        rec,
        lambda raw, rate: transcript,
        predict,
        release_tail_sec=0,
        clock=lambda: clock[0],
        wake=spotter(*results),
    )
    return worker, out, sd, clock


def hear(worker, sd, *pieces):
    """Sound on the open mic, polled as the worker does."""
    pcm = b"".join(pieces)
    n = int(0.05 * RATE) * 2
    for i in range(0, len(pcm), n):
        sd.streams[-1].feed(pcm[i : i + n])
        worker.tick()


ONE_BREATH = (quiet(1.0), speech(1.4), quiet(2.0))


def test_the_worker_listens_only_once_switched_on():
    worker, out, sd, _ = make_worker((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    hear(worker, sd, *ONE_BREATH)
    assert out == [] and not worker.recorder.hear_between
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, *ONE_BREATH)
    assert out == [{"type": "wake", "wake": 1, "conf": 1.0, "one_breath": True}]


def test_a_taken_wake_is_recognised_as_its_command():
    worker, out, sd, _ = make_worker((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, *ONE_BREATH)
    out.clear()
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    captured, result = out
    assert captured["type"] == "captured" and captured["cmd"] == "c7" and captured["wake"]
    assert captured["audio_sec"] == pytest.approx(1.7, abs=0.1)
    assert result["type"] == "result" and result["cmd"] == "c7" and result["text"] == "gear down"


def test_taken_before_the_command_is_over():
    worker, out, sd, _ = make_worker((2.4, WAKE))
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, quiet(1.0), speech(0.6), quiet(1.0))
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    hear(worker, sd, speech(1.0), quiet(1.0))
    assert [m["type"] for m in out] == ["wake", "captured", "result"]


def test_no_command_dropped_and_a_press():
    worker, out, sd, _ = make_worker((2.4, WAKE))
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, quiet(1.0), speech(0.6), quiet(1.0))
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    hear(worker, sd, quiet(5.0))
    assert out[-1] == {"type": "error", "cmd": "c7", "outcome": "no_command"}
    # Dropped (the plugin is busy): nothing more comes of it.
    worker, out, sd, _ = make_worker((2.4, WAKE))
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, quiet(1.0), speech(0.6), quiet(1.0))
    worker.handle({"op": "wake_drop", "wake": 1})
    hear(worker, sd, speech(1.0), quiet(5.0))
    assert [m["type"] for m in out] == ["wake"]
    # A key pressed while it listens: the key wins.
    worker, out, sd, _ = make_worker((2.4, WAKE))
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, quiet(1.0), speech(0.6), quiet(1.0))
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    worker.handle({"op": "start", "cmd": "c8"})
    assert out[-1] == {"type": "error", "cmd": "c7", "outcome": "wake_cancelled"}


def test_not_while_vimaan_speaks():
    worker, out, sd, clock = make_worker((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    worker.handle({"op": "wake", "on": True})
    worker.handle({"op": "wake_mute", "sec": 10.0})
    hear(worker, sd, *ONE_BREATH)
    assert out == []
    clock[0] += 11.0  # quiet again: it listens afresh
    worker.wake = spotter((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    hear(worker, sd, *ONE_BREATH)
    assert [m["type"] for m in out] == ["wake"]


def test_a_wake_never_taken_is_forgotten():
    worker, out, sd, clock = make_worker(
        (2.4, WAKE), (6.4, [w("hey", 5.0, 5.2), w("copilot", 5.2, 5.6)])
    )
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, quiet(1.0), speech(0.6), quiet(2.0))
    clock[0] += voice_worker.WAKE_DECISION_SEC + 1
    hear(worker, sd, quiet(1.4), speech(0.6), quiet(1.0))
    assert [m["wake"] for m in out if m["type"] == "wake"] == [1, 2]


def test_a_file_heard_through_the_bridge(tmp_path):
    path = tmp_path / "hey.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(quiet(1.0) + speech(1.4))  # the worker adds the quiet after
    worker, out, _, _ = make_worker((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    worker.handle({"op": "hear", "path": str(path)})
    assert out[-1]["msg"] == "hear: the wake word is off"
    worker.handle({"op": "wake", "on": True})
    worker.handle({"op": "hear", "path": str(path)})
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    assert [m["type"] for m in out][-3:] == ["wake", "captured", "result"]


def test_without_the_open_mic_there_is_no_wake_word():
    worker, out, sd, _ = make_worker((2.6, WAKE))
    worker.recorder = PushToTalkRecorder(sd=sd)  # no pre-roll: the mic opens on a press
    worker.handle({"op": "wake", "on": True})
    assert worker.wake_on is False


# ---- the recorder ----------------------------------------------------------
def test_the_recorder_hands_over_what_it_hears_between_presses():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd, preroll_sec=0.3)
    rec.listen()
    rec.wait_idle()
    sd.streams[-1].feed(speech(0.5))
    rec.poll()
    assert rec.take_heard() == (b"", False)  # off: nothing kept beyond the pre-roll
    rec.hear_between = True
    sd.streams[-1].feed(speech(0.5))
    rec.poll()
    raw, gap = rec.take_heard()
    assert len(raw) == int(0.5 * RATE) * 2 and gap is False
    sd.streams[-1].feed(speech(0.2))
    rec.poll()
    rec.start()  # a press: the listener's sound is cut off
    assert rec.take_heard() == (b"", True)
    rec.stop()
    sd.streams[-1].feed(speech(3.0))  # never taken: only the last 2 s wait
    rec.poll()
    raw, gap = rec.take_heard()
    assert len(raw) == int(2.0 * RATE) * 2 and gap is True


def test_a_wake_command_is_decoded_with_the_phrase_and_stripped():
    # The worker's wake recogniser knows the phrase: its transcript has it in
    # front, and the command is what comes after.
    class Stream:
        def __init__(self, text):
            self.text, self.fed_bytes = text, 0

        def begin(self):
            pass

        def feed(self, pcm):
            self.fed_bytes += len(pcm)

        def finish(self):
            return self.text

    worker, out, sd, _ = make_worker((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    worker.wake_stt = Stream("hey copilot gear down")
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, *ONE_BREATH)
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    assert out[-1]["type"] == "result" and out[-1]["text"] == "gear down"
    # Only the phrase heard (its last sound taken for speech after it): no
    # command, quietly (L.5.3: "hey copilot" alone was "not understood").
    worker, out, sd, _ = make_worker((2.6, [*WAKE, w("[unk]", 1.6, 2.4)]))
    worker.wake_stt = Stream("hey copilot")
    worker.handle({"op": "wake", "on": True})
    hear(worker, sd, *ONE_BREATH)
    worker.handle({"op": "wake_take", "wake": 1, "cmd": "c7"})
    assert out[-1] == {"type": "error", "cmd": "c7", "outcome": "no_command"}
