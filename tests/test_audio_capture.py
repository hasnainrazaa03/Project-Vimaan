"""Tests for vimaan_nlu.audio_capture (hold-to-talk recorder)."""

import struct

from fake_xplane import make_sounddevice
from vimaan_nlu.audio_capture import PushToTalkRecorder

ONE_SEC = struct.pack("<h", 1000) * 16000


def test_records_between_start_and_stop():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    assert rec.is_recording
    sd.streams[-1].feed(ONE_SEC)
    sd.streams[-1].feed(ONE_SEC[:16000])  # +0.5 s
    raw = rec.stop()
    assert not rec.is_recording
    assert rec.seconds(raw) == 1.5
    rec.wait_idle()
    assert sd.streams[-1].closed


def test_caps_at_max_seconds_and_flags_truncation():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd, max_sec=1.0)
    rec.start()
    rec.wait_idle()
    for _ in range(3):
        sd.streams[-1].feed(ONE_SEC)
    raw = rec.stop()
    assert rec.seconds(raw) == 1.0
    assert rec.truncated


def test_each_press_starts_fresh():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    sd.streams[-1].feed(ONE_SEC)
    rec.stop()
    rec.start()
    rec.wait_idle()
    assert rec.stop() == b""


def test_start_twice_is_harmless():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    rec.start()
    assert len(sd.streams) == 1
    rec.stop()


def test_stop_without_start_returns_empty():
    rec = PushToTalkRecorder(sd=make_sounddevice())
    assert rec.stop() == b""


def test_overflow_status_counted():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    sd.streams[-1].feed(ONE_SEC)
    sd.streams[-1].overflow_next = True
    rec.poll()
    assert rec.overflows == 1
    rec.stop()


def test_no_python_audio_callback():
    # Inside X-Plane a Python callback thread is starved of the GIL (a 9 s
    # press captured 0.03 s). PortAudio must buffer in C, drained by poll().
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    assert sd.streams[-1].callback is None
    assert sd.streams[-1].latency >= 0.2  # survives 0.25 s between polls
    rec.stop()


def test_poll_drains_incrementally():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    sd.streams[-1].feed(ONE_SEC[:3200])
    assert rec.poll() == 1600
    assert rec.poll() == 0  # nothing new: never blocks
    sd.streams[-1].feed(ONE_SEC[:3200])
    assert rec.seconds(rec.stop()) == 0.2


def test_poll_when_not_recording_is_noop():
    assert PushToTalkRecorder(sd=make_sounddevice()).poll() == 0


def test_describe_lists_only_input_devices():
    info = PushToTalkRecorder(sd=make_sounddevice()).describe()
    assert info["input_devices"] == ["Fake Mic"]
    assert info["using"] == "Fake Mic" and info["sample_rate"] == 16000


class _SlowSD:
    """sounddevice whose open/close take time, like CoreAudio (~85/118 ms)."""

    def __init__(self, delay=0.05, fail=False):
        from fake_xplane import make_sounddevice

        self._sd, self.delay, self.fail = make_sounddevice(), delay, fail
        self.streams = self._sd.streams
        self.query_devices = self._sd.query_devices

    def RawInputStream(self, **kw):
        import time

        time.sleep(self.delay)
        if self.fail:
            raise OSError("Invalid number of channels")
        stream = self._sd.RawInputStream(**kw)
        orig_close = stream.close

        def slow_close():
            time.sleep(self.delay)
            orig_close()

        stream.close = slow_close
        return stream


def test_open_and_close_never_block_the_caller():
    import time

    sd = _SlowSD(delay=0.2)
    rec = PushToTalkRecorder(sd=sd)
    t = time.perf_counter()
    rec.start()
    assert time.perf_counter() - t < 0.05 and rec.busy
    rec.wait_idle()
    sd.streams[-1].feed(ONE_SEC)
    t = time.perf_counter()
    rec.stop()
    assert time.perf_counter() - t < 0.05 and rec.busy
    rec.wait_idle()
    assert sd.streams[-1].closed and not rec.busy


def test_open_failure_reported_via_error():
    rec = PushToTalkRecorder(sd=_SlowSD(delay=0.0, fail=True))
    rec.start()
    rec.wait_idle()
    assert isinstance(rec.error, OSError)


def test_release_while_still_opening_closes_the_stream():
    sd = _SlowSD(delay=0.1)
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    assert rec.stop() == b""
    rec.wait_idle()
    assert sd.streams[-1].closed


def test_take_new_streams_without_affecting_stop():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    rec.wait_idle()
    sd.streams[-1].feed(ONE_SEC[:8000])
    rec.poll()
    assert len(rec.take_new()) == 8000
    assert rec.take_new() == b""
    sd.streams[-1].feed(ONE_SEC[:4000])
    rec.poll()
    assert len(rec.take_new()) == 4000
    assert len(rec.stop()) == 12000  # stop still returns everything
    rec.start()  # a new press starts the stream cursor over
    rec.wait_idle()
    assert rec.take_new() == b""


# ---------------------------------------------------------------------------
# Pre-roll: the mic stays open between presses
# ---------------------------------------------------------------------------
def _tone(value, sec):
    return struct.pack("<h", value) * int(16000 * sec)


def _listening(preroll_sec=0.3):
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd, preroll_sec=preroll_sec)
    rec.listen()
    rec.wait_idle()
    return rec, sd


def test_a_press_starts_with_the_sound_just_before_it():
    rec, sd = _listening()
    assert rec.is_listening
    sd.streams[-1].feed(_tone(1, 1.0))
    rec.poll()
    sd.streams[-1].feed(_tone(2, 0.5))  # not polled yet: start() takes it too
    rec.start()
    assert rec.is_open  # nothing to wait for
    sd.streams[-1].feed(_tone(3, 0.5))
    raw = rec.stop()
    assert raw == _tone(2, 0.3) + _tone(3, 0.5)
    assert rec.preroll_frames == 4800


def test_only_the_newest_sound_is_kept_between_presses():
    rec, sd = _listening()
    for _ in range(50):  # 5 s of waiting, drained every 0.1 s
        sd.streams[-1].feed(_tone(1, 0.1))
        rec.poll()
    assert rec._ring_frames <= 4800 + 1600
    rec.start()
    assert rec.seconds(rec.stop()) == 0.3


def test_a_press_soon_after_opening_has_what_there_is():
    rec, sd = _listening()
    sd.streams[-1].feed(_tone(1, 0.1))
    rec.start()
    assert rec.preroll_frames == 1600
    assert rec.seconds(rec.stop()) == 0.1


def test_stale_sound_after_a_late_drain_is_not_kept():
    rec, sd = _listening()
    sd.streams[-1].feed(_tone(1, 0.2))
    rec.poll()
    sd.streams[-1].feed(_tone(2, 0.2))
    sd.streams[-1].overflow_next = True  # PortAudio lost what came after it
    rec.poll()
    sd.streams[-1].feed(_tone(3, 0.1))
    rec.start()
    assert rec.stop() == _tone(3, 0.1)
    assert rec.overflows == 0  # between presses: not the recording's overflow


def test_the_mic_stays_open_between_presses():
    rec, sd = _listening()
    for _ in range(2):
        rec.start()
        sd.streams[-1].feed(ONE_SEC)
        assert rec.seconds(rec.stop()) == 1.0
        assert rec.is_listening and not sd.streams[-1].closed
    assert len(sd.streams) == 1


def test_sound_between_presses_is_not_part_of_the_next_recording_beyond_the_preroll():
    rec, sd = _listening()
    rec.start()
    sd.streams[-1].feed(_tone(1, 1.0))
    rec.stop()
    sd.streams[-1].feed(_tone(2, 2.0))  # talking with the key up
    rec.poll()
    rec.start()
    assert rec.stop() == _tone(2, 0.3)


def test_without_preroll_the_mic_opens_on_each_press():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.listen()
    rec.wait_idle()
    assert sd.streams == [] and not rec.is_listening
    rec.start()
    rec.wait_idle()
    assert rec.preroll_frames == 0
    rec.stop()
    rec.wait_idle()
    assert sd.streams[-1].closed


def test_a_mic_lost_between_presses_is_reported_and_opened_again():
    rec, sd = _listening()
    lost = sd.streams[-1]

    def unplugged(frames):
        raise OSError("device unavailable")

    lost.feed(_tone(1, 0.1))
    lost.read = unplugged
    assert rec.poll() == 0
    assert isinstance(rec.listen_error, OSError) and not rec.is_listening
    rec.wait_idle()
    assert lost.closed
    rec.start()  # the next press opens it again, with no pre-roll this time
    rec.wait_idle()
    assert len(sd.streams) == 2 and rec.preroll_frames == 0
    sd.streams[-1].feed(ONE_SEC)
    assert rec.seconds(rec.stop()) == 1.0
    assert rec.is_listening  # and it stays open from then on


def test_a_mic_lost_during_a_press_is_an_error():
    rec, sd = _listening()
    rec.start()

    def unplugged(frames):
        raise OSError("device unavailable")

    sd.streams[-1].feed(_tone(1, 0.1))
    sd.streams[-1].read = unplugged
    rec.poll()
    assert isinstance(rec.error, OSError) and rec.listen_error is None
    rec.wait_idle()


def test_a_failed_open_between_presses_is_not_a_press_error():
    rec = PushToTalkRecorder(sd=_SlowSD(delay=0.0, fail=True), preroll_sec=0.3)
    rec.listen()
    rec.wait_idle()
    assert isinstance(rec.listen_error, OSError) and rec.error is None


def test_close_releases_the_mic():
    rec, sd = _listening()
    rec.close()
    assert sd.streams[-1].closed and not rec.is_listening and not rec.is_open
    rec.listen()
    rec.wait_idle()
    assert len(sd.streams) == 1  # closed for good


def test_preroll_is_in_the_startup_description():
    rec, _ = _listening(0.25)
    assert rec.describe()["preroll_ms"] == 250
