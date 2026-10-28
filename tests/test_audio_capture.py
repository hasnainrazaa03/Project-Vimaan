"""Tests for vimaan_nlu.audio_capture (hold-to-talk recorder)."""

import struct

from fake_xplane import make_sounddevice
from vimaan_nlu.audio_capture import PushToTalkRecorder

ONE_SEC = struct.pack("<h", 1000) * 16000


def test_records_between_start_and_stop():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    assert rec.is_recording
    sd.streams[-1].feed(ONE_SEC)
    sd.streams[-1].feed(ONE_SEC[:16000])  # +0.5 s
    raw = rec.stop()
    assert not rec.is_recording
    assert rec.seconds(raw) == 1.5
    assert sd.streams[-1].closed


def test_caps_at_max_seconds_and_flags_truncation():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd, max_sec=1.0)
    rec.start()
    for _ in range(3):
        sd.streams[-1].feed(ONE_SEC)
    raw = rec.stop()
    assert rec.seconds(raw) == 1.0
    assert rec.truncated


def test_each_press_starts_fresh():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
    sd.streams[-1].feed(ONE_SEC)
    rec.stop()
    rec.start()
    assert rec.stop() == b""


def test_start_twice_is_harmless():
    sd = make_sounddevice()
    rec = PushToTalkRecorder(sd=sd)
    rec.start()
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
    rec._callback(b"\x00\x00", 1, None, "input overflow")
    assert rec.overflows == 1
    rec.stop()


def test_describe_lists_only_input_devices():
    info = PushToTalkRecorder(sd=make_sounddevice()).describe()
    assert info["input_devices"] == ["Fake Mic"]
    assert info["using"] == "Fake Mic" and info["sample_rate"] == 16000
