"""Tests for vimaan_nlu.audio_health (Phase 6F) — mic silence detection."""

import struct

import pytest
from vimaan_nlu.audio_health import SILENCE_RMS_THRESHOLD, is_silent, rms_from_pcm16


def _pcm16(samples):
    return struct.pack(f"<{len(samples)}h", *samples)


class TestRms:
    def test_silence_is_zero(self):
        assert rms_from_pcm16(_pcm16([0] * 100)) == 0.0

    def test_empty_is_zero(self):
        assert rms_from_pcm16(b"") == 0.0

    def test_rms_of_constant_equals_amplitude(self):
        assert rms_from_pcm16(_pcm16([1000] * 50)) == pytest.approx(1000.0)

    def test_odd_trailing_byte_ignored(self):
        assert rms_from_pcm16(_pcm16([500] * 3) + b"\x01") == pytest.approx(500.0)

    def test_loud_exceeds_quiet(self):
        assert rms_from_pcm16(_pcm16([8000] * 20)) > rms_from_pcm16(_pcm16([3] * 20))


class TestIsSilent:
    def test_zero_is_silent(self):
        assert is_silent(0.0)

    def test_loud_is_not_silent(self):
        assert not is_silent(500.0)

    def test_threshold_boundary(self):
        assert is_silent(SILENCE_RMS_THRESHOLD - 0.1)
        assert not is_silent(SILENCE_RMS_THRESHOLD)

    def test_a_dead_mic_capture_reads_silent(self):
        # All-zero PCM (muted / no device) -> rms 0 -> silent.
        assert is_silent(rms_from_pcm16(_pcm16([0] * 4000)))
