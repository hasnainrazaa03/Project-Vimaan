"""Microphone health checks (Phase 6F).

A dead or muted microphone produces near-silent audio; without a check the pilot
just gets a confusing "could not understand you" after wasting a push. These
pure helpers compute the RMS level of captured PCM and flag effective silence,
so the plugin can warn instead.

Pure-Python (``struct``/``math`` only — no numpy), so it runs in CI and imports
anywhere the package does.
"""

from __future__ import annotations

import math
import struct

# RMS below this (16-bit samples range 0..32767) is treated as effectively
# silent — a live mic in a quiet room still reads well above it.
SILENCE_RMS_THRESHOLD = 8.0


def rms_from_pcm16(raw: bytes) -> float:
    """Root-mean-square amplitude of little-endian signed 16-bit PCM bytes.

    Returns 0.0 for empty input. Trailing odd byte (if any) is ignored.
    """
    if not raw:
        return 0.0
    n = len(raw) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", raw[: n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


def is_silent(rms: float, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """True if an RMS level indicates an effectively silent / dead microphone."""
    return rms < threshold
