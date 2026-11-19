"""Speed margins against the aircraft's limits (Phase F.2), observe only.

Pure Python: the plugin builds :class:`Limits` from the loaded aircraft (the
V-speeds X-Plane publishes, and the full-flap limit from the .acf) and gets
:func:`margins` for each once-a-second flight sample, recorded with it.
Nothing acts on them yet; callouts that would (F.4) need the owner's OK.

Each margin is knots (Mach for ``mmo``) between the aircraft and the limit:
positive inside it, negative beyond it, ``None`` when it doesn't apply
(flaps up, gear up, on the ground for the stall) or the limit is unknown.

The stall speeds are the .acf's (one weight), so the stall margin is a
guide, not the aircraft's stall speed at its weight now.
"""

from __future__ import annotations

from dataclasses import dataclass

from vimaan_nlu.aircraft import flap_limit_kt

FLAPS_OUT_RATIO = 0.01  # flap handle above this: flaps out (as flight_phase)
GEAR_UP_RATIO = 0.01  # gear deploy at or below this: up
MARGINS = ("stall", "flaps", "gear", "vno", "vne", "mmo")


@dataclass(frozen=True)
class Limits:
    """An aircraft's speed limits, KIAS (``mmo`` Mach). ``None`` or 0:
    unknown, or none (a piston has no Mmo, fixed gear no Vle)."""

    vs_kt: float | None = None  # stall, clean
    vso_kt: float | None = None  # stall, landing configuration
    vfe_first_kt: float | None = None  # flaps, first notch
    vfe_full_kt: float | None = None  # flaps, full
    flap_detents: int = 0
    vle_kt: float | None = None  # gear extended
    vno_kt: float | None = None  # caution range above this (smooth air only)
    vne_kt: float | None = None  # never exceed (Vmo on jets)
    mmo: float | None = None
    gear_retractable: bool | None = None


def flap_notch(handle_ratio: float, detents: int) -> int:
    """The notch a flap handle ratio is at (0 up, ``detents`` full)."""
    if detents <= 0:
        return 0
    return max(0, min(detents, round(handle_ratio * detents)))


def _below(limit: float | None, value: float | None, digits: int = 1) -> float | None:
    if not limit or value is None:
        return None
    return round(limit - value, digits)


def margins(
    limits: Limits,
    ias_kt: float | None,
    mach: float | None = None,
    flaps: float | None = None,
    gear: float | None = None,
    on_ground: bool | None = None,
) -> dict[str, float | None]:
    """The margin to each limit now (see the module docstring)."""
    flaps_out = flaps is not None and flaps > FLAPS_OUT_RATIO
    out: dict[str, float | None] = dict.fromkeys(MARGINS)
    if ias_kt is not None and on_ground is False:
        stall = limits.vso_kt if flaps_out else limits.vs_kt
        if stall:
            out["stall"] = round(ias_kt - stall, 1)
    if flaps_out and flaps is not None:
        notch = flap_notch(flaps, limits.flap_detents) or 1
        limit = flap_limit_kt(
            notch, limits.flap_detents or 1, limits.vfe_first_kt or None, limits.vfe_full_kt or None
        )
        out["flaps"] = _below(limit, ias_kt)
    if limits.gear_retractable and gear is not None and gear > GEAR_UP_RATIO:
        out["gear"] = _below(limits.vle_kt, ias_kt)
    out["vno"] = _below(limits.vno_kt, ias_kt)
    out["vne"] = _below(limits.vne_kt, ias_kt)
    out["mmo"] = _below(limits.mmo, mach, 3)
    return out
