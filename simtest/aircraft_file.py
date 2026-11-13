"""Numbers from an aircraft's .acf file that X-Plane doesn't publish as
datarefs. ``sim/aircraft/view/acf_Vfe`` is the *first-notch* flap limit
(``_Vfe1_kts``); the full-flap limit is ``_Vfem_kts`` (Baron 152 / 122 kt,
737 250 / 162 kt), found by the A2 run on 2026-11-12."""

from __future__ import annotations

import re
from pathlib import Path

_LINE = re.compile(rb"^P acf/(_Vfe1_kts|_Vfem_kts) ([0-9.]+)\s*$", re.M)


def flap_limits(acf_path: str | Path) -> dict[str, float]:
    """{"first": first-notch limit, "full": full-flap limit} in KIAS (those found)."""
    try:
        data = Path(acf_path).read_bytes()
    except OSError:
        return {}
    names = {b"_Vfe1_kts": "first", b"_Vfem_kts": "full"}
    return {names[m.group(1)]: float(m.group(2)) for m in _LINE.finditer(data)}
