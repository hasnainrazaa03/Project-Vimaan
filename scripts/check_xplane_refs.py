"""Check every X-Plane dataref/command the plugin uses exists in a real install.

X-Plane ships its full lists as ``Resources/plugins/DataRefs.txt`` and
``Commands.txt``. A name missing from them makes ``findDataRef`` /
``findCommand`` return None in the sim, which the plugin can only report at
runtime — this catches it before a flight.

Usage::

    python scripts/check_xplane_refs.py --xplane "/path/to/X-Plane 12"
    python scripts/check_xplane_refs.py --xplane ... plugin/PI_VimaanCoPilot.py tests/fake_xplane.py

Besides the source files, every binding declared in the command table
(``ML/vimaan_nlu/commands.py``) is checked, and so are the aircraft-specific
controls the profile overrides use (``ML/vimaan_nlu/aircraft.py``, e.g.
``laminar/A333/...``): those exist only in an aircraft's own scripts, so they
are checked against what the aircraft in ``Aircraft/`` create.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "ML"))
DEFAULT_SOURCES = (
    "plugin/PI_VimaanCoPilot.py",
    "tests/fake_xplane.py",
    "ML/vimaan_nlu/aircraft.py",
    "ML/vimaan_nlu/probe.py",
)
NAME_RE = re.compile(r"[\"'](sim/[A-Za-z0-9_/\[\]]+)[\"']")
# create_command("laminar/A333/autopilot/a_thr_toggle", ...) in an xlua
# script, or create_dataref(...) for the datarefs overrides need live.
CREATED_RE = re.compile(r"create_(?:command|dataref)\(\s*[\"']([^\"']+)[\"']")


def load_names(xplane_root: str) -> tuple[set[str], set[str]]:
    plugins = os.path.join(xplane_root, "Resources", "plugins")
    datarefs: set[str] = set()
    commands: set[str] = set()
    for fname, target in (("DataRefs.txt", datarefs), ("Commands.txt", commands)):
        with open(os.path.join(plugins, fname), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("sim/"):
                    target.add(line.split()[0])
    return datarefs, commands


def names_in(path: str) -> set[str]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    # Templated names like f"sim/flight_controls/landing_gear_{state}" are
    # partial; check them by expanding the known values.
    names = {re.sub(r"\[\d+\]$", "", n) for n in NAME_RE.findall(text)}
    for template, values in (
        ("sim/flight_controls/landing_gear_{state}", ("up", "down")),
        ("sim/flight_controls/park_brake_", ("set", "release")),
    ):
        if template.split("{")[0] in text:
            prefix = template.split("{")[0]
            names.update(prefix + v for v in values)
    return {n for n in names if not n.endswith(("_", "/"))}


def aircraft_commands(xplane_root: str) -> set[str]:
    """Commands and datarefs the installed aircraft create in their xlua scripts."""
    created: set[str] = set()
    for folder, _dirs, files in os.walk(os.path.join(xplane_root, "Aircraft")):
        for fname in files:
            if fname.endswith(".lua"):
                path = os.path.join(folder, fname)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    # Not commented-out lines: the 737's autothrottle_arm_toggle
                    # is "--create_command(...)", and doesn't exist in the sim.
                    live = "\n".join(ln for ln in fh if not ln.lstrip().startswith("--"))
                    created.update(CREATED_RE.findall(live))
    return created


def table_names() -> set[str]:
    """Every X-Plane name the command table declares (bindings + handler params)."""
    from vimaan_nlu.commands import COMMANDS

    names: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str) and value.startswith("sim/"):
            names.add(value)
        elif isinstance(value, Mapping):
            for v in value.values():
                collect(v)

    for c in COMMANDS:
        names.update(c.bindings)
        collect(c.params)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xplane", required=True, help="X-Plane 12 root folder")
    parser.add_argument("sources", nargs="*", default=list(DEFAULT_SOURCES))
    args = parser.parse_args(argv)

    datarefs, commands = load_names(args.xplane)
    missing = 0
    sources = {
        src: names_in(src if os.path.isabs(src) else os.path.join(REPO_ROOT, src))
        for src in args.sources
    }
    sources["command table"] = table_names()
    for src, names in sources.items():
        for name in sorted(names):
            if name not in datarefs and name not in commands:
                print(f"MISSING  {name}  ({src})")
                missing += 1
    total = sum(len(names) for names in sources.values())
    print(f"{total - missing}/{total} names found in X-Plane's DataRefs.txt / Commands.txt")

    from vimaan_nlu.aircraft import override_names

    created = aircraft_commands(args.xplane)
    own = override_names()
    absent = [n for n in own if n not in created]
    for name in absent:
        print(f"MISSING  {name}  (no installed aircraft creates it)")
    print(
        f"{len(own) - len(absent)}/{len(own)} aircraft-specific controls found in aircraft scripts"
    )
    return 1 if missing or absent else 0


if __name__ == "__main__":
    sys.exit(main())
