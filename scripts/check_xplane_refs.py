"""Check every X-Plane dataref/command the plugin uses exists in a real install.

X-Plane ships its full lists as ``Resources/plugins/DataRefs.txt`` and
``Commands.txt``. A name missing from them makes ``findDataRef`` /
``findCommand`` return None in the sim, which the plugin can only report at
runtime — this catches it before a flight.

Usage::

    python scripts/check_xplane_refs.py --xplane "/path/to/X-Plane 12"
    python scripts/check_xplane_refs.py --xplane ... plugin/PI_VimaanCoPilot.py tests/fake_xplane.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_SOURCES = ("plugin/PI_VimaanCoPilot.py", "tests/fake_xplane.py")
NAME_RE = re.compile(r"[\"'](sim/[A-Za-z0-9_/\[\]]+)[\"']")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--xplane", required=True, help="X-Plane 12 root folder")
    parser.add_argument("sources", nargs="*", default=list(DEFAULT_SOURCES))
    args = parser.parse_args(argv)

    datarefs, commands = load_names(args.xplane)
    missing = 0
    for src in args.sources:
        path = src if os.path.isabs(src) else os.path.join(REPO_ROOT, src)
        for name in sorted(names_in(path)):
            if name not in datarefs and name not in commands:
                print(f"MISSING  {name}  ({src})")
                missing += 1
    total = sum(
        len(names_in(s if os.path.isabs(s) else os.path.join(REPO_ROOT, s))) for s in args.sources
    )
    print(f"{total - missing}/{total} names found in X-Plane's DataRefs.txt / Commands.txt")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
