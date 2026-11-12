"""python -m simtest <command>: see simtest/__init__.py."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import voice, xplane
from .bridge import Bridge, NoAnswer


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _say_line(r: dict[str, Any]) -> str:
    if not r.get("ok", True) and r.get("error"):
        return f"✕ {r['error']}"
    slots = " ".join(f"{k}={v}" for k, v in (r.get("slots") or {}).items())
    checks = r.get("checks") or []
    checked = f"  checks {sum(c['ok'] for c in checks)}/{len(checks)} ok" if checks else ""
    return (
        f"heard “{r.get('heard')}” → {r.get('intent')} {slots} → {r.get('outcome')}"
        f"  said: {' / '.join(r.get('replies') or ['(nothing)'])}{checked}"
        f"  ({r.get('latency_ms')} ms)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m simtest", description=__doc__)
    ap.add_argument("--log-dir", help="the plugin's log folder (default ~/Vimaan_Logs)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bridge", help="switch the test bridge on or off")
    b.add_argument("state", choices=("on", "off"))
    sub.add_parser("status", help="what Vimaan and the sim are doing")

    s = sub.add_parser("say", help="a command, as if spoken")
    s.add_argument("phrase", nargs="?")
    how = s.add_mutually_exclusive_group()
    how.add_argument("--text", action="store_true", help="straight to the model, no recognition")
    how.add_argument("--wav", help="a recording to play instead of generated speech")
    s.add_argument("--voice", default=voice.DEFAULT_VOICE)
    s.add_argument("--json", action="store_true", help="print the full result")

    r = sub.add_parser("read", help="read datarefs (name[i] for one element)")
    r.add_argument("datarefs", nargs="+")
    w = sub.add_parser("write", help="write a dataref")
    w.add_argument("dataref")
    w.add_argument("value", type=float)
    c = sub.add_parser("command", help="run an X-Plane command (counts as the pilot's)")
    c.add_argument("name")
    c.add_argument("--phase", choices=("once", "begin", "end"), default="once")
    pl = sub.add_parser("place", help="put the aircraft in the air, engines running")
    pl.add_argument("--lat", type=float, required=True)
    pl.add_argument("--lon", type=float, required=True)
    pl.add_argument("--alt-ft", type=float, required=True)
    pl.add_argument("--hdg", type=float, required=True, help="degrees true")
    pl.add_argument("--kt", type=float, required=True)
    a = sub.add_parser("airport", help="move to an airport")
    a.add_argument("icao")
    ac = sub.add_parser("aircraft", help="load an aircraft (.acf path)")
    ac.add_argument("path")

    sub.add_parser("launch", help="start X-Plane through Steam and wait for it")
    f = sub.add_parser("fly", help="start a flight in the air, engines running, and get it running")
    f.add_argument("aircraft", help=f"one of {', '.join(xplane.AIRCRAFT)}, or an .acf path")
    f.add_argument("--lat", type=float, default=47.45)  # over Seattle
    f.add_argument("--lon", type=float, default=-122.31)
    f.add_argument("--alt-ft", type=float, default=6000)
    f.add_argument("--hdg", type=float, default=180, help="degrees true")
    f.add_argument("--kt", type=float, default=150)
    sub.add_parser("quit", help="switch the bridge off and quit X-Plane")

    v = sub.add_parser("voices", help="offline: what Vimaan's recogniser hears in generated speech")
    v.add_argument("phrases", nargs="+")
    v.add_argument("--voice", action="append", help="voices to try (default: a few clear ones)")

    args = ap.parse_args(argv)
    if args.cmd == "voices":
        return _voices(args)
    if args.cmd in ("launch", "fly", "quit"):
        return _xplane(args)

    br = Bridge(args.log_dir)
    if args.cmd == "bridge":
        if args.state == "on":
            br.enable()
            print(
                f"Bridge on ({br.flag}). While X-Plane runs, scripts on this Mac can now "
                "control Vimaan and the sim. Vimaan says “Test bridge on”.\n"
                "Turn it off with: python -m simtest bridge off"
            )
        else:
            br.disable()
            print("Bridge off.")
        return 0
    try:
        if args.cmd == "status":
            _print(br.state())
        elif args.cmd == "say":
            if args.wav:
                result = br.say_wav(args.wav)
            elif not args.phrase:
                ap.error("say needs a phrase (or --wav)")
            elif args.text:
                result = br.say_text(args.phrase)
            else:
                result = br.say_wav(voice.synthesize(args.phrase, args.voice))
            _print(result) if args.json else print(_say_line(result))
            return 0 if result.get("ok", True) else 1
        elif args.cmd == "read":
            _print(br.read(*args.datarefs))
        elif args.cmd == "write":
            _print(br.write(args.dataref, args.value))
        elif args.cmd == "command":
            _print(br.command(args.name, args.phase))
        elif args.cmd == "place":
            _print(br.place(args.lat, args.lon, args.alt_ft, args.hdg, args.kt))
        elif args.cmd == "airport":
            _print(br.airport(args.icao))
        elif args.cmd == "aircraft":
            _print(br.aircraft(args.path))
    except NoAnswer as exc:
        print(f"✕ {exc}", file=sys.stderr)
        return 2
    return 0


def _xplane(args: argparse.Namespace) -> int:
    try:
        if args.cmd == "launch":
            print("Starting X-Plane through Steam (pick “X-Plane” if Steam asks)...")
            xplane.launch()
            print("X-Plane is up.")
        elif args.cmd == "fly":
            if not xplane.running():
                raise xplane.XPlaneError("X-Plane isn't running: python -m simtest launch")
            body = xplane.air_start_body(
                args.aircraft, args.lat, args.lon, args.alt_ft, args.hdg, args.kt
            )
            xplane.start_flight(body)
            print(f"Flying: {args.aircraft} at {args.alt_ft:.0f} ft, {args.kt:.0f} kt.")
        else:
            Bridge(args.log_dir).disable()
            xplane.quit_xplane()
            print("Asked X-Plane to quit (it asks to confirm unless told not to once).")
    except xplane.XPlaneError as exc:
        print(f"✕ {exc}", file=sys.stderr)
        return 2
    return 0


def _voices(args: argparse.Namespace) -> int:
    if not voice.available():
        print("✕ macOS 'say' is not available here", file=sys.stderr)
        return 2
    names = args.voice or list(voice.CANDIDATE_VOICES)
    installed = set(voice.voices())
    for name in names:
        if name not in installed:
            print(f"{name}: not installed")
            continue
        right = 0
        for phrase in args.phrases:
            heard = voice.hear(voice.synthesize(phrase, name))
            right += heard == phrase
            mark = "✓" if heard == phrase else "✕"
            print(f"  {mark} {name}: “{phrase}” → “{heard}”")
        print(f"{name}: {right}/{len(args.phrases)} heard exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
