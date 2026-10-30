"""Summarise a Vimaan plugin session from its telemetry files.

Reads ``~/Vimaan_Logs/vimaan_<ts>.events.jsonl`` (+ ``.faults.log``) written by
``ML/vimaan_nlu/telemetry.py`` and prints: how the session ended (clean stop
vs crash), startup timings, per-stage latency percentiles, every command with
its outcome, and every warning/error worth looking at.

Usage::

    python scripts/vimaan_log_report.py              # newest session
    python scripts/vimaan_log_report.py --list       # list sessions
    python scripts/vimaan_log_report.py path/to/vimaan_20261027_101500.events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_DIR = Path.home() / "Vimaan_Logs"
STAGES = ("model_load", "capture", "stt", "nlu", "dispatch", "self_test")
PROBLEM_KINDS = {
    "dependency_missing",
    "microphone_unavailable",
    "stage_failed",
    "uncaught_exception",
    "execute_error",
    "listen_worker_error",
    "flight_loop_error",
    "sim_stall",
    "main_thread_overrun",
    "dataref_read_error",
    "safety_hold",
    "slot_rejected",
    "worker_spawn_failed",
    "worker_exit",
    "worker_fatal",
    "stt_fallback",
    "implausible",
}


def load_events(path: Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn last line is what a hard crash mid-write looks like.
                events.append({"kind": "_torn_line", "line": n})
    return events


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


def is_problem(ev: dict) -> bool:
    if ev.get("kind") in PROBLEM_KINDS or ev.get("slow"):
        return True
    if ev.get("kind") in ("sim_write", "sim_command", "verify") and ev.get("ok") is False:
        return True
    if ev.get("kind") == "crank_end" and ev.get("result") != "running":
        return True
    if ev.get("kind") == "worker_error" and ev.get("outcome") != "too_short":
        return True
    return ev.get("kind") == "self_test_probe" and not ev.get("ok")


def report(path: Path) -> str:
    events = load_events(path)
    out: list[str] = []
    w = out.append
    w(f"Session: {path.name}  ({len(events)} events)")

    start = next((e for e in events if e["kind"] == "session_start"), {})
    if start:
        w(f"  Python {start.get('python')} @ {start.get('executable')}")
        w(
            f"  torch {start.get('torch_version', '-')}  backend={start.get('backend')}  "
            f"stt={start.get('stt_backend')}"
        )
        w(f"  import ms: {start.get('import_ms')}")
    plugin_start = next((e for e in events if e["kind"] == "plugin_start"), None)
    if plugin_start:
        w(f"  ready={plugin_start.get('ready')}  sim={plugin_start.get('sim')}")
    load = next(
        (e for e in events if e["kind"] == "stage" and e.get("stage") == "model_load"), None
    )
    if load:
        v = load.get("versions") or load  # in-process (old) or worker-reported (new)
        w(
            f"  model: {load.get('model_path')}  quantized={load.get('quantized')}  "
            f"torch {v.get('torch_version', '-')}  transformers {v.get('transformers_version', '-')}"
        )
        if load.get("executable"):
            w(
                f"  voice worker: pid {load.get('pid')}  Python {load.get('python')} @ "
                f"{load.get('executable')}  ready in {load.get('ms')} ms"
            )
        if load.get("stt_backend"):
            w(f"  speech recognition: {load.get('stt_backend')}  {load.get('stt') or ''}")
        if isinstance(load.get("mic"), dict) and load["mic"].get("using"):
            w(f"  microphone: {load['mic'].get('using')!r} of {load['mic'].get('input_devices')}")
    mic = next((e for e in events if e["kind"] == "microphone_ready"), None)
    if mic:
        w(f"  microphone: {mic.get('using')!r} of {mic.get('input_devices')}")

    kinds = {e["kind"] for e in events}
    w("")
    if "plugin_stop" in kinds or "session_end" in kinds:
        w("Ended: clean shutdown")
    else:
        last = events[-1] if events else {}
        w(
            f"Ended: ABRUPTLY (crash, hang or force-quit) — last event at t={last.get('t')}s: "
            f"{last.get('kind')}"
        )
    faults = path.with_name(path.name.replace(".events.jsonl", ".faults.log"))
    if faults.exists() and faults.stat().st_size:
        text = faults.read_text(errors="replace")
        if "Timeout (" in text and "<no Python frame>" in text:
            w("  faults file: hang-detector dump with X-Plane's main thread in its own")
            w("  code (no Python frame) — the sim stopped calling plugins for 30 s:")
            w("  menu / loading / quit screen, or a freeze inside X-Plane, not Vimaan.")
        w(f"  faults file has content ({faults.stat().st_size} bytes) — last 40 lines:")
        for line in faults.read_text(errors="replace").splitlines()[-40:]:
            w(f"    {line}")

    durations: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e["kind"] == "stage" and e.get("cmd") != "warmup":
            durations[e["stage"]].append(e["ms"])
    done = [e for e in events if e["kind"] == "command_done"]
    for e in done:
        durations["TOTAL press->done"].append(e["total_ms"])
        if "speech_to_action_ms" in e:
            durations["speech end->action"].append(e["speech_to_action_ms"])
    if durations:
        w("")
        w(f"{'Latency (ms)':<22}{'n':>5}{'p50':>10}{'p95':>10}{'max':>10}")
        for name in (*STAGES, "speech end->action", "TOTAL press->done"):
            vals = durations.get(name)
            if vals:
                w(
                    f"{name:<22}{len(vals):>5}{percentile(vals, 50):>10.1f}"
                    f"{percentile(vals, 95):>10.1f}{max(vals):>10.1f}"
                )

    if done:
        w("")
        w(f"Commands ({len(done)}): {dict(Counter(e['outcome'] for e in done))}")
        nlu = {e["cmd"]: e for e in events if e["kind"] == "stage" and e.get("stage") == "nlu"}
        for e in done:
            n = nlu.get(e["cmd"], {})
            conf = n.get("confidence")
            w(
                f"  {e['cmd']} {e['outcome']:<16} {e['total_ms']:>8.0f}ms  "
                f"{e.get('text', '')!r} -> {n.get('intent', '-')} "
                f"{'' if conf is None else f'{conf:.2f} '}{n.get('slots', '')}"
            )

    problems = [e for e in events if is_problem(e)]
    w("")
    w(f"Problems ({len(problems)}):" if problems else "Problems: none")
    for e in problems:
        detail = {k: v for k, v in e.items() if k not in ("ts", "thread", "traceback")}
        w(f"  t={e.get('t')}s {detail}")
        if e.get("traceback"):
            w("    " + e["traceback"].strip().splitlines()[-1])

    end = next((e for e in reversed(events) if e["kind"] in ("session_end", "heartbeat")), None)
    if end:
        w("")
        w(f"Peak RSS: {end.get('peak_rss_mb')} MB")
    return "\n".join(out)


def sessions(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("vimaan_*.events.jsonl"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("events", nargs="?", type=Path, help="events.jsonl (default: newest)")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="log directory")
    parser.add_argument("--list", action="store_true", help="list sessions and exit")
    args = parser.parse_args(argv)

    if args.list:
        for p in sessions(args.dir):
            print(p)
        return 0
    path = args.events
    if path is None:
        found = sessions(args.dir)
        if not found:
            print(f"No sessions in {args.dir}", file=sys.stderr)
            return 1
        path = found[-1]
    print(report(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
