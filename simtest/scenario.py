"""Scenarios: a list of steps run against a live X-Plane through the bridge,
each checked against the sim, and a report (Phase A, A2).

A step either *says* something to Vimaan (in a generated voice, so it goes
through recognition) or does something as the *pilot* (an X-Plane command).
Then it checks:

- Vimaan's outcome, and optionally that a reply contains some words;
- Vimaan's own read-backs (``checks`` in the say result) all passed;
- the sim itself (``SimCheck``): a dataref, or something computed from
  several, must hold, possibly within a deadline (the aircraft has to turn
  or level off first);
- optionally a callout Vimaan should (or shouldn't) make.

Recognition and function are scored separately: when the generated voice is
misheard, the step is recorded as a recognition miss and repeated as text,
so a slip of the recogniser doesn't hide whether the sim did the right thing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import voice as voice_mod
from .bridge import Bridge


@dataclass(frozen=True)
class SimCheck:
    """``value(readings)`` must satisfy ``ok(value)`` (within ``within_sec``)."""

    what: str
    datarefs: tuple[str, ...]
    ok: Callable[[Any], bool]
    value: Callable[[dict[str, Any]], Any] = lambda r: next(iter(r.values()))
    within_sec: float = 0.0
    expected: str = ""


def equals(
    what: str, dataref: str, want: float, tol: float = 0.0, within_sec: float = 0.0
) -> SimCheck:
    return SimCheck(
        what,
        (dataref,),
        lambda v: v is not None and abs(float(v) - want) <= tol,
        within_sec=within_sec,
        expected=f"{want}" + (f" ± {tol}" if tol else ""),
    )


def heading_near(target: float, tol: float, within_sec: float) -> SimCheck:
    def diff(r: dict[str, Any]) -> float | None:
        h = r.get("sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot")
        return None if h is None else round(abs((float(h) - target + 180) % 360 - 180), 1)

    return SimCheck(
        f"turned to {target:03.0f}",
        ("sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot",),
        lambda d: d is not None and d <= tol,
        value=diff,
        within_sec=within_sec,
        expected=f"within {tol}° of {target:03.0f}",
    )


def levelled_at(
    target_ft: float, within_sec: float, alt_tol: float = 150, vs_tol: float = 300
) -> SimCheck:
    alt = "sim/cockpit2/gauges/indicators/altitude_ft_pilot"
    vs = "sim/cockpit2/gauges/indicators/vvi_fpm_pilot"

    def level(r: dict[str, Any]) -> dict[str, Any] | None:
        if r.get(alt) is None or r.get(vs) is None:
            return None
        return {"altitude": round(float(r[alt])), "vs": round(float(r[vs]))}

    return SimCheck(
        f"levelled off at {target_ft:.0f} ft",
        (alt, vs),
        lambda v: (
            v is not None and abs(v["altitude"] - target_ft) <= alt_tol and abs(v["vs"]) <= vs_tol
        ),
        value=level,
        within_sec=within_sec,
        expected=f"{target_ft:.0f} ± {alt_tol} ft, |V/S| ≤ {vs_tol}",
    )


@dataclass(frozen=True)
class Step:
    name: str
    say: str | None = None  # a phrase for Vimaan
    command: str | None = None  # or an X-Plane command, as the pilot
    outcomes: tuple[str, ...] = ("executed",)
    reply: str | None = None  # words one of Vimaan's replies must contain
    sim: tuple[SimCheck, ...] = ()
    callout: str | None = None  # Vimaan must say this (within a few seconds)
    no_callout: str | None = None  # and must not say this
    vimaan_checks: bool = True  # Vimaan's own read-backs must pass
    settle_sec: float = 0.0  # wait before checking (a knob turning, a mode settling)
    note: str = ""
    # Run only when this holds now (e.g. above flap speed); else skipped.
    only_if: SimCheck | None = None
    # Run only if the named step ran (wasn't skipped).
    after: str | None = None


@dataclass
class Runner:
    bridge: Bridge
    voice: str | None = voice_mod.DEFAULT_VOICE  # None: text only, no recognition
    # understood(result, phrase): did a voice result mean what the phrase
    # means? (simtest.understand.same). None: only an exact transcript counts.
    understood: Callable[[dict[str, Any], str], bool] | None = None
    poll_sec: float = 1.0
    callout_wait_sec: float = 4.0
    log: Callable[[str], None] = print
    results: list[dict[str, Any]] = field(default_factory=list)

    def run(self, steps: Sequence[Step]) -> list[dict[str, Any]]:
        ran: set[str] = set()
        for step in steps:
            skip = (step.after is not None and step.after not in ran) or (
                step.only_if is not None and not self._check(step.only_if)["ok"]
            )
            if skip:
                r = {
                    "step": step.name,
                    "skipped": True,
                    "ok": True,
                    "problems": [],
                    "note": step.note,
                }
            else:
                r = self.run_step(step)
                ran.add(step.name)
            self.results.append(r)
            self.log(line(r))
        return self.results

    # ------------------------------------------------------------------
    def run_step(self, step: Step) -> dict[str, Any]:
        r: dict[str, Any] = {"step": step.name, "said": step.say, "command": step.command}
        started = datetime.now()
        problems: list[str] = []
        if step.say:
            said = self._say(step.say, r)
            r.update(
                {
                    k: said.get(k)
                    for k in (
                        "cmd",
                        "intent",
                        "slots",
                        "outcome",
                        "replies",
                        "checks",
                        "latency_ms",
                    )
                }
            )
            if not said.get("ok", True):
                problems.append(f"bridge: {said.get('error')}")
            if said.get("outcome") not in step.outcomes:
                problems.append(f"outcome {said.get('outcome')}, wanted {'/'.join(step.outcomes)}")
            if step.reply and not any(
                step.reply.lower() in t.lower() for t in said.get("replies") or []
            ):
                problems.append(f"no reply with “{step.reply}”")
            if step.vimaan_checks and any(not c.get("ok") for c in said.get("checks") or []):
                problems.append("a read-back failed")
        elif step.command:
            done = self.bridge.command(step.command)
            if not done.get("ok"):
                problems.append(f"command: {done.get('error')}")
        if step.settle_sec:
            time.sleep(step.settle_sec)
        r["sim"] = []
        for check in step.sim:
            got = self._check(check)
            r["sim"].append(got)
            if not got["ok"]:
                problems.append(f"{check.what}: got {got['actual']}, wanted {check.expected}")
        if step.callout or step.no_callout:
            spoken = self._spoken_since(started, self.callout_wait_sec)
            r["spoken"] = spoken
            if step.callout and not any(step.callout.lower() in s.lower() for s in spoken):
                problems.append(f"Vimaan didn't say “{step.callout}”")
            if step.no_callout and any(step.no_callout.lower() in s.lower() for s in spoken):
                problems.append(f"Vimaan said “{step.no_callout}”")
        r["problems"] = problems
        r["ok"] = not problems
        r["note"] = step.note
        return r

    def _say(self, phrase: str, r: dict[str, Any]) -> dict[str, Any]:
        if self.voice is None:
            r["heard"], r["heard_ok"] = phrase, None
            return self.bridge.say_text(phrase)
        said = self.bridge.say_wav(voice_mod.synthesize(phrase, self.voice))
        r["heard"] = said.get("heard")
        r["heard_ok"] = said.get("heard") == phrase
        r["understood_ok"] = r["heard_ok"] or bool(
            self.understood is not None and said.get("ok", True) and self.understood(said, phrase)
        )
        if r["understood_ok"]:
            return said
        # Misunderstood: score recognition separately, then do it as text so
        # the check is about the sim, not the recogniser.
        r["misheard"] = {k: said.get(k) for k in ("heard", "intent", "slots", "outcome", "replies")}
        return self.bridge.say_text(phrase)

    def _check(self, check: SimCheck) -> dict[str, Any]:
        deadline = time.monotonic() + check.within_sec
        while True:
            readings = self.bridge.read(*check.datarefs)
            actual = check.value(readings)
            ok = bool(check.ok(actual))
            if ok or time.monotonic() >= deadline:
                return {"what": check.what, "expected": check.expected, "actual": actual, "ok": ok}
            time.sleep(self.poll_sec)

    def _spoken_since(self, since: datetime, wait_sec: float) -> list[str]:
        """What Vimaan said from ``since`` on, after waiting ``wait_sec``."""
        time.sleep(wait_sec)
        events = Path(self.bridge.state()["events"])
        cutoff = since.isoformat(timespec="milliseconds")
        out = []
        for raw in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") == "speak" and str(ev.get("ts", "")) >= cutoff and ev.get("text"):
                out.append(ev["text"])
        return out


def line(r: dict[str, Any]) -> str:
    if r.get("skipped"):
        return f"· {r['step']}: skipped (doesn't apply now)"
    mark = "✓" if r["ok"] else "✕"
    heard = ""
    if r.get("understood_ok") is False:
        heard = f"  [misunderstood “{r.get('heard')}”, redone as text]"
    elif r.get("heard_ok") is False:
        heard = f"  [heard “{r.get('heard')}”, same meaning]"
    detail = "; ".join(r["problems"]) if r["problems"] else " / ".join(r.get("replies") or [])
    return f"{mark} {r['step']}: {detail}{heard}"


def summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    said = [r for r in results if r.get("heard_ok") is not None]
    ran = [r for r in results if not r.get("skipped")]
    return {
        "steps": len(ran),
        "skipped": len(results) - len(ran),
        "passed": sum(r["ok"] for r in ran),
        "failed": [r["step"] for r in ran if not r["ok"]],
        "heard_exactly": sum(bool(r["heard_ok"]) for r in said),
        "understood_by_voice": sum(bool(r.get("understood_ok")) for r in said),
        "said_by_voice": len(said),
    }


def write_report(folder: Path, run: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = (
        folder / f"{run['started'].replace(':', '').replace('-', '')[:15]}_{run['aircraft']}.json"
    )
    path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    return path
