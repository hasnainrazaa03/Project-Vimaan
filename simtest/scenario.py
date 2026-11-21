"""Scenarios: a list of steps run against a live X-Plane through the bridge,
each checked against the sim, and a report (Phase A, A2).

A step either *says* something to Vimaan (in a generated voice, so it goes
through recognition) or does something as the *pilot* (an X-Plane command).
Then it checks:

- Vimaan's outcome, and optionally that a reply contains some words;
- for a question (Tier 4), that the spoken answer says what the sim shows
  (``Answer``: the datarefs are read again right after the reply) and that
  the question changed nothing (``no_actions``: no sim write or command);
- Vimaan's own read-backs (``checks`` in the say result) all passed;
- the sim itself (``SimCheck``): a dataref, or something computed from
  several, must hold, possibly within a deadline (the aircraft has to turn
  or level off first); ``PHASE`` in its datarefs is Vimaan's phase of
  flight (Phase F), from the bridge state;
- optionally a callout Vimaan should (or shouldn't) make.

Recognition and function are scored separately: when the generated voice is
misheard, the step is recorded as a recognition miss and repeated as text,
so a slip of the recogniser doesn't hide whether the sim did the right thing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import voice as voice_mod
from .bridge import Bridge

# Not a dataref: Vimaan's phase of flight now, read from the bridge state.
PHASE = "vimaan:phase"


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


def phase_is(*phases: str, within_sec: float = 0.0) -> SimCheck:
    """Vimaan's phase tracker shows one of ``phases`` (within ``within_sec``)."""
    return SimCheck(
        f"phase {' or '.join(phases)}",
        (PHASE,),
        lambda p: p in phases,
        within_sec=within_sec,
        expected=" or ".join(phases),
    )


def above(what: str, dataref: str, value: float, within_sec: float = 0.0) -> SimCheck:
    """``dataref`` at or above ``value``: a wait for the aircraft (``within_sec``)."""
    return SimCheck(
        what,
        (dataref,),
        lambda v: v is not None and float(v) >= value,
        within_sec=within_sec,
        expected=f"≥ {value:g}",
    )


def below(what: str, dataref: str, value: float, within_sec: float = 0.0) -> SimCheck:
    return SimCheck(
        what,
        (dataref,),
        lambda v: v is not None and float(v) <= value,
        within_sec=within_sec,
        expected=f"≤ {value:g}",
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
class Answer:
    """A spoken answer checked against the sim: ``datarefs`` are read right
    after the reply, and ``ok(reply, readings)`` must hold for one of the
    replies; ``expected(readings)`` says what was wanted, for the report."""

    what: str
    datarefs: tuple[str, ...]
    ok: Callable[[str, dict[str, Any]], bool]
    expected: Callable[[dict[str, Any]], str] = lambda r: ""


@dataclass(frozen=True)
class Step:
    name: str
    say: str | None = None  # a phrase for Vimaan
    command: str | None = None  # or an X-Plane command, as the pilot
    write: tuple[str, float] | None = None  # or a dataref the pilot sets (thrust levers)
    # or something the pilot does with the bridge (simtest.flights: a pitch or
    # an airspeed held from here on); its __name__ goes in the report.
    pilot: Callable[[Any], Any] | None = None
    outcomes: tuple[str, ...] = ("executed",)
    reply: str | None = None  # words one of Vimaan's replies must contain
    sim: tuple[SimCheck, ...] = ()
    callout: str | None = None  # Vimaan must say this (within a few seconds)
    no_callout: str | None = None  # and must not say this
    vimaan_checks: bool = True  # Vimaan's own read-backs must pass
    ask: bool = False  # said on the question key (Shift+Z): only ever answered
    answer: Answer | None = None  # what the spoken answer must say (questions)
    no_actions: bool = False  # the command must not write to or command the sim
    settle_sec: float = 0.0  # wait before checking (a knob turning, a mode settling)
    note: str = ""
    # Run only when this holds now (e.g. above flap speed); else skipped.
    only_if: SimCheck | None = None
    # Run only if the named step ran (wasn't skipped).
    after: str | None = None
    # If this step fails, the rest isn't run: these are, instead (a flight
    # that missed the glideslope turns back west, into the installed scenery).
    abort: tuple[Step, ...] | None = None


@dataclass
class Runner:
    bridge: Bridge
    voice: str | None = voice_mod.DEFAULT_VOICE  # None: text only, no recognition
    # The pilot's own recordings (simtest.record), phrase -> WAV, instead of
    # generated speech. A phrase not recorded is sent as text.
    recordings: Mapping[str, Path] | None = None
    # understood(result, phrase): did a voice result mean what the phrase
    # means? (simtest.understand.same). None: only an exact transcript counts.
    understood: Callable[[dict[str, Any], str], bool] | None = None
    poll_sec: float = 1.0
    callout_wait_sec: float = 4.0
    log: Callable[[str], None] = print
    results: list[dict[str, Any]] = field(default_factory=list)

    def run(self, steps: Sequence[Step]) -> list[dict[str, Any]]:
        ran: set[str] = set()
        aborted: str | None = None
        for step in steps:
            if aborted is not None:
                r = {
                    "step": step.name,
                    "skipped": True,
                    "ok": True,
                    "problems": [],
                    "note": f"not run: {aborted} failed",
                }
                self.results.append(r)
                self.log(line(r))
                continue
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
            if not r["ok"] and step.abort is not None:
                aborted = step.name
                for undo in step.abort:
                    u = self.run_step(undo)
                    self.results.append(u)
                    self.log(line(u))
        return self.results

    # ------------------------------------------------------------------
    def run_step(self, step: Step) -> dict[str, Any]:
        r: dict[str, Any] = {
            "step": step.name,
            "said": step.say,
            "command": step.command,
            "write": step.write,
            "pilot": None if step.pilot is None else getattr(step.pilot, "__name__", "pilot"),
        }
        started = datetime.now()
        problems: list[str] = []
        if step.say:
            # A reading that moves (the 737's EGT with the autothrottle hunting)
            # can leave tolerance between the answer and the read after it, so
            # the sim is read just before the question too.
            before = self._read(step.answer.datarefs) if step.answer is not None else None
            said = self._say(step.say, r, ask=step.ask)
            r.update(
                {
                    k: said.get(k)
                    for k in (
                        "cmd",
                        "intent",
                        "slots",
                        "outcome",
                        "replies",
                        "actions",
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
            if step.no_actions and said.get("actions"):
                problems.append(f"changed the sim: {said['actions']}")
            if step.answer is not None:
                got = self._answer(step.answer, said.get("replies") or [], before or {})
                r["answer"] = got
                if not got["ok"]:
                    problems.append(
                        f"{step.answer.what}: said {got['replies']}, sim shows {got['expected']}"
                        f" (just before the question: {got['expected_before']})"
                    )
        elif step.command:
            done = self.bridge.command(step.command)
            if not done.get("ok"):
                problems.append(f"command: {done.get('error')}")
        elif step.write:
            done = self.bridge.write(*step.write)
            if not done.get("ok"):
                problems.append(f"write: {done.get('error') or done.get('readback')}")
        elif step.pilot is not None:
            step.pilot(self.bridge)
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

    def _say(self, phrase: str, r: dict[str, Any], ask: bool = False) -> dict[str, Any]:
        # The question key only when asked: the talk key's requests stay as they were.
        key = {"ask": True} if ask else {}
        wav: Path | str | None = None
        if self.recordings is not None:
            wav = self.recordings.get(phrase)
            if wav is None:
                r["not_recorded"] = True
        elif self.voice is not None:
            wav = voice_mod.synthesize(phrase, self.voice)
        if wav is None:
            r["heard"], r["heard_ok"] = phrase, None
            return self.bridge.say_text(phrase, **key)
        said = self.bridge.say_wav(wav, **key)
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
        return self.bridge.say_text(phrase, **key)

    def _answer(self, answer: Answer, replies: list[str], before: dict[str, Any]) -> dict[str, Any]:
        """The reply against the sim just after it, or just before the
        question (``before``): the plugin read it in between."""
        readings = self._read(answer.datarefs)  # PHASE too
        return {
            "what": answer.what,
            "replies": replies,
            "readings": readings,
            "readings_before": before,
            "expected": answer.expected(readings),
            "expected_before": answer.expected(before),
            "ok": any(answer.ok(reply, r) for reply in replies for r in (readings, before)),
        }

    def _read(self, datarefs: Sequence[str]) -> dict[str, Any]:
        """The datarefs, and ``PHASE`` from the bridge state."""
        refs = [d for d in datarefs if d != PHASE]
        readings = self.bridge.read(*refs) if refs else {}
        if PHASE in datarefs:
            readings[PHASE] = self.bridge.state().get("phase")
        return readings

    def _check(self, check: SimCheck) -> dict[str, Any]:
        deadline = time.monotonic() + check.within_sec
        while True:
            readings = self._read(check.datarefs)
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
        why = r.get("note") if str(r.get("note", "")).startswith("not run") else "doesn't apply now"
        return f"· {r['step']}: skipped ({why})"
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
