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
- optionally a callout Vimaan should (or shouldn't) make;
- for a checklist (Phase L.4, ``Follow``), each item as Vimaan reads it, in
  its events: every offer answered as the pilot would, then how it ended.

Recognition and function are scored separately: when the voice is misheard,
the step is recorded as a recognition miss and repeated as text, so a slip of
the recogniser doesn't hide whether the sim did the right thing. A misheard
phrase that Vimaan already acted on is not repeated (a turn or climb "by" would
be flown twice): the sim checks judge what the voice did.
"""

from __future__ import annotations

import json
import os
import sys
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


ANSWERS = ("confirm", "cancel", "wait")
# A confirmed offer's command went ahead ("pending": a knob still turning).
GONE_AHEAD = ("executed", "pending", "no_op")


@dataclass(frozen=True)
class Follow:
    """A checklist the step's phrase started, followed in Vimaan's events
    (``checklist_item``, ``checklist_offer``, ``checklist_end``).

    Each offer is answered "confirm", "cancel" or "wait" (no answer: it runs
    out), by ``answers`` for its item, which may be a function of the bridge
    that picks one (the gear: only below its limit speed), else ``default``.
    Then: every offer settled as answered; what is left open is exactly the
    wrong items not offered and the offers left open; Vimaan says the
    summary for those; and each item set passes its ``sets`` check, read
    from the sim, not from Vimaan."""

    name: str
    answers: Mapping[str, str | Callable[[Any], str]] = field(default_factory=dict)
    default: str = "confirm"
    # Confirmed offers a safety rule holds (gear up on the ground): the hold
    # is cancelled, and the item is left open.
    held: tuple[str, ...] = ()
    # "cancel" said straight after the start, between items: it stops.
    stop: bool = False
    # Asked again: every item the last run of this checklist set reads right.
    again: bool = False
    sets: Mapping[str, SimCheck] = field(default_factory=dict)
    within_sec: float = 300.0


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
    follow: Follow | None = None  # the checklist ``say`` starts (Phase L.4)
    # ``say`` is heard on the open mic, as the wake word listens (Phase L.5):
    # "hey copilot gear down", " ... " a pause. Always as sound (generated or
    # recorded), never as text, and never redone: hearing it is the test.
    wake: bool = False


class Tail:
    """New events from a telemetry file, from ``offset`` on; a line not yet
    written to its end waits for the next read."""

    def __init__(self, path: Path, offset: int) -> None:
        self.path, self.offset, self._rest = path, offset, b""

    def read(self) -> list[dict[str, Any]]:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except OSError:
            return []
        self.offset += len(data)
        *lines, self._rest = (self._rest + data).split(b"\n")
        out = []
        for raw in lines:
            try:
                ev = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                out.append(ev)
        return out


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
    # The last run of each checklist followed (Follow.again), by name.
    checklist_runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    follow_poll_sec: float = 0.25
    summary_wait_sec: float = 6.0  # the summary is said just after the end

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
            # A checklist's items come after its command's result: from here.
            tail = self._tail() if step.follow is not None else None
            said = self._say(step.say, r, ask=step.ask, wake=step.wake)
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
            if step.follow is not None and tail is not None:
                if said.get("outcome") == "executed":
                    r["checklist"] = self._follow(step.follow, tail, problems)
                else:
                    problems.append("the checklist didn't start")
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

    def _say(
        self, phrase: str, r: dict[str, Any], ask: bool = False, wake: bool = False
    ) -> dict[str, Any]:
        if wake:
            return self._hear(phrase, r)
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
        if acted(said):
            # Already done in the sim: again as text, a relative command would
            # be done twice. The sim checks judge what the voice did.
            r["acted_as_heard"] = True
            return said
        return self.bridge.say_text(phrase, **key)

    def _hear(self, phrase: str, r: dict[str, Any]) -> dict[str, Any]:
        """``phrase`` heard on the open mic (the wake word). A wake taken
        answers as a say; nothing woken has the outcome "no_wake". What was
        heard is scored against the command alone (the phrase taken off)."""
        wav: Path | str | None = None
        if self.recordings is not None:
            wav = self.recordings.get(phrase)
            if wav is None:
                r["not_recorded"] = True
        if wav is None:
            wav = voice_mod.synthesize(phrase, self.voice or voice_mod.DEFAULT_VOICE)
        said = dict(self.bridge.hear_wav(wav))
        if said.get("woke") is False:
            said["outcome"] = said.get("outcome") or "no_wake"
        command = wake_command(phrase)
        r["heard"] = said.get("heard")
        r["wake_step"] = True
        if command:
            r["heard_ok"] = said.get("heard") == command
            r["understood_ok"] = r["heard_ok"] or bool(
                self.understood is not None
                and said.get("ok", True)
                and said.get("intent")
                and self.understood(said, command)
            )
        return said

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

    # ------------------------------------------------------------------
    # Checklists (Phase L.4): Follow
    # ------------------------------------------------------------------
    def _tail(self) -> Tail:
        """Vimaan's events from now on."""
        path = Path(self.bridge.state()["events"])
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return Tail(path, size)

    def _follow(self, f: Follow, tail: Tail, problems: list[str]) -> dict[str, Any]:
        """The checklist ``f`` from its start to its spoken summary."""
        rec: dict[str, Any] = {
            "name": f.name,
            "skipped": [],
            "items": [],
            "answers": [],
            "offers": {},
            "end": None,
            "summary": None,
        }
        if f.stop:  # at once: the first item isn't read yet
            rec["answers"].append(self._reply(f, None, "cancel", problems))
        deadline = time.monotonic() + f.within_sec
        summary_words: str | None = None
        while True:
            for ev in tail.read():
                kind = ev.get("kind")
                if kind == "speak":
                    if summary_words is not None and ev.get("text") == summary_words:
                        rec["summary"] = summary_words
                    continue
                if ev.get("checklist") != f.name or rec["end"] is not None:
                    continue
                if kind == "checklist_start":
                    rec["skipped"] = list(ev.get("skipped") or [])
                elif kind == "checklist_item":
                    item = {k: ev.get(k) for k in ("item", "state", "result", "said")}
                    rec["items"].append(item)
                    # Stopping: an offer read before the "cancel" isn't answered.
                    if item["result"] == "offered" and not f.stop:
                        how = self._pick(f, str(item["item"]))
                        rec["answers"].append(self._reply(f, str(item["item"]), how, problems))
                elif kind == "checklist_offer":
                    rec["offers"][ev.get("item")] = ev.get("result")
                elif kind == "checklist_end":
                    rec["end"] = {k: ev.get(k) for k in ("reason", "open", "read", "items")}
                    if ev.get("reason") == "done":
                        summary_words = _checklist_words().summary(
                            f.name, list(ev.get("open") or [])
                        )
                        deadline = min(deadline, time.monotonic() + self.summary_wait_sec)
            if rec["end"] is not None and (summary_words is None or rec["summary"] is not None):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(self.follow_poll_sec)
        self._judge(f, rec, summary_words, problems)
        self.checklist_runs[f.name] = rec
        return rec

    def _pick(self, f: Follow, item: str) -> str:
        how = f.answers.get(item, f.default)
        answer = how(self.bridge) if callable(how) else how
        if answer not in ANSWERS:
            raise ValueError(f"{item}: answer {answer!r}, not one of {ANSWERS}")
        return answer

    def _reply(
        self, f: Follow, item: str | None, answer: str, problems: list[str]
    ) -> dict[str, Any]:
        """An offer answered (``item``), or the checklist cancelled (None)."""
        a: dict[str, Any] = {"item": item, "answer": answer}
        if answer == "wait":
            return a
        said = self._say(answer, a)
        outcome, replies = said.get("outcome"), list(said.get("replies") or [])
        a.update(outcome=outcome, replies=replies)
        if answer == "cancel":
            want = "Cancelled" if item is not None else "Checklist cancelled"
            if outcome != "cancelled" or want not in replies:
                what = item or "the checklist"
                problems.append(f"{what}: “cancel” → {outcome} {replies}, wanted “{want}”")
            return a
        if outcome == "safety_hold":
            if item not in f.held:
                problems.append(f"{item}: held by a safety rule: {replies[-1:]}")
            undo = self._say("cancel", {})  # the hold, not the checklist
            a["hold"] = {"outcome": undo.get("outcome"), "replies": undo.get("replies")}
            if undo.get("outcome") != "cancelled":
                problems.append(f"{item}: the hold wasn't cancelled: {undo.get('replies')}")
        elif item in f.held:
            problems.append(f"{item}: “confirm” → {outcome} {replies}, wanted a safety hold")
        elif outcome not in GONE_AHEAD:
            problems.append(f"{item}: “confirm” → {outcome} {replies}")
        return a

    def _judge(
        self, f: Follow, rec: dict[str, Any], summary_words: str | None, problems: list[str]
    ) -> None:
        end = rec["end"]
        if end is None:
            problems.append(f"the {f.name} checklist didn't end within {f.within_sec:.0f} s")
            return
        want = "cancelled" if f.stop else "done"
        if end["reason"] != want:
            problems.append(f"it ended {end['reason']}, wanted {want}")
        if f.stop:
            return
        if summary_words is not None and rec["summary"] is None:
            problems.append(f"Vimaan didn't say “{summary_words}”")
        for a in rec["answers"]:
            item = a["item"]
            gone = a["answer"] == "confirm" and a.get("outcome") in GONE_AHEAD
            want_offer = "set" if gone else "open"
            got = rec["offers"].get(item)
            if got != want_offer:
                problems.append(
                    f"{item}: the offer ended {got or 'unsettled'}, wanted {want_offer}"
                )
        # Left open: the wrong items it couldn't offer, and the offers not set.
        left = [i["item"] for i in rec["items"] if i["result"] == "open"]
        left += [item for item, result in rec["offers"].items() if result == "open"]
        if sorted(end["open"] or []) != sorted(left):
            problems.append(f"open {end['open']}, wanted {left}")
        rec["sim"] = []
        for item, check in f.sets.items():
            if rec["offers"].get(item) == "set":
                got = self._check(check)
                rec["sim"].append(got)
                if not got["ok"]:
                    problems.append(f"{check.what}: got {got['actual']}, wanted {check.expected}")
        if f.again:
            last = self.checklist_runs.get(f.name)
            if last is None:
                problems.append(f"asked again, but the {f.name} checklist hadn't run")
                return
            now = {i["item"]: i["result"] for i in rec["items"]}
            for item, result in last["offers"].items():
                if result == "set" and now.get(item) != "checked":
                    problems.append(f"{item}: set last time, now {now.get(item) or 'not read'}")


WAKE_PHRASE = "hey copilot"


def wake_command(phrase: str) -> str:
    """The command in a wake step's phrase: "hey copilot ... gear down" ->
    "gear down"; "" when there is none (the wake word alone, or talk
    without it)."""
    words = phrase.replace(" ... ", " ").replace(",", " ").split()
    wake = WAKE_PHRASE.split()
    return " ".join(words[len(wake) :]) if words[: len(wake)] == wake else ""


def _checklist_words() -> Any:
    """vimaan_nlu.checklists (pure Python): what Vimaan says at a checklist's end."""
    ml = str(Path(__file__).resolve().parents[1] / "ML")
    if ml not in sys.path:
        sys.path.insert(0, ml)
    from vimaan_nlu import checklists

    return checklists


# Outcomes after which Vimaan has changed the sim, or is changing it (the
# A330's heading knob is "pending" while it clicks).
ACTED = ("executed", "pending")


def acted(result: Mapping[str, Any]) -> bool:
    """Did this say result change the sim (or start to)?"""
    return result.get("outcome") in ACTED or bool(result.get("actions"))


def line(r: dict[str, Any]) -> str:
    if r.get("skipped"):
        why = r.get("note") if str(r.get("note", "")).startswith("not run") else "doesn't apply now"
        return f"· {r['step']}: skipped ({why})"
    mark = "✓" if r["ok"] else "✕"
    heard = ""
    if r.get("acted_as_heard"):
        heard = f"  [misunderstood “{r.get('heard')}”, acted on as heard]"
    elif r.get("understood_ok") is False:
        redone = "" if r.get("wake_step") else ", redone as text"
        heard = f"  [misunderstood “{r.get('heard')}”{redone}]"
    elif r.get("heard_ok") is False:
        heard = f"  [heard “{r.get('heard')}”, same meaning]"
    detail = "; ".join(r["problems"]) if r["problems"] else " / ".join(r.get("replies") or [])
    run = r.get("checklist")
    if run and not r["problems"]:
        answered = ", ".join(f"{a['item']} {a['answer']}" for a in run["answers"] if a["item"])
        detail = run.get("summary") or f"ended {(run.get('end') or {}).get('reason')}"
        detail += f" ({answered})" if answered else ""
    return f"{mark} {r['step']}: {detail}{heard}"


def _said(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything said by voice: each step's phrase, and a checklist's answers."""
    out = []
    for r in results:
        out.append(r)
        out += (r.get("checklist") or {}).get("answers", [])
    return [s for s in out if s.get("heard_ok") is not None]


def summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    said = _said(results)
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
