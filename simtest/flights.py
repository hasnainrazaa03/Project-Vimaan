"""Whole flights for Phase F.3: a takeoff, a landing and a go-around at
Portland (KPDX runway 10R), flown with no one at the controls, so Vimaan's
phase tracker and speed margins are recorded on real takeoffs and landings
(the tier checklists fly only in the air). Each step checks the phase
Vimaan's tracker shows.

Vimaan does what it can be asked to: the parking brake, gear, flaps,
autopilot and its modes, the autothrottle, the ILS. The runner flies the
rest, as the pilot (docs/IN_SIM_TESTING.md, "Flights"):

- the thrust levers (takeoff thrust, idle before touchdown, a go-around);
- the stick for the rotation: pulled at Vr until the nose is up, then
  let go, and the autopilot climbs straight after;
- the throttle on the Baron and King Air, which have no autothrottle: held
  on an airspeed (:class:`SpeedHold`);
- the brakes after touchdown (the parking brake lever, partly on).

Learned flying them (2026-11-19):

- X-Plane's final-approach start is about 1,500 ft above the ground with the
  gear and flaps up, autopilot off, whatever the distance; at 8 nm the
  glideslope is captured from below near 4.7 nm.
- The localizer captures with the course set (103° magnetic: X-Plane's nav
  data gives 37199.113 = 103 × 360 + 119.113° true). With the heading only,
  the Baron's HSI never captured it.
- The 737 flies the glideslope well on one autopilot and flares a little by
  itself; engaging the second autopilot late (950 ft) dived it into the
  ground. On its autothrottle it then holds the approach speed along the
  runway at 2 ft, so the thrust comes off before touchdown.
- A fixed piston throttle doesn't hold the approach speed: at 30 % with the
  gear down the Baron slowed to 81 kt, its autopilot let go near the stall,
  and it flew into the ground.
- The bridge answers every quarter second, so a read and a write take
  about 0.6 s: too slow to hold a pitch on the stick (the Baron, "held" at
  8°, flew back into the runway from 75 ft). After an open-loop pull the
  trimmed aircraft keeps climbing until the autopilot takes over; the 737
  and the Baron engage their autopilots right after liftoff.
- The A330's NAV 1 and 2 (its ILS receivers) start switched off, so its
  approach never captured; Vimaan now switches a radio on when it tunes it.
- "shut down both engines" left the 737 and King Air running (fixed: the
  737's own cutoff, a turboprop's condition lever). Starts: the King Air
  needs its igniters, a jet the APU's air (Vimaan starts the APU first), and
  the A330 its mode selector and masters (its FADEC starts it).
- A flight that misses the glideslope and flies on east leaves the
  installed scenery; X-Plane pauses under a dialog only a click closes. A
  missed glideslope now turns the aircraft back west (``TURN_BACK``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .scenario import SimCheck, Step, above, below, equals, levelled_at, phase_is
from .tier1 import DONE_OR_SET

# A heading: the A330 and the GFC-700 aircraft turn the knob over frames
# ("pending", heading_by_knob).
HEADING_SET = (*DONE_OR_SET, "pending")

AIRPORT, RUNWAY = "KPDX", "10R"
FINAL_NM = 8.0
ILS = "tune nav one to one one zero decimal five"  # IPDX 110.50, ILS CAT III
COURSE = "one zero three"  # magnetic (see the module docstring)

THROTTLE = "sim/cockpit2/engine/actuators/throttle_ratio_all"
YOKE = "sim/joystick/yoke_pitch_ratio"
PITCH = "sim/flightmodel/position/theta"
IAS = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"
VS = "sim/flightmodel/position/vh_ind_fpm"
AGL_M = "sim/flightmodel/position/y_agl"
ON_GROUND = "sim/flightmodel/failures/onground_any"
GEAR = "sim/flightmodel2/gear/deploy_ratio[0]"
PARK = "sim/flightmodel/controls/parkbrake"
SPEEDBRAKE = "sim/cockpit2/controls/speedbrake_ratio"  # armed: below 0
FLAPS = "sim/cockpit2/controls/flap_handle_request_ratio"
GLIDESLOPE = "sim/cockpit2/autopilot/glideslope_status"
FEET = 0.3048
ROLLOUT_BRAKE = 0.4  # the parking brake lever, partly on: brakes after touchdown
CRUISE_FT = 4000  # where the takeoff levels off
GO_AROUND_AT_FT = 500  # above the ground
GO_AROUND_TO_FT = 3000


@dataclass(frozen=True)
class Plane:
    """How each aircraft is flown (numbers from the .acf and from flying it)."""

    icao: str
    jet: bool
    rotate_kt: float  # Vr: the runner pulls the stick
    rotate_pitch: float  # and lets go at this pitch
    climb_kt: int  # after takeoff: the jets' speed bug, the pistons' throttle hold
    approach_kt: int  # until the glideslope
    final_kt: int  # on the glideslope, landing flaps out
    landing_flaps: tuple[str, ...]  # said in order on the glideslope
    takeoff_flaps: str | None = None
    flaps_up_kt: float = 0.0  # jets: flaps up from this speed in the climb
    go_around_flaps: str = "flaps up"
    throttle_from: float = 0.5  # the throttle hold starts here (pistons)
    climb_levers: float | None = None  # A330: the CL detent after takeoff
    pull: float = 0.3  # the stick (yoke pitch ratio) for the rotation
    # The autopilot engages from this height (an Airbus not below 100 ft; its
    # flight controls hold the climb until then).
    autopilot_from_ft: float = 0.0
    # Why "shut down both engines" then "start both engines" can't be flown
    # on this aircraft yet (the restart flight is refused), or "".
    no_restart: str = ""
    # The clean stall speed in its .acf (the plugin's stall_clean_kt, as the
    # session events showed it, X-Plane 12.4.4): "flaps up" below this + 10 kt
    # is held. 0: not known, nothing to check.
    vs_clean_kt: float = 0.0


PLANES: dict[str, Plane] = {
    "baron": Plane(
        "BE58",
        False,
        85,
        7,
        120,
        115,
        95,
        ("flaps down", "flaps full"),
        throttle_from=0.4,
        vs_clean_kt=84,
    ),
    "kingair": Plane(
        "BE9L",
        False,
        100,
        7,
        140,
        130,
        115,
        ("flaps down", "flaps full"),
        throttle_from=0.4,
        vs_clean_kt=100,
    ),
    "737": Plane(
        "B738",
        True,
        145,
        8,
        210,
        160,
        145,
        ("flaps fifteen", "flaps thirty"),
        takeoff_flaps="flaps five",
        flaps_up_kt=190,
        go_around_flaps="flaps fifteen",
        pull=0.4,
        vs_clean_kt=139,
    ),
    "a330": Plane(
        "A333",
        True,
        145,
        8,
        210,
        170,
        # Not 140: its FCU won't select below VLS (the dial went to 143.6 and
        # 144.6 kt in CONF FULL, 2026-11-19).
        150,
        ("flaps two", "flaps three", "flaps full"),
        takeoff_flaps="flaps one",
        flaps_up_kt=200,
        go_around_flaps="flaps three",
        climb_levers=0.53,
        pull=0.4,
        autopilot_from_ft=100,
        vs_clean_kt=175,
        # The A330's NAV 1 and 2 (its ILS receivers) start off: Vimaan now
        # switches the radio on when it tunes it (flown 2026-11-19: LOC and
        # G/S captured, it flared itself and landed).
    ),
}

SPOKEN = {
    95: "nine five",
    115: "one one five",
    130: "one three zero",
    140: "one four zero",
    145: "one four five",
    150: "one five zero",
    160: "one six zero",
    170: "one seven zero",
    210: "two one zero",
}


# ---- the runner's own hands -------------------------------------------------
class SpeedHold:
    """The throttle held on an airspeed (the Baron and King Air have no
    autothrottle), in a thread: PI with damping on the airspeed's rate. A
    fixed integral alone swung ±17 kt with a minute's period; this held the
    King Air ±3 kt in level flight. A tick is a read and a write through the
    bridge (about 0.85 s with the wait), so the rate is over the real
    interval (0.25 per kt/s: the damping it had when the rate was taken over
    0.25 s)."""

    period_sec = 0.25

    def __init__(self, throttle_from: float = 0.5) -> None:
        self.throttle_from = throttle_from
        self.target_kt = 0.0
        self.bridge: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._integral = throttle_from
        self._prev: tuple[float, float] | None = None  # (time, airspeed)
        self._rate = 0.0

    def start(self, bridge: Any) -> None:
        self.stop()
        self.bridge, self._stop = bridge, threading.Event()
        self._integral, self._prev, self._rate = self.throttle_from, None, 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick(time.monotonic())
            except Exception:  # noqa: BLE001 - a missed tick; the next one tries again
                pass
            self._stop.wait(self.period_sec)

    def tick(self, now: float) -> float | None:
        """One step: the throttle written, or None with no airspeed."""
        ias = self.bridge.read(IAS).get(IAS)
        if ias is None:
            return None
        ias = float(ias)
        if self._prev is not None and now > self._prev[0]:
            rate = (ias - self._prev[1]) / (now - self._prev[0])
            self._rate = 0.7 * self._rate + 0.3 * rate
        self._prev = (now, ias)
        err = self.target_kt - ias
        self._integral = min(1.0, max(0.0, self._integral + 0.0005 * err))
        throttle = round(min(1.0, max(0.0, self._integral + 0.02 * err - 0.25 * self._rate)), 3)
        self.bridge.write(THROTTLE, throttle)
        return throttle


@dataclass
class Hands:
    """The runner's hands for one flight."""

    speed: SpeedHold

    def release(self, bridge: Any) -> None:
        """At the end of a flight, or when it stops: the controls given back
        (the throttle hold stopped, the stick in the middle)."""
        self.speed.stop()
        try:
            bridge.write(YOKE, 0.0)
        except Exception:  # noqa: BLE001 - X-Plane gone: nothing to give back
            pass


def _do(name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    fn.__name__ = name  # the report says what the pilot did
    return fn


# If the approach can't go on: back west, level, over the installed scenery.
TURN_BACK = (
    Step("turn back west", say="heading two eight three", outcomes=HEADING_SET),
    Step("hold the altitude", say="altitude hold", outcomes=DONE_OR_SET),
)


# ---- checks -----------------------------------------------------------------
def _agl_below(ft: float, within_sec: float) -> SimCheck:
    return SimCheck(
        f"below {ft:.0f} ft",
        (AGL_M,),
        lambda v: v is not None and v <= ft,
        value=lambda r: None if r.get(AGL_M) is None else round(float(r[AGL_M]) / FEET),
        within_sec=within_sec,
        expected=f"≤ {ft:.0f} ft above the ground",
    )


def _agl_above(ft: float, within_sec: float) -> SimCheck:
    return SimCheck(
        f"above {ft:.0f} ft",
        (AGL_M,),
        lambda v: v is not None and v >= ft,
        value=lambda r: None if r.get(AGL_M) is None else round(float(r[AGL_M]) / FEET),
        within_sec=within_sec,
        expected=f"≥ {ft:.0f} ft above the ground",
    )


def _speed_words(kt: int) -> str:
    return SPOKEN[kt]


# ---- flights ----------------------------------------------------------------
def _held(name: str, say: str, reply: str, still: SimCheck, **kw: Any) -> list[Step]:
    """A phrase a safety rule holds ("Confirm …"), then "cancel": nothing
    changed. ``kw`` (only_if) is for the first step; the cancel runs only
    after it."""
    first = f"{name} is held"
    return [
        Step(
            first,
            say=say,
            outcomes=("safety_hold",),
            reply=reply,
            vimaan_checks=False,
            **kw,
        ),
        Step(
            f"cancel ({name})",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            sim=(still,),
            after=first,
        ),
    ]


def takeoff(aircraft: str, hands: Hands) -> list[Step]:
    """From the runway, engines running and parked: brake off, takeoff
    thrust, rotate at Vr, autopilot, gear up with a positive rate, climb and
    level off at 4,000 ft. Phases: parked, taxi (the first 40 kt), takeoff
    roll, initial climb, climb, cruise."""
    p = PLANES[aircraft]
    out = [
        Step("parked on the runway", sim=(phase_is("parked", within_sec=20),)),
        Step(
            "gear up on the ground is held",
            say="gear up",
            outcomes=("safety_hold",),
            reply="Confirm gear up",
            vimaan_checks=False,
        ),
        Step(
            "cancel the gear",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            sim=(equals("gear still down", GEAR, 1, tol=0.01, within_sec=2),),
        ),
        # Set now: after the rotation the autopilot has to climb at once.
        Step(f"altitude {CRUISE_FT}", say="altitude four thousand", outcomes=DONE_OR_SET),
        Step("heading 103", say=f"heading {COURSE}", outcomes=HEADING_SET),
    ]
    if p.takeoff_flaps:
        out.append(Step(f"takeoff flaps ({p.takeoff_flaps})", say=p.takeoff_flaps))
    out += [
        Step(
            "parking brake off",
            say="parking brake off",
            outcomes=DONE_OR_SET,
            sim=(below("brake off", PARK, 0.1, within_sec=3),),
        ),
        Step(
            "takeoff thrust (pilot)",
            write=(THROTTLE, 1.0),
            sim=(phase_is("taxi", "takeoff_roll", within_sec=20),),
            note="the first 40 kt are a taxi",
        ),
        Step("takeoff roll", sim=(phase_is("takeoff_roll", within_sec=40),)),
        Step(
            "parking brake at speed is held",
            say="parking brake on",
            outcomes=("safety_hold",),
            reply="Confirm parking brake",
            vimaan_checks=False,
        ),
        Step(
            "cancel the parking brake",
            say="cancel",
            outcomes=("cancelled",),
            vimaan_checks=False,
            sim=(below("brake still off", PARK, 0.1, within_sec=2),),
        ),
    ]
    if p.jet:  # the Baron and King Air have none
        out += _held(
            "speed brake on the takeoff roll",
            "extend the speed brakes",
            "Confirm speed brake, takeoff roll",
            below("speed brake still in", SPEEDBRAKE, 0.05, within_sec=2),
        )
    out += [
        Step(f"accelerate to Vr {p.rotate_kt:.0f} kt", sim=(above("Vr", IAS, p.rotate_kt, 90),)),
        Step(
            "rotate (pilot)",
            write=(YOKE, p.pull),
            sim=(above(f"nose up {p.rotate_pitch:.0f}°", PITCH, p.rotate_pitch, 15),),
        ),
        Step("stick released (pilot)", write=(YOKE, 0.0)),
    ]
    if p.autopilot_from_ft:
        out.append(
            Step(f"{p.autopilot_from_ft:.0f} ft", sim=(_agl_above(p.autopilot_from_ft, 30),))
        )
    out += [
        Step("autopilot on", say="autopilot on", outcomes=DONE_OR_SET),
        Step(
            "climb on the autopilot",
            say="climb at one thousand five hundred feet per minute",
            reply="to 4000",
        ),
    ]
    # Climb thrust and speed straight away: left at takeoff thrust to 1,000
    # ft, the A330 went through its flaps-1 limit (215 kt) to 242 kt; after
    # the gear came up, still 16 kt through it. The King Air on full throttle
    # levelled off 22 kt over Vne (both recorded by the speed margins).
    if p.jet:
        if p.climb_levers is not None:
            out.append(Step("thrust levers to CL (pilot)", write=(THROTTLE, p.climb_levers)))
        out += [
            Step("autothrottle on", say="auto throttle on", outcomes=DONE_OR_SET),
            Step(f"speed {p.climb_kt}", say=f"speed {_speed_words(p.climb_kt)}"),
        ]
    out += [
        Step("airborne", sim=(equals("off the ground", ON_GROUND, 0, within_sec=20),)),
        Step("initial climb", sim=(phase_is("initial_climb", within_sec=10),)),
        Step("200 ft", sim=(_agl_above(200, within_sec=60),)),
        Step(
            "gear up with a positive rate",
            say="gear up",
            sim=(below("gear up", GEAR, 0.01, within_sec=20),),
        ),
    ]
    if p.jet:  # flaps up as soon as the speed allows
        out += [
            Step(f"flap speed {p.flaps_up_kt:.0f}", sim=(above("speed", IAS, p.flaps_up_kt, 90),)),
            Step("flaps up", say="flaps up"),
        ]
    else:
        out.append(
            Step(
                f"throttle held on {p.climb_kt} kt (pilot)",
                pilot=_do(
                    f"speed hold {p.climb_kt}", lambda br: _hold_speed(hands, br, p.climb_kt)
                ),
            )
        )
    out += [
        Step("heading mode", say="heading mode", outcomes=DONE_OR_SET),
        Step("climb", sim=(phase_is("climb", within_sec=120),)),
    ]
    out += [
        Step(f"level at {CRUISE_FT}", sim=(levelled_at(CRUISE_FT, within_sec=300),)),
        Step("cruise", sim=(phase_is("cruise", within_sec=20),)),
        # Left flying east after the run, the A330 reached the end of the
        # installed scenery and X-Plane paused under its dialog (2026-11-19).
        *TURN_BACK,
    ]
    return out


def _hold_speed(hands: Hands, bridge: Any, kt: float) -> None:
    hands.speed.target_kt = kt
    if not hands.speed.running:
        hands.speed.start(bridge)


def _established(aircraft: str, hands: Hands) -> list[Step]:
    """From the final-approach start to the glideslope with the landing
    configuration: autopilot, the ILS, approach mode, gear and flaps."""
    p = PLANES[aircraft]
    out = [Step("autopilot on", say="autopilot on", outcomes=DONE_OR_SET)]
    if p.jet:
        out += [
            Step("autothrottle on", say="auto throttle on", outcomes=DONE_OR_SET),
            Step(f"speed {p.approach_kt}", say=f"speed {_speed_words(p.approach_kt)}"),
        ]
    else:
        out.append(
            Step(
                f"throttle held on {p.approach_kt} kt (pilot)",
                pilot=_do(
                    f"speed hold {p.approach_kt}",
                    lambda br: _hold_speed(hands, br, p.approach_kt),
                ),
            )
        )
    out += [
        Step("altitude hold", say="altitude hold", outcomes=DONE_OR_SET),
        Step("heading 103", say=f"heading {COURSE}", outcomes=HEADING_SET),
        Step("heading mode", say="heading mode", outcomes=DONE_OR_SET),
        Step("the ILS", say=ILS),
        Step("course 103", say=f"course {COURSE}"),
        Step("approach mode", say="approach mode", outcomes=DONE_OR_SET),
        Step(
            "on the glideslope",
            # Captured near 4.7 nm from the 8 nm start: 105 s at 115 kt. Not
            # much longer: an aircraft that misses it flies on, level, out of
            # the installed scenery (X-Plane then pauses under a dialog).
            sim=(equals("glideslope captured", GLIDESLOPE, 2, within_sec=150),),
            abort=TURN_BACK,
        ),
        Step("gear down", say="gear down", sim=(equals("gear down", GEAR, 1, within_sec=20),)),
    ]
    out += [Step(f"landing flaps ({f})", say=f, settle_sec=3) for f in p.landing_flaps]
    if p.jet:
        out.append(Step(f"speed {p.final_kt}", say=f"speed {_speed_words(p.final_kt)}"))
    else:
        out.append(
            Step(
                f"throttle held on {p.final_kt} kt (pilot)",
                pilot=_do(
                    f"speed hold {p.final_kt}", lambda br: _hold_speed(hands, br, p.final_kt)
                ),
            )
        )
    out.append(Step("approach", sim=(phase_is("approach", within_sec=30),)))
    # Phase F.4's rules, on the approach (owner's OK 2026-11-19).
    out += _held(
        "gear up on approach",
        "gear up",
        "Confirm gear up, on approach",
        equals("gear still down", GEAR, 1, tol=0.01, within_sec=2),
    )
    from vimaan_nlu.safety import FLAPS_UP_MIN_MARGIN_KT  # ML/ is on the path by now

    slow_below = p.vs_clean_kt + FLAPS_UP_MIN_MARGIN_KT
    if p.final_kt < slow_below:  # the jets; the pistons' finals are faster
        out += _held(
            "flaps up when slow",
            "flaps up",
            "Confirm flaps up, slow",
            above("flaps still out", FLAPS, 0.3, within_sec=2),
            # Said only when slow enough to be held, with 2 kt to spare: an
            # aircraft still slowing would have its flaps taken up.
            only_if=below("slow enough", IAS, slow_below - 2, within_sec=30),
        )
    return out


def landing(aircraft: str, hands: Hands) -> list[Step]:
    """From an 8 nm final: an ILS approach on the autopilot, thrust off before
    touchdown, brakes, stopped. Phases: approach, landing (the last 100 ft),
    rollout, taxi (below 40 kt), parked."""
    p = PLANES[aircraft]
    out = _established(aircraft, hands)
    out.append(Step("landing", sim=(phase_is("landing", within_sec=300),)))
    if p.jet:  # on its autothrottle the 737 held 145 kt along the runway at 2 ft
        out += [
            Step("50 ft", sim=(_agl_below(50, within_sec=30),)),
            Step("autothrottle off", say="auto throttle off", outcomes=DONE_OR_SET),
            Step("thrust idle (pilot)", write=(THROTTLE, 0.0)),
        ]
    else:  # on the throttle hold the Baron floated 30 s at 2-4 ft (2026-11-19)
        out += [
            Step("50 ft", sim=(_agl_below(50, within_sec=30),)),
            Step("thrust idle (pilot)", pilot=_do("idle", lambda br: _idle(hands, br))),
        ]
    out += [
        Step("touchdown", sim=(equals("on the ground", ON_GROUND, 1, within_sec=40),)),
        Step(
            "thrust idle and brakes (pilot)",
            pilot=_do("idle and brakes", lambda br: _idle_and_brake(hands, br)),
        ),
        Step("autopilot off", say="autopilot off", outcomes=DONE_OR_SET),
        Step("rollout", sim=(phase_is("rollout", within_sec=10),)),
        # The 737 stopped from 40 kt in under 3 s (a phase's dwell): rollout
        # straight to parked.
        Step("taxi speed", sim=(phase_is("taxi", "parked", within_sec=90),)),
        Step("stopped", sim=(phase_is("parked", within_sec=90),)),
        Step("parking brake (pilot)", write=(PARK, 1.0)),
    ]
    return out


def _idle(hands: Hands, bridge: Any) -> None:
    hands.speed.stop()
    bridge.write(THROTTLE, 0.0)


def _idle_and_brake(hands: Hands, bridge: Any) -> None:
    _idle(hands, bridge)
    bridge.write(PARK, ROLLOUT_BRAKE)


def go_around(aircraft: str, hands: Hands) -> list[Step]:
    """From an 8 nm final: the same approach, then at 500 ft full thrust and
    a climb on the autopilot to 3,000 ft, gear and flaps up. Phases:
    approach, go-around, climb, cruise."""
    p = PLANES[aircraft]
    out = _established(aircraft, hands)
    out += [
        Step(f"{GO_AROUND_AT_FT} ft", sim=(_agl_below(GO_AROUND_AT_FT, within_sec=300),)),
        Step(
            "go-around thrust (pilot)",
            pilot=_do("full thrust", lambda br: _full_thrust(hands, br)),
        ),
        Step(f"altitude {GO_AROUND_TO_FT}", say="altitude three thousand", outcomes=DONE_OR_SET),
        Step(
            "climb away",
            say="climb at one thousand five hundred feet per minute",
            reply="to 3000",
        ),
        Step("go-around", sim=(phase_is("go_around", within_sec=20),)),
        Step("positive rate", sim=(above("climbing", VS, 300, within_sec=15),)),
        Step("gear up", say="gear up", sim=(below("gear up", GEAR, 0.01, within_sec=20),)),
        Step(f"flaps ({p.go_around_flaps})", say=p.go_around_flaps),
    ]
    if not p.jet:  # left at full throttle the King Air levelled off 24 kt over Vne
        out.append(
            Step(
                f"throttle held on {p.climb_kt} kt (pilot)",
                pilot=_do(
                    f"speed hold {p.climb_kt}", lambda br: _hold_speed(hands, br, p.climb_kt)
                ),
            )
        )
    out += [
        Step("climb", sim=(phase_is("climb", within_sec=120),)),
        Step(f"level at {GO_AROUND_TO_FT}", sim=(levelled_at(GO_AROUND_TO_FT, within_sec=240),)),
        Step("cruise", sim=(phase_is("cruise", within_sec=20),)),
        *TURN_BACK,  # back over the installed scenery
    ]
    return out


def _full_thrust(hands: Hands, bridge: Any) -> None:
    hands.speed.stop()
    bridge.write(THROTTLE, 1.0)


RUNNING = "sim/flightmodel/engine/ENGN_running"


def _engines(what: str, value: int, within_sec: float) -> SimCheck:
    return SimCheck(
        what,
        (f"{RUNNING}[0]", f"{RUNNING}[1]"),
        lambda v: v == (value, value),
        value=lambda r: (r.get(f"{RUNNING}[0]"), r.get(f"{RUNNING}[1]")),
        within_sec=within_sec,
        expected=f"both {'running' if value else 'stopped'}",
    )


def restart(aircraft: str, hands: Hands) -> list[Step]:
    """On the runway: "shut down both engines", "start both engines", then
    the takeoff (A3, the engines-off start: a cold aircraft's switches are
    its own procedure, so the engines are started warm)."""
    return [
        Step("parked on the runway", sim=(phase_is("parked", within_sec=20),)),
        Step(
            "shut down both engines",
            say="shut down both engines",
            sim=(_engines("engines stopped", 0, within_sec=30),),
        ),
        # As a pilot would: let them wind down before starting again.
        Step("engines winding down", settle_sec=20),
        Step(
            "start both engines",
            say="start both engines",
            outcomes=("pending",),
            vimaan_checks=False,
            # A jet with no starting air starts its APU first (about a minute),
            # then the engines (the A330's FADEC took 35 s to idle).
            sim=(_engines("engines running", 1, within_sec=240 if PLANES[aircraft].jet else 60),),
        ),
        *takeoff(aircraft, hands),
    ]


FLIGHTS: dict[str, Callable[[str, Hands], list[Step]]] = {
    "takeoff": takeoff,
    "restart": restart,
    "landing": landing,
    "go-around": go_around,
}


def cannot_fly(flight: str, aircraft: str) -> str:
    """Why this aircraft can't fly this flight yet, or ""."""
    return PLANES[aircraft].no_restart if flight == "restart" else ""


def start_body(flight: str, aircraft: str) -> dict[str, Any]:
    """Where each flight starts: on the runway, or on an 8 nm final."""
    from . import xplane

    final = None if flight in ("takeoff", "restart") else FINAL_NM
    return xplane.runway_start_body(aircraft, AIRPORT, RUNWAY, final_nm=final)


def new_hands(aircraft: str) -> Hands:
    return Hands(SpeedHold(PLANES[aircraft].throttle_from))
