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

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .scenario import SimCheck, Step, above, below, equals, levelled_at, phase_is
from .tier1 import DONE_OR_SET, HSI, SETUPS, _air_start_settles

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
RUDDER = "sim/joystick/yoke_heading_ratio"  # the pedals: + is right
PSI = "sim/flightmodel/position/psi"  # true heading
LAT = "sim/flightmodel/position/latitude"
LON = "sim/flightmodel/position/longitude"
RUNWAY_TRUE_DEG = 118.8  # KPDX 10R as X-Plane lines an aircraft up on it (measured)
THRESHOLD_10R = (45.5951448, -122.6214766)  # X-Plane's apt.dat
FIELD_FT = 31  # KPDX
CENTRELINE_M = 15.0  # the takeoff roll stays this close: 10R is 45 m wide
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
    # Phase L.2: what flies the speed and what the aircraft has.
    autothrottle: bool = False  # flies the speed bug; else the runner's throttle hold
    speed_brakes: bool = False
    retractable_gear: bool = True
    engines: int = 2
    # The autopilot's climb (takeoff and go-around): one the aircraft holds at
    # climb_kt on full power. At 1,500 fpm the C172 (about 700 fpm at most)
    # bled from 75 to 48 kt on full throttle; its autopilot let go at the
    # stall, 650 ft up, and it spun in (2026-11-22).
    climb_fpm: int = 1500
    climb_words: str = "one thousand five hundred"
    # The approach starts in the air on the extended centreline at the
    # approach speed (final_air_start), not X-Plane's own final-approach
    # start: that left the SR22 diving at 4,700 fpm to 400 ft (twice, once
    # into the ground) and the Citation X climbing at 1,500 fpm through the
    # glideslope (2026-11-22). The first four flew X-Plane's start well.
    air_final: bool = False
    final_agl_ft: float = 1500  # the air start's height above the field
    # How far out the air start is. The Citation X and MD-82 come out of it
    # climbing at 1,800-2,800 fpm and level near 3,000 ft above the field a
    # minute later: from 8 nm that was above the glideslope, which can only be
    # captured from below (both armed, neither captured, 2026-11-22). From 14
    # nm they level under it (about 3,500 ft at 11 nm).
    final_nm: float = 8.0
    # Gear, landing flaps and the final speed while level, before the
    # glideslope. The MD-82 ballooned when its flaps came out on the
    # glideslope (+1,100 fpm, 28 at 172 kt), and dived after it to 10 ft
    # over the runway (2026-11-22).
    configure_early: bool = False


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
        autothrottle=True,
        speed_brakes=True,
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
        autothrottle=True,
        speed_brakes=True,
        # The A330's NAV 1 and 2 (its ILS receivers) start off: Vimaan now
        # switches the radio on when it tunes it (flown 2026-11-19: LOC and
        # G/S captured, it flared itself and landed).
    ),
    # The SR22 dives at any start low and slow (3,400 fpm one second after an
    # air start at 100 kt, 800 ft lost in 6 s): it starts 1,000 ft higher. Its
    # autopilot held the altitude at 120 kt but not at 100: the throttle hold
    # then took the power off to keep 100 kt and it sank into the ground
    # (2026-11-22). With one notch its limit is 150 kt.
    # The MD-82 lands on flaps 28 (the MD-80s' usual landing flaps; 40 is for
    # short runways): flaps 40 ballooned it twice, at 172 and 145 kt, and its
    # autopilot dropped the approach modes (2026-11-22).
    # The light singles fly the coupled ILS with one notch, as pilots do: with
    # full flaps at 70 kt the C172's S-Tec 55 couldn't follow the glideslope
    # down and climbed for five minutes on the throttle hold (2026-11-22).
    # Phase L.2 (speeds from each aircraft's manual, flaps and stall speeds
    # from its .acf as Vimaan reads it, X-Plane 12.4.4). The approach speed is
    # under the full-flap limit (its .acf): at 90 kt the C172's "flaps full"
    # (85 kt) was rightly held; the SR22's is 110 kt (2026-11-22). The Citation X and
    # SF50 have no autothrottle: the runner holds the speed on the throttle,
    # as on the Baron.
    "c172": Plane(
        "C172",
        False,
        55,
        7,
        75,
        75,
        70,
        ("flaps down",),
        throttle_from=0.6,
        vs_clean_kt=48,
        retractable_gear=False,
        engines=1,
        climb_fpm=500,
        climb_words="five hundred",
        air_final=True,
    ),
    "sr22": Plane(
        "SR22",
        False,
        73,
        7,
        105,
        120,
        100,
        ("flaps down",),
        throttle_from=0.6,
        vs_clean_kt=74,
        retractable_gear=False,
        engines=1,
        climb_fpm=700,
        climb_words="seven hundred",
        air_final=True,
        final_agl_ft=2500,
    ),
    "citx": Plane(
        "C750",
        True,
        145,
        8,
        200,
        170,
        140,
        ("flaps fifteen", "flaps full"),
        takeoff_flaps="flaps five",
        flaps_up_kt=180,
        go_around_flaps="flaps fifteen",
        throttle_from=0.6,
        pull=0.4,
        vs_clean_kt=135,
        speed_brakes=True,
        air_final=True,
        final_nm=14.0,
    ),
    "md82": Plane(
        "MD82",
        True,
        140,
        8,
        210,
        170,
        145,
        ("flaps twenty eight",),
        takeoff_flaps="flaps eleven",
        flaps_up_kt=190,
        go_around_flaps="flaps fifteen",
        pull=0.4,
        vs_clean_kt=150,
        autothrottle=True,
        speed_brakes=True,
        air_final=True,
        final_nm=14.0,
        configure_early=True,
    ),
    "sf50": Plane(
        "SF50",
        True,
        90,
        8,
        150,
        130,
        100,
        ("flaps full",),
        takeoff_flaps="flaps ten",
        flaps_up_kt=130,
        go_around_flaps="flaps ten",
        throttle_from=0.6,
        vs_clean_kt=75,
        engines=1,
        climb_fpm=1000,
        climb_words="one thousand",
        air_final=True,
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


class RunwayHold:
    """The runner's feet: the rudder on the runway, in a thread.

    With no one on the rudder, the C172 at full power swung 55° left in 6 s
    and left the runway before 20 kt (no wind; the SR22 the same): a single
    propeller's torque and slipstream, which a pilot holds with right rudder.
    The Citation X ran straight (3 m by 100 kt). Proportional on the heading
    error and the distance off the centreline, it held the C172 within 9.6 m
    to 50 kt (2026-11-22, ~/Vimaan_Logs/measuring/l2/takeoff_roll_probe.py).

    Straight through X-Plane's web API, not Vimaan's bridge: a bridge tick
    is about 0.85 s, and the C172 swung 18° in 2 s.

    ``start()`` on the takeoff roll holds the line it starts on (X-Plane
    lines the aircraft up on the centreline) and lets go 1 s after the
    wheels leave the ground. ``start(centreline=False)`` on the rollout
    holds the runway's heading until stopped."""

    period_sec = 0.1
    gain = 0.1  # rudder per degree off the heading
    metres_as_degrees = 0.5  # a metre off the centreline counts as this much heading

    def __init__(self) -> None:
        from . import xplane

        self.lookup: Callable[[str], int] = xplane.dataref_id
        self.read: Callable[[int], Any] = xplane.read_id
        self.write: Callable[[int, Any], None] = xplane.write_id
        self.worst_m: float | None = None  # farthest off the centreline, on the ground
        self.worst_deg: float | None = None  # farthest off the heading, on the ground
        self.heading = RUNWAY_TRUE_DEG
        self._origin: tuple[float, float] | None = None
        self._ids: dict[str, int] = {}
        self._airborne_since: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, centreline: bool = True) -> None:
        self.stop()
        self._ids = {n: self.lookup(n) for n in (PSI, LAT, LON, ON_GROUND, RUDDER)}
        self.worst_m = self.worst_deg = None
        self._airborne_since = None
        if centreline:
            self.heading = float(self.read(self._ids[PSI]))
            self._origin = (float(self.read(self._ids[LAT])), float(self.read(self._ids[LON])))
        else:
            self.heading, self._origin = RUNWAY_TRUE_DEG, None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
            self._centre()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _centre(self) -> None:
        try:
            self.write(self._ids[RUDDER], 0.0)
        except Exception:  # noqa: BLE001 - X-Plane gone: nothing to centre
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.tick(time.monotonic()) is None:
                    self._centre()
                    return
            except Exception:  # noqa: BLE001 - a missed tick; the next one tries again
                pass
            self._stop.wait(self.period_sec)

    def right_of_line_m(self, lat: float, lon: float) -> float:
        """Metres right (+) of the line from the start along the heading."""
        assert self._origin is not None
        lat0, lon0 = self._origin
        north = (lat - lat0) * 111_320.0
        east = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        h = math.radians(self.heading)
        return -north * math.sin(h) + east * math.cos(h)

    def tick(self, now: float) -> float | None:
        """One step: the rudder written, or None once the takeoff is flying."""
        on_ground = bool(self.read(self._ids[ON_GROUND]))
        if self._origin is not None and not on_ground:
            self._airborne_since = self._airborne_since or now
            if now - self._airborne_since >= 1.0:
                return None
        else:
            self._airborne_since = None
        psi = float(self.read(self._ids[PSI]))
        off_deg = ((psi - self.heading + 180) % 360) - 180  # + is right of the heading
        off_m = 0.0
        if self._origin is not None:
            lat, lon = self.read(self._ids[LAT]), self.read(self._ids[LON])
            off_m = self.right_of_line_m(float(lat), float(lon))
        if on_ground:
            self.worst_deg = max(self.worst_deg or 0.0, abs(off_deg))
            if self._origin is not None:
                self.worst_m = max(self.worst_m or 0.0, abs(off_m))
        rudder = round(
            max(-1.0, min(1.0, -self.gain * (off_deg + self.metres_as_degrees * off_m))), 3
        )
        self.write(self._ids[RUDDER], rudder)
        return rudder


@dataclass
class Hands:
    """The runner's hands (and feet) for one flight."""

    speed: SpeedHold
    steer: RunwayHold = field(default_factory=RunwayHold)

    def release(self, bridge: Any) -> None:
        """At the end of a flight, or when it stops: the controls given back
        (the throttle hold and the rudder stopped, the stick in the middle)."""
        self.speed.stop()
        self.steer.stop()
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
def _level(within_sec: float, fpm: float = 300) -> SimCheck:
    return SimCheck(
        "level",
        (VS,),
        lambda v: v is not None and abs(float(v)) <= fpm,
        within_sec=within_sec,
        expected=f"|V/S| ≤ {fpm:.0f} fpm",
    )


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


def _fixed_gear(name: str) -> Step:
    """Fixed gear: "gear up" is refused before any safety rule."""
    return Step(
        name, say="gear up", outcomes=("rejected",), reply="fixed gear", vimaan_checks=False
    )


def _to_glideslope_sec(p: Plane) -> float:
    """Time to wait for the glideslope: from the 8 nm start to about 4.7 nm
    at the approach speed, plus 45 s, and never less than the 150 s the first
    four flew with (the C172 at 75 kt needs 203 s). Not much longer: an
    aircraft that misses it flies on, level, out of the installed scenery."""
    final_nm = p.final_nm if p.air_final else FINAL_NM
    return max(150, round((final_nm - 4.7) / p.approach_kt * 3600 + 45))


# ---- flights ----------------------------------------------------------------
def _climb_sec(p: Plane, climb_ft: float, floor: float) -> float:
    """How long the level-off check waits: the climb at the plane's rate, half
    again for the slow start and the capture, and a minute (at least ``floor``,
    the wait the 1,500 fpm aircraft were flown with). The C172 at 500 fpm was
    at 3,601 ft when a fixed 300 s ran out (2026-11-22)."""
    return max(floor, climb_ft / p.climb_fpm * 60 * 1.5 + 60)


def _takeoff_thrust(hands: Hands, bridge: Any) -> None:
    hands.steer.start()  # the rudder first: full power swings a single left at once
    bridge.write(THROTTLE, 1.0)


def _on_the_runway(hands: Hands) -> SimCheck:
    """The takeoff roll kept within CENTRELINE_M of the centreline."""
    return SimCheck(
        "the farthest off the centreline on the roll",
        (),
        lambda m: m is not None and m <= CENTRELINE_M,
        value=lambda _r: None if hands.steer.worst_m is None else round(hands.steer.worst_m, 1),
        expected=f"≤ {CENTRELINE_M:.0f} m",
    )


def _straight_rollout(hands: Hands) -> SimCheck:
    """The rollout kept within 10° of the runway's heading."""
    return SimCheck(
        "the farthest off the runway heading on the rollout",
        (),
        lambda d: d is not None and d <= 10,
        value=lambda _r: None if hands.steer.worst_deg is None else round(hands.steer.worst_deg, 1),
        expected="≤ 10°",
    )


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
    out = [Step("parked on the runway", sim=(phase_is("parked", within_sec=20),))]
    if p.retractable_gear:
        out += [
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
        ]
    else:
        out.append(_fixed_gear("gear up refused (fixed gear)"))
    out += [
        # Set now: after the rotation the autopilot has to climb at once.
        Step(f"altitude {CRUISE_FT}", say="altitude four thousand", outcomes=DONE_OR_SET),
        Step("heading 103", say=f"heading {COURSE}", outcomes=HEADING_SET),
    ]
    if p.takeoff_flaps:
        # X-Plane's runway start sets the Citation X's flaps 5 itself: "Flaps 5
        # already set" (2026-11-22).
        out.append(
            Step(f"takeoff flaps ({p.takeoff_flaps})", say=p.takeoff_flaps, outcomes=DONE_OR_SET)
        )
    out += [
        Step(
            "parking brake off",
            say="parking brake off",
            outcomes=DONE_OR_SET,
            sim=(below("brake off", PARK, 0.1, within_sec=3),),
        ),
        Step(
            "rudder and takeoff thrust (pilot)",
            pilot=_do("rudder on, takeoff thrust", lambda br: _takeoff_thrust(hands, br)),
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
    if p.speed_brakes:  # the pistons and the SF50 have none
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
            say=f"climb at {p.climb_words} feet per minute",
            reply="to 4000",
        ),
    ]
    # Climb thrust and speed straight away: left at takeoff thrust to 1,000
    # ft, the A330 went through its flaps-1 limit (215 kt) to 242 kt; after
    # the gear came up, still 16 kt through it. The King Air on full throttle
    # levelled off 22 kt over Vne (both recorded by the speed margins).
    if p.autothrottle:
        if p.climb_levers is not None:
            out.append(Step("thrust levers to CL (pilot)", write=(THROTTLE, p.climb_levers)))
        out += [
            Step("autothrottle on", say="auto throttle on", outcomes=DONE_OR_SET),
            Step(f"speed {p.climb_kt}", say=f"speed {_speed_words(p.climb_kt)}"),
        ]
    out += [
        Step("airborne", sim=(equals("off the ground", ON_GROUND, 0, within_sec=20),)),
        Step("stayed on the runway", sim=(_on_the_runway(hands),)),
        Step("initial climb", sim=(phase_is("initial_climb", within_sec=10),)),
        Step("200 ft", sim=(_agl_above(200, within_sec=60),)),
    ]
    if p.retractable_gear:
        out.append(
            Step(
                "gear up with a positive rate",
                say="gear up",
                sim=(below("gear up", GEAR, 0.01, within_sec=20),),
            )
        )
    if not p.autothrottle:  # the runner's hand on the throttle from here
        out.append(
            Step(
                f"throttle held on {p.climb_kt} kt (pilot)",
                pilot=_do(
                    f"speed hold {p.climb_kt}", lambda br: _hold_speed(hands, br, p.climb_kt)
                ),
            )
        )
    if p.flaps_up_kt:  # the jets: flaps up as soon as the speed allows
        out += [
            Step(f"flap speed {p.flaps_up_kt:.0f}", sim=(above("speed", IAS, p.flaps_up_kt, 90),)),
            Step("flaps up", say="flaps up"),
        ]
    out += [
        Step("heading mode", say="heading mode", outcomes=DONE_OR_SET),
        Step("climb", sim=(phase_is("climb", within_sec=120),)),
    ]
    out += [
        Step(
            f"level at {CRUISE_FT}",
            sim=(levelled_at(CRUISE_FT, within_sec=_climb_sec(p, CRUISE_FT, 300)),),
        ),
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
    setup = SETUPS[aircraft]
    out: list[Step] = []
    if setup.settle_first_sec:  # the MD-82 takes no "autopilot on" straight after loading
        out.append(
            Step(
                "the start settles (pilot)",
                pilot=_air_start_settles,
                settle_sec=setup.settle_first_sec,
            )
        )
    out.append(Step("autopilot on", say="autopilot on", outcomes=DONE_OR_SET))
    if p.autothrottle:
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
        # Level before the approach is armed: still climbing at 1,500 fpm, the
        # Citation X went up through the glideslope, captured it climbing and
        # ballooned with full flaps to 20,000 ft (2026-11-22).
        Step("level", sim=(_level(within_sec=60),), abort=TURN_BACK),
    ]
    if p.configure_early:
        out += _landing_configuration(p, hands)
    out += [
        Step("the ILS", say=ILS),
        # The pilot waits for the needles before arming the approach. The
        # C172's S-Tec 55 armed only the localizer, never the glideslope,
        # when "approach" came 0.3 s after the tuning (4 of 4; the needle
        # still near the middle reads as on the glideslope), and once at
        # 3.3 s; 5 s after, it armed every time (2026-11-22).
        Step("course 103", say=f"course {COURSE}", settle_sec=5),
    ]
    if setup.nav_source:  # the Garmin's HSI starts on GPS: "approach" armed a GPS approach
        out.append(
            Step(
                "HSI to NAV 1 (pilot)",
                write=(HSI, 0),
                sim=(equals("HSI on NAV 1", HSI, 0, within_sec=2),),
            )
        )
    out += [
        Step("approach mode", say="approach mode", outcomes=DONE_OR_SET),
        Step(
            "on the glideslope",
            # Captured near 4.7 nm from the 8 nm start: 105 s at 115 kt. Not
            # much longer: an aircraft that misses it flies on, level, out of
            # the installed scenery (X-Plane then pauses under a dialog).
            sim=(equals("glideslope captured", GLIDESLOPE, 2, within_sec=_to_glideslope_sec(p)),),
            abort=TURN_BACK,
        ),
    ]
    if not p.configure_early:
        out += _landing_configuration(p, hands)
    # A failed approach turns back west, level: the Citation X flew on,
    # climbing, out of the installed scenery.
    out.append(Step("approach", sim=(phase_is("approach", within_sec=30),), abort=TURN_BACK))
    # Phase F.4's rules, on the approach (owner's OK 2026-11-19).
    if p.retractable_gear:
        out += _held(
            "gear up on approach",
            "gear up",
            "Confirm gear up, on approach",
            equals("gear still down", GEAR, 1, tol=0.01, within_sec=2),
        )
    from vimaan_nlu.safety import FLAPS_UP_MIN_MARGIN_KT  # ML/ is on the path by now

    slow_below = p.vs_clean_kt + FLAPS_UP_MIN_MARGIN_KT
    if p.final_kt < slow_below:  # the airliners; the others' finals are faster
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


def _landing_configuration(p: Plane, hands: Hands) -> list[Step]:
    """Gear down, the landing flaps, the final speed."""
    out: list[Step] = []
    if p.retractable_gear:
        out.append(
            Step("gear down", say="gear down", sim=(equals("gear down", GEAR, 1, within_sec=20),))
        )
    out += [Step(f"landing flaps ({f})", say=f, settle_sec=3) for f in p.landing_flaps]
    if p.autothrottle:
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
    return out


def landing(aircraft: str, hands: Hands) -> list[Step]:
    """From an 8 nm final: an ILS approach on the autopilot, thrust off before
    touchdown, brakes, stopped. Phases: approach, landing (the last 100 ft),
    rollout, taxi (below 40 kt), parked."""
    p = PLANES[aircraft]
    out = _established(aircraft, hands)
    out.append(Step("landing", sim=(phase_is("landing", within_sec=300),), abort=TURN_BACK))
    if p.autothrottle:  # on its autothrottle the 737 held 145 kt along the runway at 2 ft
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
            "thrust idle, brakes and rudder (pilot)",
            pilot=_do("idle, brakes and rudder", lambda br: _idle_and_brake(hands, br)),
        ),
        Step("autopilot off", say="autopilot off", outcomes=DONE_OR_SET),
        Step("rollout", sim=(phase_is("rollout", within_sec=10),)),
        # The 737 stopped from 40 kt in under 3 s (a phase's dwell): rollout
        # straight to parked.
        Step("taxi speed", sim=(phase_is("taxi", "parked", within_sec=90),)),
        Step("stopped", sim=(phase_is("parked", within_sec=90),)),
        Step("parking brake (pilot)", write=(PARK, 1.0)),
        Step("rolled out straight", sim=(_straight_rollout(hands),)),
    ]
    return out


def _idle(hands: Hands, bridge: Any) -> None:
    hands.speed.stop()
    bridge.write(THROTTLE, 0.0)


def _idle_and_brake(hands: Hands, bridge: Any) -> None:
    _idle(hands, bridge)
    bridge.write(PARK, ROLLOUT_BRAKE)
    hands.steer.start(centreline=False)  # the runway's heading to the stop


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
            say=f"climb at {p.climb_words} feet per minute",
            reply="to 3000",
        ),
        Step("go-around", sim=(phase_is("go_around", within_sec=20),)),
        Step("positive rate", sim=(above("climbing", VS, 300, within_sec=15),)),
    ]
    if p.retractable_gear:
        out.append(
            Step("gear up", say="gear up", sim=(below("gear up", GEAR, 0.01, within_sec=20),))
        )
    out.append(Step(f"flaps ({p.go_around_flaps})", say=p.go_around_flaps))
    if not p.autothrottle:  # left at full throttle the King Air levelled off 24 kt over Vne
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
        Step(
            f"level at {GO_AROUND_TO_FT}",
            sim=(
                levelled_at(
                    GO_AROUND_TO_FT,
                    within_sec=_climb_sec(p, GO_AROUND_TO_FT - GO_AROUND_AT_FT, 240),
                ),
            ),
        ),
        Step("cruise", sim=(phase_is("cruise", within_sec=20),)),
        *TURN_BACK,  # back over the installed scenery
    ]
    return out


def _full_thrust(hands: Hands, bridge: Any) -> None:
    hands.speed.stop()
    bridge.write(THROTTLE, 1.0)


RUNNING = "sim/flightmodel/engine/ENGN_running"


def _engines(what: str, value: int, within_sec: float, engines: int = 2) -> SimCheck:
    refs = tuple(f"{RUNNING}[{i}]" for i in range(engines))
    return SimCheck(
        what,
        refs,
        lambda v: v == (value,) * engines,
        value=lambda r: tuple(r.get(x) for x in refs),
        within_sec=within_sec,
        expected=f"{'both' if engines == 2 else 'all'} {'running' if value else 'stopped'}",
    )


def restart(aircraft: str, hands: Hands) -> list[Step]:
    """On the runway: "shut down both engines", "start both engines", then
    the takeoff (A3, the engines-off start: a cold aircraft's switches are
    its own procedure, so the engines are started warm). One engine: "shut
    down the engine", "start the engine"."""
    p = PLANES[aircraft]
    which = "both engines" if p.engines == 2 else "the engine"
    return [
        Step("parked on the runway", sim=(phase_is("parked", within_sec=20),)),
        Step(
            f"shut down {which}",
            say=f"shut down {which}",
            sim=(_engines("engines stopped", 0, within_sec=30, engines=p.engines),),
        ),
        # As a pilot would: let them wind down before starting again.
        Step("engines winding down", settle_sec=20),
        Step(
            f"start {which}",
            say=f"start {which}",
            outcomes=("pending",),
            vimaan_checks=False,
            # A jet with no starting air starts its APU first (about a minute),
            # then the engines (the A330's FADEC took 35 s to idle).
            sim=(
                _engines("engines running", 1, within_sec=240 if p.jet else 60, engines=p.engines),
            ),
        ),
        *takeoff(aircraft, hands),
    ]


def checklists_on_the_runway(aircraft: str, hands: Hands) -> list[Step]:
    """Phase L.4.6: the checklists on the runway (simtest.checklists)."""
    from .checklists import ground

    return ground(aircraft, hands)


FLIGHTS: dict[str, Callable[[str, Hands], list[Step]]] = {
    "takeoff": takeoff,
    "restart": restart,
    "landing": landing,
    "go-around": go_around,
    "checklists-ground": checklists_on_the_runway,
}
# Flights that start on the runway; the rest on a final.
ON_THE_RUNWAY = ("takeoff", "restart", "checklists-ground")


def cannot_fly(flight: str, aircraft: str) -> str:
    """Why this aircraft can't fly this flight yet, or ""."""
    return PLANES[aircraft].no_restart if flight == "restart" else ""


def start_body(flight: str, aircraft: str) -> dict[str, Any]:
    """Where each flight starts: on the runway, or on an 8 nm final."""
    from . import xplane

    final = None if flight in ON_THE_RUNWAY else FINAL_NM
    if final is not None and PLANES[aircraft].air_final:
        return final_air_start(aircraft)
    return xplane.runway_start_body(aircraft, AIRPORT, RUNWAY, final_nm=final)


def final_point(nm: float = FINAL_NM) -> tuple[float, float]:
    """The extended centreline ``nm`` before the 10R threshold."""
    lat0, lon0 = THRESHOLD_10R
    d = nm * 1852.0
    h = math.radians(RUNWAY_TRUE_DEG)
    lat = lat0 - d * math.cos(h) / 111_320.0
    lon = lon0 - d * math.sin(h) / (111_320.0 * math.cos(math.radians(lat0)))
    return lat, lon


def final_air_start(aircraft: str) -> dict[str, Any]:
    """In the air on the centreline (8 nm out, the Citation X and MD-82 14),
    1,500 ft above the field (the SR22 2,500), heading down the runway at
    the approach speed."""
    from . import xplane

    lat, lon = final_point(PLANES[aircraft].final_nm)
    return xplane.air_start_body(
        aircraft,
        lat,
        lon,
        FIELD_FT + PLANES[aircraft].final_agl_ft,
        RUNWAY_TRUE_DEG,
        PLANES[aircraft].approach_kt,
    )


def new_hands(aircraft: str) -> Hands:
    return Hands(SpeedHold(PLANES[aircraft].throttle_from))
