"""The Tier 4 checklist (PHASE_PLAN, Tier 4 part C) as scenarios: questions
asked on the question key (Shift+Z), in cruise with the autopilot flying like
Tiers 1-3, each spoken answer checked against the sim read just after it.

Needs a question model (q1 or later) beside the command model: the runner
refuses to start without one.

An answer passes when it names what was asked ("Altitude ...", "Fuel ...")
and its value is what the sim shows, within what can move between the reply
and the read (altitude in hold 60 ft, speeds 5 kt, heading 3 degrees, ...).
The units are the cockpit's, taken from the read-only probe of 2026-11-16 and
not from the plugin's own tables, so a wrong table is caught: fuel in US
gallons on the Baron, pounds on the King Air and 737, kilograms on the A330;
the altimeter in inches or hPa as the 737's and A330's own selectors show;
outside air temperature in Celsius; height above ground from the radio
altimeter dataref, which every aircraft reports.

Every question must change nothing in the sim (no write, no command). A
command said on the question key is refused with the sim unchanged, and on
the talk key a status form the command model won't act on is answered.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vimaan_nlu import answers
from vimaan_nlu.questions import REPLY as REFUSED
from vimaan_nlu.readback import spell_digits

from .scenario import PHASE, Answer, SimCheck, Step, equals
from .tier1 import SETUPS, hands_off
from .tier3 import _lights

G = "sim/cockpit2/gauges/indicators/"
AP = "sim/cockpit2/autopilot/"
ENG = "sim/cockpit2/engine/indicators/"
RADIO = "sim/cockpit2/radios/actuators/"
ALT = G + "altitude_ft_pilot"
IAS = G + "airspeed_kts_pilot"
HDG = G + "heading_AHARS_deg_mag_pilot"
VVI = G + "vvi_fpm_pilot"
GS = G + "ground_speed_kt"
MACH = G + "mach_pilot"
AGL = G + "radio_altimeter_height_ft_pilot"
WIND_DIR = G + "wind_heading_deg_mag"
WIND_KT = G + "wind_speed_kts"
OAT = "sim/cockpit2/temperature/outside_air_temp_degc"
BARO_IN = "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot"
BARO_STD = "sim/cockpit2/gauges/actuators/barometer_setting_is_std_pilot"
FUEL = "sim/flightmodel/weight/m_fuel_total"
FUEL_FLOW = ENG + "fuel_flow_kg_sec"
COM1, COM1_SBY = RADIO + "com1_frequency_hz_833", RADIO + "com1_standby_frequency_hz_833"
NAV1, NAV1_SBY = RADIO + "nav1_frequency_hz", RADIO + "nav1_standby_frequency_hz"
XPDR_CODE, XPDR_MODE = RADIO + "transponder_code", RADIO + "transponder_mode"
SERVOS = AP + "servos_on"
HDG_STATUS = AP + "heading_status"
ALT_HOLD = AP + "altitude_hold_status"
HDG_DIAL = AP + "heading_dial_deg_mag_pilot"
ALT_DIAL = AP + "altitude_dial_ft"
GEAR = "sim/flightmodel2/gear/deploy_ratio"
GEAR_HANDLE = "sim/cockpit2/controls/gear_handle_down"
FLAPS = "sim/cockpit2/controls/flap_handle_request_ratio"
WPT_NM = "sim/cockpit2/radios/indicators/gps_dme_distance_nm"
WPT_MIN = "sim/cockpit2/radios/indicators/gps_dme_time_min"
DEST = "sim/cockpit/gps/destination_index"
ZULU = "sim/time/zulu_time_sec"
ENGINES = 2  # every checklist aircraft is a twin
# The question model's intent the runner looks for before it starts.
NEEDS_QUESTION_INTENT = "ask_flight_data"
# X-Plane's transponder_mode as the plugin answers it (the Baron and King Air;
# the 737 and A330 answer from their own selectors).
XPDR_MODE_SPOKEN = {
    0: "off",
    1: "standby",
    2: "on",
    3: "altitude",
    5: "on",
    6: "altitude",
    7: "altitude",
}


@dataclass(frozen=True)
class Cockpit:
    """What each aircraft's own gauges show (the probe of 2026-11-16)."""

    fuel_unit: str  # the word the answer uses
    kg_per_unit: float  # X-Plane's m_fuel_total is kilograms on every aircraft
    engine: str  # piston / turboprop / jet
    baro_hpa: tuple[str, float] | None  # the selector dataref and its hPa value
    xpdr_from_xplane: bool  # the mode word comes from X-Plane's transponder_mode


COCKPITS: dict[str, Cockpit] = {
    "baron": Cockpit("gallons", 6.0 * answers.KG_PER_LB, "piston", None, True),
    "kingair": Cockpit("pounds", answers.KG_PER_LB, "turboprop", None, True),
    "737": Cockpit(
        "pounds",
        answers.KG_PER_LB,
        "jet",
        ("laminar/B738/EFIS_control/capt/baro_in_hpa", 1.0),
        False,
    ),
    "a330": Cockpit(
        "kilograms", 1.0, "jet", ("laminar/A333/barometer/capt_inHg_hPa_pos", 1.0), False
    ),
}

# ---------------------------------------------------------------------------
# Reading a spoken answer
# ---------------------------------------------------------------------------
_DIGITS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "niner": "9",
    "decimal": ".",
}
THREE = r"(\w+ \w+ \w+)"  # a heading said digit by digit


def spoken_number(words: str) -> float | None:
    """Digits said one by one ("two niner niner two", "one two one decimal
    niner") as a number; None when a word isn't a digit."""
    said = "".join(_DIGITS.get(w, "?") for w in words.split())
    try:
        return float(said)
    except ValueError:
        return None


def _match(pattern: str, reply: str) -> tuple[str, ...] | None:
    m = re.fullmatch(pattern, reply.strip())
    return m.groups() if m else None


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _near(said: float | None, actual: Any, tol: float) -> bool:
    a = _num(actual)
    return said is not None and a is not None and abs(said - a) <= tol


def _angle_near(said: float | None, actual: Any, tol: float) -> bool:
    a = _num(actual)
    return said is not None and a is not None and abs((said - a + 180) % 360 - 180) <= tol


def _show(value: Any, step: float = 1.0) -> str:
    v = _num(value)
    return "?" if v is None else f"{round(v / step) * step:g}"


# ---------------------------------------------------------------------------
# Answers, each against the datarefs the plugin reads for it
# ---------------------------------------------------------------------------
def altitude_answer(tol: float = 60.0) -> Answer:
    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(r"Altitude (-?\d+) feet", reply)
        return g is not None and _near(float(g[0]), r.get(ALT), tol)

    return Answer(
        "altitude", (ALT,), ok, lambda r: f"Altitude {_show(r.get(ALT), 10)} feet ± {tol:g}"
    )


def speed_answer(word: str, dataref: str, tol: float = 5.0) -> Answer:
    """ "Airspeed 165 knots", "Ground speed 285 knots"."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(rf"{word} (\d+) knots", reply)
        return g is not None and _near(float(g[0]), r.get(dataref), tol)

    return Answer(
        word.lower(), (dataref,), ok, lambda r: f"{word} {_show(r.get(dataref))} knots ± {tol:g}"
    )


def heading_answer(tol: float = 3.0) -> Answer:
    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(rf"Heading {THREE}", reply)
        return g is not None and _angle_near(spoken_number(g[0]), r.get(HDG), tol)

    return Answer("heading", (HDG,), ok, lambda r: f"Heading {_show(r.get(HDG))} ± {tol:g}°")


def vertical_speed_answer(tol: float = 250.0) -> Answer:
    """ "Level" under 100 ft/min either way, else climbing / descending."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        v = _num(r.get(VVI))
        if v is None:
            return False
        if reply.strip() == "Level":
            return abs(v) < 100 + tol
        g = _match(r"(Climbing|Descending) (\d+) feet per minute", reply)
        if g is None:
            return False
        said = float(g[1]) * (1 if g[0] == "Climbing" else -1)
        return abs(said - v) <= tol

    return Answer(
        "vertical speed", (VVI,), ok, lambda r: f"{_show(r.get(VVI), 50)} ft/min ± {tol:g}"
    )


def wind_answer(dir_tol: float = 10.0, kt_tol: float = 3.0) -> Answer:
    def ok(reply: str, r: dict[str, Any]) -> bool:
        kt = _num(r.get(WIND_KT))
        if kt is None or r.get(WIND_DIR) is None:
            return False
        if reply.strip() == "Wind calm":
            return kt < 2 + kt_tol
        g = _match(rf"Wind {THREE} at (\d+) knots", reply)
        if g is None or not _near(float(g[1]), kt, kt_tol):
            return False
        return kt < 5 or _angle_near(spoken_number(g[0]), r.get(WIND_DIR), dir_tol)

    return Answer(
        "wind",
        (WIND_DIR, WIND_KT),
        ok,
        lambda r: f"Wind {_show(r.get(WIND_DIR))}° at {_show(r.get(WIND_KT))} kt",
    )


def temperature_answer(tol: float = 1.0) -> Answer:
    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(r"Outside air temperature (minus )?(\d+) degrees Celsius", reply)
        if g is None:
            return False
        return _near(-float(g[1]) if g[0] else float(g[1]), r.get(OAT), tol)

    return Answer(
        "outside air temperature", (OAT,), ok, lambda r: f"{_show(r.get(OAT))} °C ± {tol:g}"
    )


def altimeter_answer(cockpit: Cockpit) -> Answer:
    """Inches ("Altimeter two niner niner two"), hPa when the aircraft's own
    selector shows hPa ("Q N H one zero one three"), or "standard"."""
    unit = cockpit.baro_hpa
    refs = (BARO_IN, BARO_STD) + ((unit[0],) if unit else ())

    def hpa(r: dict[str, Any]) -> bool:
        return unit is not None and _num(r.get(unit[0])) == unit[1]

    def ok(reply: str, r: dict[str, Any]) -> bool:
        inches = _num(r.get(BARO_IN))
        if inches is None:
            return False
        if _num(r.get(BARO_STD)) == 1:
            return reply.strip() == "Altimeter standard"
        if hpa(r):
            g = _match(r"Q N H ([a-z ]+)", reply)
            return g is not None and _near(spoken_number(g[0]), inches * 33.8639, 1.0)
        g = _match(r"Altimeter ([a-z ]+)", reply)
        return g is not None and _near(spoken_number(g[0]), inches * 100, 1.0)

    def expected(r: dict[str, Any]) -> str:
        inches = _num(r.get(BARO_IN))
        if inches is None:
            return "?"
        return answers.altimeter(inches, _num(r.get(BARO_STD)) == 1, hpa(r))

    return Answer("altimeter", refs, ok, expected)


def mach_answer(tol: float = 0.01) -> Answer:
    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(r"Mach point (\w+ \w+)", reply)
        n = spoken_number(g[0]) if g else None
        return n is not None and _near(n / 100, r.get(MACH), tol)

    return Answer("mach", (MACH,), ok, lambda r: f"Mach {_show(r.get(MACH), 0.01)} ± {tol:g}")


def phase_answer() -> Answer:
    """The phase of flight Vimaan's tracker shows (Phase F.4, from q2)."""

    def said(r: dict[str, Any]) -> str:
        return answers.phase(r.get(PHASE))

    return Answer("phase of flight", (PHASE,), lambda reply, r: reply == said(r), said)


def height_answer() -> Answer:
    """The radio altimeter dataref, on every aircraft (the Baron has none in
    its panel, X-Plane reports it anyway). The terrain moves under a cruising
    aircraft: 100 ft or 5%."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(r"(\d+) feet above the ground", reply)
        agl = _num(r.get(AGL))
        return g is not None and agl is not None and abs(float(g[0]) - agl) <= max(100, agl * 0.05)

    return Answer(
        "height above ground", (AGL,), ok, lambda r: f"{_show(r.get(AGL), 10)} ft ± 100 or 5%"
    )


def fuel_answer(cockpit: Cockpit) -> Answer:
    """In the cockpit's unit, rounded as a gauge is read (gallons to 1,
    pounds and kilograms to 10, to 100 from 10,000); 1% for what burns."""

    def want(r: dict[str, Any]) -> float | None:
        kg = _num(r.get(FUEL))
        return None if kg is None else kg / cockpit.kg_per_unit

    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(rf"Fuel (\d+) {cockpit.fuel_unit}", reply)
        w = want(r)
        if g is None or w is None:
            return False
        step = 1 if cockpit.fuel_unit == "gallons" else (100 if w >= 10000 else 10)
        return abs(float(g[0]) - w) <= step + 0.01 * w

    return Answer(
        "fuel",
        (FUEL,),
        ok,
        lambda r: f"Fuel {_show(want(r))} {cockpit.fuel_unit} (from {_show(r.get(FUEL))} kg)",
    )


def _flows() -> tuple[str, ...]:
    return tuple(f"{FUEL_FLOW}[{i}]" for i in range(ENGINES))


def _minutes_left(r: dict[str, Any]) -> float | None:
    kg = _num(r.get(FUEL))
    flow = sum(_num(r.get(name)) or 0.0 for name in _flows())
    return None if kg is None or flow <= 1e-6 else kg / flow / 60


def endurance_answer() -> Answer:
    """At the present fuel flow, which moves: 2 minutes and 10%."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        left = _minutes_left(r)
        if reply.startswith("No fuel flow"):
            return left is None
        g = _match(r"About (?:(\d+) hours? )?(\d+) minutes of fuel at this fuel flow", reply)
        if g is None or left is None:
            return False
        said = int(g[0] or 0) * 60 + int(g[1])
        return abs(said - left) <= 2 + 0.1 * left

    return Answer(
        "fuel endurance",
        (FUEL, *_flows()),
        ok,
        lambda r: f"{_show(_minutes_left(r))} minutes at this fuel flow ± 10%",
    )


def engine_answer(
    reading: str,
    dataref: str,
    unit: str,
    tol: float,
    convert: Callable[[float], float] = lambda v: v,
) -> Answer:
    """ "Oil pressure 60 P S I on both engines", or engine by engine
    ("Engine 1 N1 85 percent, engine 2 N1 86 percent")."""
    name = answers.READING_NAMES[reading]
    refs = tuple(f"{dataref}[{i}]" for i in range(ENGINES))

    def values(r: dict[str, Any]) -> list[float] | None:
        got = [_num(r.get(x)) for x in refs]
        return None if any(v is None for v in got) else [convert(v) for v in got if v is not None]

    def ok(reply: str, r: dict[str, Any]) -> bool:
        want = values(r)
        text = reply.strip()
        if want is None or not text.lower().startswith((name.lower(), "engine 1")):
            return False
        said = [float(x) for x in re.findall(rf"(\d+(?:\.\d+)?) {unit}", text)]
        if len(said) == 1 and text.endswith("on both engines"):
            return all(abs(said[0] - w) <= tol for w in want)
        return len(said) == len(want) and all(abs(s - w) <= tol for s, w in zip(said, want))

    def expected(r: dict[str, Any]) -> str:
        want = values(r)
        shown = "?" if want is None else " / ".join(f"{w:.1f}" for w in want)
        return f"{name} {shown} {unit} ± {tol:g}"

    return Answer(name, refs, ok, expected)


def _exact(what: str, refs: tuple[str, ...], say: Callable[[dict[str, Any]], str]) -> Answer:
    """For readings that don't move (radios, dials): the reply is exactly
    what the plugin's wording (vimaan_nlu.answers) makes of the sim."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        try:
            return reply.strip() == say(r)
        except (TypeError, ValueError):
            return False

    def expected(r: dict[str, Any]) -> str:
        try:
            return say(r)
        except (TypeError, ValueError):
            return f"? (read {r})"

    return Answer(what, refs, ok, expected)


def com1_answer(which: str | None = None) -> Answer:
    return _exact(
        "COM 1" + (f" {which}" if which else ""),
        (COM1, COM1_SBY),
        lambda r: answers.radio("COM 1", int(r[COM1]), int(r[COM1_SBY]), 1000, which),
    )


def nav1_answer() -> Answer:
    return _exact(
        "NAV 1",
        (NAV1, NAV1_SBY),
        lambda r: answers.radio("NAV 1", int(r[NAV1]), int(r[NAV1_SBY]), 100, None),
    )


def transponder_answer(cockpit: Cockpit) -> Answer:
    """The code always; the mode word from X-Plane's mode where the plugin
    reads it there (the 737 and A330 read their own selectors)."""

    def code(r: dict[str, Any]) -> str:
        return f"Squawk {spell_digits(f'{int(r[XPDR_CODE]):04d}')}"

    def ok(reply: str, r: dict[str, Any]) -> bool:
        try:
            said = code(r)
            mode = XPDR_MODE_SPOKEN.get(int(r[XPDR_MODE]))
        except (KeyError, TypeError, ValueError):
            return False
        text = reply.strip()
        if cockpit.xpdr_from_xplane:
            return text == (f"{said}, transponder {mode}" if mode else said)
        return text == said or text.startswith(f"{said}, transponder ")

    def expected(r: dict[str, Any]) -> str:
        try:
            return f"{code(r)} (mode {r.get(XPDR_MODE)})"
        except (KeyError, TypeError, ValueError):
            return "?"

    return Answer("transponder", (XPDR_CODE, XPDR_MODE), ok, expected)


def _set_feet(r: dict[str, Any]) -> str:
    return f"altitude set {int(round(float(r[ALT_DIAL]) / 10) * 10)} feet"


def autopilot_answer() -> Answer:
    """ "Autopilot on, heading two seven zero, altitude hold, altitude set
    6000 feet": on / off as the servos are, and each active mode named."""
    refs = (SERVOS, HDG_STATUS, ALT_HOLD, HDG_DIAL, ALT_DIAL)

    def ok(reply: str, r: dict[str, Any]) -> bool:
        try:
            on = bool(int(r[SERVOS]))
            parts = [p.strip() for p in reply.split(",")]
            if parts[0] != ("Autopilot on" if on else "Autopilot off"):
                return False
            if int(r[HDG_STATUS]) == 2:
                if f"heading {answers.three_digits(float(r[HDG_DIAL]))}" not in parts:
                    return False
            if int(r[ALT_HOLD]) == 2 and "altitude hold" not in parts:
                return False
            return not on or _set_feet(r) in parts
        except (KeyError, TypeError, ValueError):
            return False

    def expected(r: dict[str, Any]) -> str:
        return (
            f"servos {r.get(SERVOS)}, heading mode {r.get(HDG_STATUS)}, "
            f"altitude hold {r.get(ALT_HOLD)}, bug {r.get(HDG_DIAL)}, set {r.get(ALT_DIAL)}"
        )

    return Answer("autopilot status", refs, ok, expected)


def altitude_set_answer() -> Answer:
    return _exact("altitude set", (ALT_DIAL,), lambda r: _set_feet(r).capitalize())


def heading_bug_answer() -> Answer:
    return _exact(
        "heading bug",
        (HDG_DIAL,),
        lambda r: f"Heading bug {answers.three_digits(float(r[HDG_DIAL]))}",
    )


def _gear_words(r: dict[str, Any]) -> str:
    """As the plugin says it (every checklist aircraft's gear retracts)."""
    ratios = [v for v in (_num(r.get(f"{GEAR}[{i}]")) for i in range(3)) if v is not None]
    if ratios and all(x >= 0.99 for x in ratios):
        return "Gear down"
    if ratios and all(x <= 0.01 for x in ratios):
        return "Gear up"
    return "Gear moving down" if _num(r.get(GEAR_HANDLE)) == 1 else "Gear moving up"


def gear_answer() -> Answer:
    return _exact("gear", (*(f"{GEAR}[{i}]" for i in range(3)), GEAR_HANDLE), _gear_words)


def flaps_answer() -> Answer:
    """ "Flaps up" with the handle at 0; else any notch the cockpit labels."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        ratio = _num(r.get(FLAPS))
        if ratio is None:
            return False
        text = reply.strip()
        return (
            text == "Flaps up" if ratio < 0.01 else text.startswith("Flaps ") and text != "Flaps up"
        )

    return Answer("flaps", (FLAPS,), ok, lambda r: f"flap handle {_show(r.get(FLAPS), 0.01)}")


def waypoint_answer(tol_nm: float = 2.0) -> Answer:
    """No flight plan on a checklist flight: "No waypoint set". With one,
    the distance within 2 nm (it shrinks as we fly)."""

    def no_plan(r: dict[str, Any]) -> bool:
        dest, nm = _num(r.get(DEST)), _num(r.get(WPT_NM))
        return dest is None or dest < 0 or nm is None or nm <= 0

    def ok(reply: str, r: dict[str, Any]) -> bool:
        if no_plan(r):
            return reply.strip() == "No waypoint set"
        g = _match(r"Next waypoint (\d+) miles?(?:, \d+ minutes?)?", reply)
        return g is not None and _near(float(g[0]), r.get(WPT_NM), tol_nm)

    return Answer(
        "next waypoint",
        (WPT_NM, WPT_MIN, DEST),
        ok,
        lambda r: "No waypoint set" if no_plan(r) else f"{_show(r.get(WPT_NM))} nm ± {tol_nm:g}",
    )


def time_answer() -> Answer:
    """Zulu, read after the reply: this minute or the one before."""

    def ok(reply: str, r: dict[str, Any]) -> bool:
        t = _num(r.get(ZULU))
        return t is not None and reply.strip() in (
            answers.clock(t, "zulu"),
            answers.clock(t - 60, "zulu"),
        )

    return Answer(
        "time",
        (ZULU,),
        ok,
        lambda r: answers.clock(_num(r.get(ZULU)) or 0.0, "zulu") + " (or the minute before)",
    )


def status_answer() -> Answer:
    """ "Altitude 6000 feet, heading two seven zero, airspeed 150 knots,
    autopilot on"."""
    refs = (ALT, HDG, IAS, SERVOS)

    def ok(reply: str, r: dict[str, Any]) -> bool:
        g = _match(
            rf"Altitude (-?\d+) feet, heading {THREE}, airspeed (\d+) knots, autopilot (on|off)",
            reply,
        )
        if g is None:
            return False
        servos = _num(r.get(SERVOS))
        return (
            _near(float(g[0]), r.get(ALT), 60)
            and _angle_near(spoken_number(g[1]), r.get(HDG), 3)
            and _near(float(g[2]), r.get(IAS), 5)
            and servos is not None
            and (g[3] == "on") == bool(servos)
        )

    return Answer(
        "status",
        refs,
        ok,
        lambda r: (
            f"alt {_show(r.get(ALT), 10)}, hdg {_show(r.get(HDG))}, "
            f"ias {_show(r.get(IAS))}, servos {r.get(SERVOS)}"
        ),
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def _ask(
    name: str,
    say: str,
    answer: Answer | None = None,
    reply: str | None = None,
    ask: bool = True,
    note: str = "",
) -> Step:
    """A question, answered (never acted on), and the sim unchanged."""
    return Step(
        name,
        say=say,
        ask=ask,
        outcomes=("answered",),
        answer=answer,
        reply=reply,
        no_actions=True,
        vimaan_checks=False,
        note=note,
    )


N1_NOTE = (
    "q1 tags 'n 1' (number words become digits); the postprocessor reads it back "
    "as 'n one', the plugin's name (2026-11-18)"
)


def _engine_steps(kind: str) -> list[Step]:
    """Oil pressure on every aircraft, and what its engine kind shows."""
    out = [
        _ask(
            "oil pressure",
            "what's the oil pressure",
            engine_answer("oil pressure", ENG + "oil_pressure_psi", "P S I", 3.0),
        )
    ]
    n1 = _ask(
        "N1",
        "what's the n one",
        engine_answer("n one", ENG + "N1_percent", "percent", 1.5),
        note=N1_NOTE,
    )
    if kind == "piston":
        out.append(
            _ask(
                "manifold pressure",
                "what's the manifold pressure",
                engine_answer("manifold pressure", ENG + "MPR_in_hg", "inches", 0.5),
            )
        )
    elif kind == "turboprop":
        out += [
            _ask(
                "torque",
                "what's the torque",
                engine_answer(
                    "torque",
                    ENG + "torque_n_mtr",
                    "foot pounds",
                    40.0,
                    lambda v: v * answers.NM_PER_FT,
                ),
            ),
            n1,
        ]
    else:
        out += [
            _ask(
                "E G T",
                "what's the e g t",
                engine_answer("e g t", ENG + "EGT_deg_C", "degrees", 10.0),
            ),
            n1,
        ]
    return out


def gear_still() -> SimCheck:
    """The gear stays where it is in cruise: the handle up."""
    return equals("gear handle still up", GEAR_HANDLE, 0)


def _landing_lights_on(aircraft: str) -> Step:
    """Set on the talk key first (Tier 3's step, its checks), so the answer
    has a known state to say."""
    return next(s for s in _lights(aircraft) if s.name == "landing lights on")


def steps(aircraft: str) -> list[Step]:
    if aircraft not in SETUPS:
        raise KeyError(aircraft)
    c = COCKPITS[aircraft]
    out = hands_off(aircraft) + [_landing_lights_on(aircraft)]
    out += [
        _ask("altitude", "what's our altitude", altitude_answer()),
        _ask("airspeed", "what's our airspeed", speed_answer("Airspeed", IAS)),
        _ask("heading", "what's the heading", heading_answer()),
        _ask("vertical speed", "what's our vertical speed", vertical_speed_answer()),
        _ask("ground speed", "what's our ground speed", speed_answer("Ground speed", GS)),
        _ask("wind", "what's the wind", wind_answer()),
        _ask("outside temperature", "what's the outside temperature", temperature_answer()),
        _ask("altimeter", "what's the altimeter setting", altimeter_answer(c)),
        _ask("height above ground", "how high are we above the ground", height_answer()),
        _ask("phase of flight", "what phase of flight are we in", phase_answer()),
    ]
    if c.engine == "jet":
        out.append(_ask("mach", "what's our mach number", mach_answer()))
    out += [
        _ask("fuel", "how much fuel do we have", fuel_answer(c)),
        _ask("fuel endurance", "how long will our fuel last", endurance_answer()),
        *_engine_steps(c.engine),
        _ask("COM 1", "what's com one", com1_answer()),
        _ask("COM 1 standby", "what's com one standby", com1_answer("standby")),
        _ask("NAV 1", "what's nav one", nav1_answer()),
        _ask("transponder", "what's our squawk", transponder_answer(c)),
        _ask("autopilot status", "autopilot status", autopilot_answer()),
        _ask("altitude set", "what altitude is set", altitude_set_answer()),
        _ask("heading bug", "what's the heading bug", heading_bug_answer()),
        _ask("landing lights", "are the landing lights on", reply="Landing lights on"),
        _ask("gear", "is the gear down", gear_answer()),
        _ask("flaps", "where are the flaps", flaps_answer()),
        _ask("next waypoint", "how far to the next waypoint", waypoint_answer()),
        _ask("time", "what time is it", time_answer()),
        _ask("status", "status report", status_answer()),
        # A command on the question key is refused, and nothing moves.
        Step(
            "gear down on the question key is refused",
            say="gear down",
            ask=True,
            outcomes=("non_actionable",),
            reply=REFUSED,
            no_actions=True,
            vimaan_checks=False,
            sim=(gear_still(),),
        ),
        Step(
            "autopilot off on the question key is refused",
            say="autopilot off",
            ask=True,
            outcomes=("non_actionable",),
            reply=REFUSED,
            no_actions=True,
            vimaan_checks=False,
            sim=(equals("autopilot still on", SERVOS, 1),),
        ),
        # On the talk key: a status form the command model won't act on is
        # the question model's to answer, and so is a question.
        _ask(
            "landing lights status on the talk key",
            "landing lights status",
            reply="Landing lights on",
            ask=False,
        ),
        _ask("a question on the talk key", "what's our altitude", altitude_answer(), ask=False),
    ]
    return out


# Which steps ask which question family (the tests check every family is asked).
FAMILIES = {
    "ask_flight_data": (
        "altitude",
        "airspeed",
        "heading",
        "vertical speed",
        "ground speed",
        "wind",
        "outside temperature",
        "altimeter",
        "height above ground",
        "mach",
        "phase of flight",
    ),
    "ask_fuel": ("fuel",),
    "ask_fuel_time": ("fuel endurance",),
    "ask_engine": ("oil pressure", "manifold pressure", "torque", "E G T", "N1"),
    "ask_com": ("COM 1", "COM 1 standby"),
    "ask_nav": ("NAV 1",),
    "ask_transponder": ("transponder",),
    "ask_autopilot": ("autopilot status", "altitude set", "heading bug"),
    "ask_switch": ("landing lights", "gear", "flaps"),
    "ask_waypoint": ("next waypoint",),
    "ask_time": ("time",),
    "ask_status_generic": ("status",),
}
