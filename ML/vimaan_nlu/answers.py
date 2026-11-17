"""Spoken answers to Tier 4 questions: readings in, words out.

The plugin reads the sim and hands the numbers here; nothing in this module
touches X-Plane, so every answer is unit-tested. Each answer names what it
answers ("Altitude 7400 feet", "Fuel 23000 pounds"), so a question that was
misheard as another is obvious from the reply.

Units are the ones the cockpit shows, found in the sim on 2026-11-16 (read-only
probe of the four aircraft, docs/AIRCRAFT_PROFILES.md):
- fuel: Baron US gallons (its fuel-flow gauge reads GAL/HR), King Air pounds,
  737 pounds (its display's total 23.0 = X-Plane's 10,439 kg), A330 kilograms;
- the altimeter in the unit the aircraft's own selector shows (737 and A330:
  inches unless switched to hPa), "standard" when STD is set;
- outside air temperature in Celsius everywhere (the King Air's gauge reads
  Fahrenheit; Celsius is what ATC and the airliners use).

Pure Python: no torch, no X-Plane.
"""

from __future__ import annotations

from collections.abc import Sequence

from vimaan_nlu.readback import spell_digits

KG_PER_LB = 0.45359237
# Fuel densities for a volume answer (gallons): avgas 6.0 lb/US gal, jet fuel 6.7.
KG_PER_GALLON = {"avgas": 6.0 * KG_PER_LB, "jet": 6.7 * KG_PER_LB}
NM_PER_FT = 0.7375621  # ft-lb per newton metre
FUEL_WORDS = {"gal": "gallons", "lb": "pounds", "kg": "kilograms"}
# Engine kinds (vimaan_nlu.aircraft): which readings their gauges show.
ENGINE_READINGS = {
    "piston": (
        "r p m",
        "manifold pressure",
        "e g t",
        "oil temperature",
        "oil pressure",
        "fuel flow",
    ),
    "turboprop": (
        "torque",
        "i t t",
        "n one",
        "r p m",
        "oil temperature",
        "oil pressure",
        "fuel flow",
    ),
    "jet": ("n one", "n two", "e g t", "oil temperature", "oil pressure", "fuel flow"),
}
# What "engine status" reads, per kind.
ENGINE_SUMMARY = {
    "piston": ("r p m", "manifold pressure", "fuel flow"),
    "turboprop": ("torque", "i t t", "n one"),
    "jet": ("n one", "e g t", "fuel flow"),
}
READING_NAMES = {
    "n one": "N1",
    "n two": "N2",
    "r p m": "RPM",
    "manifold pressure": "manifold pressure",
    "torque": "torque",
    "i t t": "I T T",
    "e g t": "E G T",
    "oil temperature": "oil temperature",
    "oil pressure": "oil pressure",
    "fuel flow": "fuel flow",
}


def _round(value: float, step: float) -> int:
    return int(round(value / step) * step)


def three_digits(degrees: float) -> str:
    """A heading or course as said: "two seven zero", 0 as "three six zero"."""
    d = _round(degrees, 1) % 360 or 360
    return spell_digits(f"{d:03d}")


def altitude(feet: float) -> str:
    return f"Altitude {_round(feet, 10)} feet"


def heading(degrees: float) -> str:
    return f"Heading {three_digits(degrees)}"


def airspeed(knots: float) -> str:
    return f"Airspeed {_round(knots, 1)} knots"


def ground_speed(knots: float) -> str:
    return f"Ground speed {_round(knots, 1)} knots"


def vertical_speed(fpm: float) -> str:
    """Nearest 50 ft/min; under 100 either way is level."""
    if abs(fpm) < 100:
        return "Level"
    word = "Climbing" if fpm > 0 else "Descending"
    return f"{word} {_round(abs(fpm), 50)} feet per minute"


def mach(number: float) -> str:
    return "Mach point " + spell_digits(f"{round(number, 2):.2f}".split(".")[1])


def height_above_ground(feet: float) -> str:
    return f"{_round(max(feet, 0.0), 10)} feet above the ground"


def wind(from_degrees: float, knots: float) -> str:
    if knots < 2:
        return "Wind calm"
    return f"Wind {three_digits(from_degrees)} at {_round(knots, 1)} knots"


def temperature(celsius: float) -> str:
    c = _round(celsius, 1)
    return f"Outside air temperature {'minus ' if c < 0 else ''}{abs(c)} degrees Celsius"


def altimeter(inches_hg: float, standard: bool, hpa: bool) -> str:
    """Inches as said ("two niner niner two"), hPa as Q N H, or standard."""
    if standard:
        return "Altimeter standard"
    if hpa:
        return f"Q N H {spell_digits(str(_round(inches_hg * 33.8639, 1)))}"
    return f"Altimeter {spell_digits(f'{inches_hg:.2f}'.replace('.', ''))}"


def fuel_amount(kg: float, unit: str, fuel: str = "jet") -> str:
    """X-Plane's kilograms in the gauge's unit, rounded as a gauge is read."""
    if unit == "gal":
        value = _round(kg / KG_PER_GALLON[fuel], 1)
    elif unit == "lb":
        lb = kg / KG_PER_LB
        value = _round(lb, 100 if lb >= 10000 else 10)
    else:
        value = _round(kg, 100 if kg >= 10000 else 10)
    return f"{value} {FUEL_WORDS[unit]}"


def fuel(kg: float, unit: str, fuel_kind: str = "jet") -> str:
    return f"Fuel {fuel_amount(kg, unit, fuel_kind)}"


def fuel_time(kg: float, kg_per_sec: float) -> str:
    """Endurance at the present fuel flow."""
    if kg_per_sec <= 1e-6:
        return "No fuel flow: the engines aren't using fuel"
    minutes = int(kg / kg_per_sec / 60)
    hours, mins = divmod(minutes, 60)
    if hours == 0:
        return f"About {mins} minutes of fuel at this fuel flow"
    hour_word = "hour" if hours == 1 else "hours"
    return f"About {hours} {hour_word} {mins} minutes of fuel at this fuel flow"


def engine_value(reading: str, value: float, fuel_unit: str = "kg", fuel_kind: str = "jet") -> str:
    """One engine reading in its gauge's terms (no reading name)."""
    if reading in ("n one", "n two"):
        return f"{_round(value, 1)} percent"
    if reading == "r p m":
        return str(_round(value, 10))
    if reading == "manifold pressure":
        return f"{round(value, 1):g} inches"
    if reading == "torque":
        return f"{_round(value * NM_PER_FT, 10)} foot pounds"
    if reading in ("i t t", "e g t", "oil temperature"):
        return f"{_round(value, 5)} degrees"
    if reading == "oil pressure":
        return f"{_round(value, 1)} P S I"
    if reading == "fuel flow":
        per_hour = value * 3600
        if fuel_unit == "gal":
            return f"{round(per_hour / KG_PER_GALLON[fuel_kind], 1):g} gallons per hour"
        if fuel_unit == "lb":
            return f"{_round(per_hour / KG_PER_LB, 10)} pounds per hour"
        return f"{_round(per_hour, 10)} kilograms per hour"
    raise ValueError(reading)


def engine(
    reading: str,
    values: Sequence[float],
    instance: str | None,
    fuel_unit: str = "kg",
    fuel_kind: str = "jet",
) -> str:
    """ "N1 85 percent on both engines", or engine by engine when they differ
    or one was asked for. ``values``: one per engine."""
    name = READING_NAMES[reading]
    said = [engine_value(reading, v, fuel_unit, fuel_kind) for v in values]
    if instance in ("1", "2") and int(instance) <= len(values):
        return f"Engine {instance} {name} {said[int(instance) - 1]}"
    if len(said) == 1:
        return _cap(f"{name} {said[0]}")
    if len(set(said)) == 1:
        both = "both engines" if len(said) == 2 else "all engines"
        return _cap(f"{name} {said[0]} on {both}")
    return _cap(", ".join(f"engine {i} {name} {s}" for i, s in enumerate(said, 1)))


def _cap(text: str) -> str:
    """First letter up, the rest as written ("N1", "I T T" stay)."""
    return text[:1].upper() + text[1:]


def engines_running(running: Sequence[bool], instance: str | None) -> str:
    if instance in ("1", "2") and int(instance) <= len(running):
        on = running[int(instance) - 1]
        return f"Engine {instance} {'running' if on else 'not running'}"
    if all(running):
        return "Both engines running" if len(running) == 2 else "Engines running"
    if not any(running):
        return "Engines not running"
    return _cap(
        ", ".join(
            f"engine {i} {'running' if on else 'not running'}" for i, on in enumerate(running, 1)
        )
    )


def frequency(units: int, per_mhz: int) -> str:
    """A radio dataref (COM in kHz, NAV in 10 kHz) as said."""
    mhz = units / per_mhz
    text = f"{mhz:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return spell_digits(text)


def radio(name: str, active: int, standby: int, per_mhz: int, which: str | None) -> str:
    """ "COM 1 one two one decimal niner, standby one three two decimal four five"."""
    if which == "active":
        return f"{name} active {frequency(active, per_mhz)}"
    if which == "standby":
        return f"{name} standby {frequency(standby, per_mhz)}"
    return f"{name} {frequency(active, per_mhz)}, standby {frequency(standby, per_mhz)}"


def course(nav: str, degrees: float) -> str:
    return f"{nav} course {three_digits(degrees)}"


XPDR_MODES = {0: "off", 1: "standby", 2: "on", 3: "alt"}


def transponder(code: int, mode: str | None) -> str:
    said = f"Squawk {spell_digits(f'{int(code):04d}')}"
    return f"{said}, transponder {mode}" if mode else said


def waypoint(distance_nm: float, minutes: float, has_destination: bool) -> str:
    if not has_destination or distance_nm <= 0:
        return "No waypoint set"
    miles = _round(distance_nm, 1)
    said = f"Next waypoint {miles} {'mile' if miles == 1 else 'miles'}"
    if minutes > 0:
        m = _round(minutes, 1)
        said += f", {m} {'minute' if m == 1 else 'minutes'}"
    return said


def clock(seconds: float, zone: str) -> str:
    """ "Time one four three five Zulu" / "Local time zero seven three five"."""
    total = int(seconds) % 86400
    hhmm = f"{total // 3600:02d}{(total % 3600) // 60:02d}"
    if zone == "local":
        return f"Local time {spell_digits(hhmm)}"
    return f"Time {spell_digits(hhmm)} Zulu"
