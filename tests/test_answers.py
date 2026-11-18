"""vimaan_nlu.answers: the words of every Tier 4 answer, in the cockpit's units."""

import pytest
from vimaan_nlu import answers as a


@pytest.mark.parametrize(
    ("spoken", "want"),
    [
        (a.altitude(7404), "Altitude 7400 feet"),
        (a.heading(270.4), "Heading two seven zero"),
        (a.heading(0.2), "Heading three six zero"),
        (a.airspeed(164.6), "Airspeed 165 knots"),
        (a.ground_speed(285.2), "Ground speed 285 knots"),
        (a.vertical_speed(-1187), "Descending 1200 feet per minute"),
        (a.vertical_speed(740), "Climbing 750 feet per minute"),
        (a.vertical_speed(60), "Level"),
        (a.mach(0.7812), "Mach point seven eight"),
        (a.height_above_ground(4972.8), "4970 feet above the ground"),
        (a.wind(271, 34.6), "Wind two seven one at 35 knots"),
        (a.wind(90, 1), "Wind calm"),
        (a.temperature(-4.6), "Outside air temperature minus 5 degrees Celsius"),
        (a.temperature(15.2), "Outside air temperature 15 degrees Celsius"),
    ],
)
def test_flight_readings(spoken, want):
    assert spoken == want


def test_altimeter_in_the_unit_the_aircraft_shows():
    assert a.altimeter(29.92, standard=False, hpa=False) == "Altimeter two niner niner two"
    assert a.altimeter(29.92, standard=False, hpa=True) == "Q N H one zero one three"
    assert a.altimeter(29.92, standard=True, hpa=False) == "Altimeter standard"


def test_fuel_in_the_gauges_unit():
    # The probe's readings (2026-11-16): X-Plane says kilograms on every aircraft.
    assert a.fuel(10439.4, "lb") == "Fuel 23000 pounds"  # 737: its display read 23.0
    assert a.fuel(54463.1, "kg") == "Fuel 54500 kilograms"  # A330: FOB 54,420 KG
    assert a.fuel(271.6, "gal", "avgas") == "Fuel 100 gallons"  # Baron
    assert a.fuel(556.1, "lb") == "Fuel 1230 pounds"  # King Air


def test_endurance():
    assert a.fuel_time(10439.4, 0.24) == "About 12 hours 4 minutes of fuel at this fuel flow"
    assert a.fuel_time(100.0, 0.05) == "About 33 minutes of fuel at this fuel flow"
    assert a.fuel_time(100.0, 0.0) == "No fuel flow: the engines aren't using fuel"


def test_engine_readings():
    assert a.engine("n one", [40.66, 40.66], None) == "N1 41 percent on both engines"
    assert a.engine("n one", [40.6, 52.0], None) == "Engine 1 N1 41 percent, engine 2 N1 52 percent"
    assert a.engine("oil pressure", [58, 57], "2") == "Engine 2 oil pressure 57 P S I"
    # 0.12 kg/s per engine: the 737's display read 0.94 (thousand lb/h).
    assert a.engine("fuel flow", [0.12, 0.12], None, "lb") == (
        "Fuel flow 950 pounds per hour on both engines"
    )
    assert a.engine("torque", [709.0, 709.0], None) == "Torque 520 foot pounds on both engines"
    assert a.engine("r p m", [2704.8], None) == "RPM 2700"
    assert a.engine("n one", [80.0] * 4, None) == "N1 80 percent on all engines"


def test_engines_running():
    assert a.engines_running([True, True], None) == "Both engines running"
    assert a.engines_running([True, False], None) == "Engine 1 running, engine 2 not running"
    assert a.engines_running([True, False], "2") == "Engine 2 not running"


def test_radios_and_transponder():
    assert a.radio("COM 1", 121900, 132450, 1000, None) == (
        "COM 1 one two one decimal niner, standby one three two decimal four five"
    )
    assert a.radio("NAV 1", 11050, 11390, 100, "standby") == (
        "NAV 1 standby one one three decimal niner"
    )
    assert a.radio("COM 2", 118005, 121500, 1000, "active") == (
        "COM 2 active one one eight decimal zero zero five"
    )
    assert a.course("NAV 2", 180) == "NAV 2 course one eight zero"
    assert a.transponder(1200, "altitude") == "Squawk one two zero zero, transponder altitude"
    assert a.transponder(120, None) == "Squawk zero one two zero"


def test_waypoint_and_time():
    assert a.waypoint(12.4, 3.6, True) == "Next waypoint 12 miles, 4 minutes"
    assert a.waypoint(0.0, 0.0, False) == "No waypoint set"
    assert a.clock(52500, "zulu") == "Time one four three five Zulu"
    assert a.clock(27300, "local") == "Local time zero seven three five"
