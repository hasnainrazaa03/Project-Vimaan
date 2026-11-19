"""Tier 4 in the plugin: questions answered from the sim, never acted on.

The fake sim starts as a Baron-like piston twin; the 737 is loaded with its
own switches (test_switches.load_737)."""

import pytest
import test_plugin
import test_switches
from test_plugin import events, load_aircraft, say

G = "sim/cockpit2/gauges/indicators/"
AP = "sim/cockpit2/autopilot/"
E = "sim/cockpit2/engine/indicators/"
JET, PISTON = 7, 1


@pytest.fixture
def sim(monkeypatch, tmp_path):
    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    yield module, plugin, xp
    plugin.XPluginStop()


def ask(plugin, text, intent, slots=None, **kw):
    return say(plugin, text, intent, slots or {}, **kw)


def answered(plugin):
    return events(plugin, "question_answered")[-1]


@pytest.mark.parametrize(
    ("reading", "datarefs", "want"),
    [
        ("altitude", {G + "altitude_ft_pilot": 7404.0}, "Altitude 7400 feet"),
        ("heading", {G + "heading_AHARS_deg_mag_pilot": 270.3}, "Heading two seven zero"),
        ("airspeed", {G + "airspeed_kts_pilot": 164.6}, "Airspeed 165 knots"),
        ("vertical speed", {G + "vvi_fpm_pilot": -1187.0}, "Descending 1200 feet per minute"),
        ("ground speed", {G + "ground_speed_kt": 285.2}, "Ground speed 285 knots"),
        ("mach", {G + "mach_pilot": 0.513}, "Mach point five one"),
        (
            "height above ground",
            {G + "radio_altimeter_height_ft_pilot": 4972.8},
            "4970 feet above the ground",
        ),
        (
            "wind",
            {G + "wind_heading_deg_mag": 45.0, G + "wind_speed_kts": 12.0},
            "Wind zero four five at 12 knots",
        ),
        (
            "temperature",
            {"sim/cockpit2/temperature/outside_air_temp_degc": -4.77},
            "Outside air temperature minus 5 degrees Celsius",
        ),
        (
            "altimeter",
            {"sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot": 30.01},
            "Altimeter three zero zero one",
        ),
    ],
)
def test_flight_readings(sim, reading, datarefs, want):
    _, plugin, xp = sim
    xp.datarefs.update(datarefs)
    assert ask(plugin, f"what's our {reading}", "ask_flight_data", {"reading": reading}) == [want]
    assert xp.writes == [] and xp.commands_fired == []
    assert events(plugin, "command_done")[-1]["outcome"] == "answered"
    assert answered(plugin)["answer"] == want


@pytest.mark.parametrize(
    ("phase", "want"),
    [
        ("cruise", "Cruise"),
        ("approach", "On approach"),
        ("takeoff_roll", "Takeoff roll"),
        ("go_around", "Go-around"),
        ("unknown", "I can't tell the phase of flight yet"),
    ],
)
def test_the_phase_of_flight_is_answered_from_the_tracker(sim, phase, want):
    # Phase F.4: "what phase are we in" (the question model learns it in q2).
    _, plugin, xp = sim
    plugin.phase.phase = phase
    said = ask(plugin, "what phase are we in", "ask_flight_data", {"reading": "phase"})
    assert said == [want]
    assert xp.writes == [] and xp.commands_fired == []
    assert answered(plugin)["answer"] == want


def test_altimeter_standard_and_the_737_in_hpa(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/cockpit2/gauges/actuators/barometer_setting_is_std_pilot"] = 1
    slots = {"reading": "altimeter"}
    assert ask(plugin, "what's the altimeter", "ask_flight_data", slots) == ["Altimeter standard"]
    xp.datarefs["sim/cockpit2/gauges/actuators/barometer_setting_is_std_pilot"] = 0
    xp.datarefs["sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot"] = 29.92
    test_switches.load_737(plugin, xp)
    xp.datarefs["laminar/B738/EFIS_control/capt/baro_in_hpa"] = 0.0
    assert ask(plugin, "what's the altimeter", "ask_flight_data", slots) == [
        "Altimeter two niner niner two"
    ]
    xp.datarefs["laminar/B738/EFIS_control/capt/baro_in_hpa"] = 1.0
    assert ask(plugin, "what's the altimeter", "ask_flight_data", slots) == [
        "Q N H one zero one three"
    ]


def test_fuel_in_each_aircrafts_unit(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/weight/m_fuel_total"] = 271.6
    assert ask(plugin, "how much fuel do we have", "ask_fuel") == ["Fuel 100 gallons"]  # piston
    test_switches.load_737(plugin, xp)
    xp.datarefs["sim/flightmodel/weight/m_fuel_total"] = 10439.4
    assert ask(plugin, "how much fuel do we have", "ask_fuel") == ["Fuel 23000 pounds"]


def test_fuel_on_a_jet_with_no_profile_is_in_kilograms(sim):
    _, plugin, xp = sim
    load_aircraft(plugin, xp, JET)
    xp.datarefs["sim/flightmodel/weight/m_fuel_total"] = 10439.4
    assert ask(plugin, "how much fuel do we have", "ask_fuel") == ["Fuel 10400 kilograms"]


def test_endurance_from_fuel_flow(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/weight/m_fuel_total"] = 271.6
    xp.datarefs[E + "fuel_flow_kg_sec"] = [0.01, 0.01] + [0.0] * 14
    assert ask(plugin, "how long will our fuel last", "ask_fuel_time") == [
        "About 3 hours 46 minutes of fuel at this fuel flow"
    ]


def test_engine_readings_by_engine_kind(sim):
    _, plugin, xp = sim
    xp.datarefs[E + "oil_pressure_psi"] = [58.0, 57.0] + [0.0] * 14
    slots = {"engine_reading": "oil pressure"}
    assert ask(plugin, "what's the oil pressure", "ask_engine", slots) == [
        "Engine 1 oil pressure 58 P S I, engine 2 oil pressure 57 P S I"
    ]
    assert ask(plugin, "what's engine 2 oil pressure", "ask_engine", slots) == [
        "Engine 2 oil pressure 57 P S I"
    ]
    # A piston has no N1 gauge (X-Plane still reports one).
    assert ask(plugin, "what's the n 1", "ask_engine", {"engine_reading": "n one"}) == [
        "This aircraft shows no N1"
    ]
    load_aircraft(plugin, xp, JET)
    xp.datarefs[E + "N1_percent"] = [40.66, 40.66] + [0.0] * 14
    assert ask(plugin, "what's the n 1", "ask_engine", {"engine_reading": "n one"}) == [
        "N1 41 percent on both engines"
    ]


def test_engine_running_and_status(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/engine/ENGN_running"] = [1, 0] + [0] * 14
    assert ask(plugin, "is engine 2 running", "ask_engine", {"instance": "2"}) == [
        "Engine 2 not running"
    ]
    xp.datarefs[E + "prop_speed_rpm"] = [2700.0, 2700.0] + [0.0] * 14
    xp.datarefs[E + "MPR_in_hg"] = [24.0, 24.0] + [0.0] * 14
    xp.datarefs[E + "fuel_flow_kg_sec"] = [0.01, 0.01] + [0.0] * 14
    assert ask(plugin, "engine status", "ask_engine") == [
        "RPM 2700 on both engines. Manifold pressure 24 inches on both engines. "
        "Fuel flow 13.2 gallons per hour on both engines"
    ]


def test_radios_and_transponder(sim):
    _, plugin, xp = sim
    r = "sim/cockpit2/radios/actuators/"
    xp.datarefs[r + "com1_frequency_hz_833"] = 121900
    xp.datarefs[r + "com1_standby_frequency_hz_833"] = 132450
    assert ask(plugin, "what's com 1", "ask_com", {"com_port": "1"}) == [
        "COM 1 one two one decimal niner, standby one three two decimal four five"
    ]
    assert ask(
        plugin, "what's com 1 standby", "ask_com", {"com_port": "1", "which": "standby"}
    ) == ["COM 1 standby one three two decimal four five"]
    xp.datarefs[r + "nav2_obs_deg_mag_pilot"] = 180.0
    assert ask(
        plugin, "what's the course on nav 2", "ask_nav", {"nav_port": "2", "which": "course"}
    ) == ["NAV 2 course one eight zero"]
    xp.datarefs[r + "transponder_code"] = 4217
    xp.datarefs[r + "transponder_mode"] = 3
    assert ask(plugin, "what's our squawk", "ask_transponder") == [
        "Squawk four two one seven, transponder altitude"
    ]
    assert xp.writes == [] and xp.commands_fired == []


def test_autopilot_status_and_targets(sim):
    _, plugin, xp = sim
    assert ask(plugin, "autopilot status", "ask_autopilot") == ["Autopilot off"]
    xp.datarefs.update(
        {
            AP + "servos_on": 1,
            AP + "heading_status": 2,
            AP + "altitude_hold_status": 2,
            AP + "heading_dial_deg_mag_pilot": 270.0,
            AP + "altitude_dial_ft": 8000.0,
        }
    )
    assert ask(plugin, "is the autopilot on", "ask_autopilot") == [
        "Autopilot on, heading two seven zero, altitude hold, altitude set 8000 feet"
    ]
    assert ask(plugin, "what altitude is set", "ask_autopilot", {"ap_target": "altitude"}) == [
        "Altitude set 8000 feet"
    ]
    assert ask(plugin, "what heading is set", "ask_autopilot", {"ap_target": "heading"}) == [
        "Heading bug two seven zero"
    ]
    assert xp.writes == [] and xp.commands_fired == []


def test_switch_states(sim):
    _, plugin, xp = sim
    assert ask(plugin, "are the landing lights on", "ask_switch", {"light": "landing"}) == [
        "Landing lights off"
    ]
    xp.datarefs["sim/cockpit2/switches/yaw_damper_on"] = 1
    assert ask(plugin, "is the yaw damper on", "ask_switch", {"switch_system": "yaw damper"}) == [
        "Yaw damper on"
    ]
    assert ask(plugin, "is the gear down", "ask_switch", {"switch_system": "gear"}) == ["Gear down"]
    xp.datarefs["sim/flightmodel/controls/parkbrake"] = 1.0
    assert ask(
        plugin, "is the parking brake set", "ask_switch", {"switch_system": "parking brake"}
    ) == ["Parking brake set"]
    # A piston twin has no auto brake.
    assert ask(
        plugin, "what's the auto brake set to", "ask_switch", {"switch_system": "auto brake"}
    ) == ["This aircraft has no auto brake"]
    assert ask(plugin, "where are the flaps", "ask_switch", {"switch_system": "flaps"}) == [
        "Flaps up"
    ]
    assert xp.writes == [] and xp.commands_fired == []


def test_flaps_by_the_cockpits_label(sim):
    # The 737's handle notches are labelled 1 2 5 10 15 25 30 40 (aircraft.py).
    import dataclasses
    from types import MappingProxyType

    _, plugin, xp = sim
    profile = plugin.profile
    labels = ("1", "2", "5", "10", "15", "25", "30", "40")
    plugin.profile = dataclasses.replace(
        profile,
        info=dataclasses.replace(profile.info, flap_detents=8),
        settings=MappingProxyType({**profile.settings, "flap_labels": labels}),
    )
    xp.datarefs["sim/cockpit2/controls/flap_handle_request_ratio"] = 5 / 8
    slots = {"switch_system": "flaps"}
    assert ask(plugin, "where are the flaps", "ask_switch", slots) == ["Flaps 15"]


def test_waypoint_time_and_status(sim):
    _, plugin, xp = sim
    assert ask(plugin, "how far to the next way point", "ask_waypoint") == ["No waypoint set"]
    xp.datarefs["sim/cockpit/gps/destination_index"] = 3
    xp.datarefs["sim/cockpit2/radios/indicators/gps_dme_distance_nm"] = 12.4
    xp.datarefs["sim/cockpit2/radios/indicators/gps_dme_time_min"] = 3.6
    assert ask(plugin, "how far to the next way point", "ask_waypoint") == [
        "Next waypoint 12 miles, 4 minutes"
    ]
    xp.datarefs["sim/time/zulu_time_sec"] = 52500.0
    assert ask(plugin, "what time is it", "ask_time") == ["Time one four three five Zulu"]
    xp.datarefs["sim/time/local_time_sec"] = 27300.0
    assert ask(plugin, "what's the local time", "ask_time", {"zone": "local"}) == [
        "Local time zero seven three five"
    ]
    xp.datarefs[G + "altitude_ft_pilot"] = 8400.0
    xp.datarefs[G + "heading_AHARS_deg_mag_pilot"] = 270.0
    xp.datarefs[G + "airspeed_kts_pilot"] = 250.0
    assert ask(plugin, "status report", "ask_status_generic") == [
        "Altitude 8400 feet, heading two seven zero, airspeed 250 knots, autopilot off"
    ]


# --- never acted on --------------------------------------------------------


def test_a_bare_reading_is_answered_not_acted_on(sim):
    # "what's the heading" heard "what the heading", or "heading": v26 read it
    # as heading MODE. A reading named alone is answered.
    _, plugin, xp = sim
    xp.datarefs[G + "heading_AHARS_deg_mag_pilot"] = 95.0
    for heard in ("what the heading", "heading"):
        assert say(plugin, heard, "set_autopilot_mode", {"mode": "heading"}) == [
            "Heading zero niner five"
        ]
    assert xp.writes == [] and xp.commands_fired == []
    assert answered(plugin)["intent"] == "ask_flight_data"


def test_the_question_key_never_acts(sim):
    _, plugin, xp = sim
    # A command said on the question key: refused, nothing changes.
    spoken = say(plugin, "autopilot on", "toggle_autopilot", {"state": "on"}, ask=True)
    assert spoken == ["I can't answer that"]
    assert xp.writes == [] and xp.commands_fired == []
    assert events(plugin, "implausible")[-1]["ask"] is True
    # A question on it is answered.
    xp.datarefs[G + "altitude_ft_pilot"] = 5000.0
    assert say(plugin, "altitude", "ask_flight_data", {"reading": "altitude"}, ask=True) == [
        "Altitude 5000 feet"
    ]


def test_shift_z_is_the_question_key(sim):
    module, plugin, xp = sim
    keys = {(key, flags): cb for key, flags, _desc, cb in xp.hotkeys}
    assert (xp.VK_Z, xp.DownFlag | xp.ShiftFlag) in keys
    assert (xp.VK_Z, xp.DownFlag) in keys  # the talk key is unchanged
    plugin.OnAskPressCallback(None)
    assert plugin.worker.sent[-1]["op"] == "start" and plugin.worker.sent[-1]["ask"] is True
    assert events(plugin, "ptt_press")[-1]["ask"] is True
    plugin.OnReleaseCallback(None)
    starts = [m for m in plugin.worker.sent if m["op"] == "start"]
    assert starts[-1]["ask"] is True


def test_an_unsure_question_is_not_answered(sim):
    _, plugin, xp = sim
    # Not a bare reading ("what's our altitude" would be answered by the rule).
    slots = {"engine_reading": "fuel flow"}
    spoken = say(plugin, "what's the fuel flow", "ask_engine", slots, conf=0.3)
    assert spoken == ["I can't answer that"]


def test_a_transcript_the_question_grammar_redid_never_acts(sim):
    # "are the landing lights on" -> "our the landing lights on": whatever the
    # model reads, a re-decoded transcript is a question.
    _, plugin, xp = sim
    test_plugin.NLU["our the landing lights on"] = (
        "toggle_lights",
        {"light": "landing", "switch": "on"},
        0.99,
    )
    p = test_plugin._fake_predict("our the landing lights on")
    plugin._execute_command(
        {
            "cmd": "c1",
            "text": "our the landing lights on",
            "prediction": {
                **p,
                "nlu_ms": 5.0,
                "normalized": p["normalized_text"],
                "heard": "r the landing lights on",
            },
            "t_press": 0.0,
            "t_release": 0.0,
        }
    )
    assert xp.spoken[-1] == "I can't answer that"
    assert xp.writes == [] and xp.commands_fired == []


# --- with a question model (qN) beside the command model --------------------


def say_two(plugin, text, command, question, ask=False):
    """Dispatch as the worker does with a question model loaded: the command
    model's reading, and the question model's beside it (``question``:
    (intent, slots, confidence))."""
    import sys
    import time

    intent, slots, conf = command
    q_intent, q_slots, q_conf = question
    xp = sys.modules["XPPython3"].xp
    before = len(xp.spoken)
    now = time.perf_counter()
    plugin._execute_command(
        {
            "cmd": "c9998",
            "text": text,
            "prediction": {
                "intent": intent,
                "slots": dict(slots),
                "confidence": conf,
                "normalized_text": text,
                "nlu_ms": 5.0,
                "normalized": text,
                "to_reading": None,
                "question": {
                    "intent": q_intent,
                    "slots": dict(q_slots),
                    "confidence": q_conf,
                    "normalized": text,
                },
            },
            "t_press": now,
            "t_release": now,
            "ask": ask,
        }
    )
    return xp.spoken[before:]


def test_a_question_said_like_a_command_is_answered_by_the_question_model(sim):
    # v26 reads "landing lights status" as toggle_lights with no on/off:
    # nothing it would act on. The question model reads a question.
    _, plugin, xp = sim
    spoken = say_two(
        plugin,
        "landing lights status",
        ("toggle_lights", {"light": "landing"}, 0.99),
        ("ask_switch", {"light": "landing"}, 0.97),
    )
    assert spoken == ["Landing lights off"]
    assert xp.writes == [] and xp.commands_fired == []
    assert answered(plugin)["intent"] == "ask_switch"


def test_a_command_the_command_model_would_do_is_done(sim):
    # The question model can't take a command away: when v26 would act, it acts.
    _, plugin, xp = sim
    say_two(
        plugin,
        "beacon on",
        ("toggle_lights", {"light": "beacon", "switch": "on"}, 0.99),
        ("ask_switch", {"light": "beacon"}, 0.9),
    )
    assert test_switches.run(plugin, xp) == ["Beacon on"]
    assert xp.commands_fired == ["sim/lights/beacon_lights_on"]
    assert not events(plugin, "question_answered")


def test_with_a_question_model_its_reading_answers_the_question(sim):
    # "what's the gear doing": v26 says status (its own old question intent);
    # the question model's family and slots are the ones answered.
    _, plugin, xp = sim
    spoken = say_two(
        plugin,
        "what's the gear doing",
        ("ask_status_generic", {}, 0.95),
        ("ask_switch", {"switch_system": "gear"}, 0.98),
    )
    assert spoken == ["Gear down"]
    assert answered(plugin)["intent"] == "ask_switch"


def test_with_a_question_model_an_unsure_question_is_not_answered(sim):
    # The question model isn't sure: the command model's own question
    # intents no longer answer, and nothing is acted on.
    _, plugin, xp = sim
    spoken = say_two(
        plugin,
        "what's the gear doing",
        ("ask_status_generic", {}, 0.95),
        ("ask_switch", {"switch_system": "gear"}, 0.3),
    )
    assert spoken == ["I can't answer that"]
    assert xp.writes == [] and xp.commands_fired == []


def test_a_system_named_alone_still_asks_which_way(sim):
    # "flaps" alone asks "which way", as before the question model.
    _, plugin, xp = sim
    spoken = say_two(
        plugin,
        "flaps",
        ("None", {}, 0.99),
        ("ask_switch", {"switch_system": "flaps"}, 0.9),
    )
    assert not events(plugin, "question_answered")
    assert events(plugin, "bare_command")[-1]["intent"] == "set_flaps"
    assert xp.writes == [] and spoken


def test_self_test_probes_the_question_model_too(sim):
    _, plugin, _ = sim
    plugin._model_intents = {"toggle_lights"}
    plugin._question_intents = set()
    assert {p[1] for p in plugin._self_test_probes()} == {"toggle_lights"}
    plugin._question_intents = {"ask_flight_data", "None"}
    assert {p[1] for p in plugin._self_test_probes()} == {"toggle_lights", "ask_flight_data"}


@pytest.mark.parametrize("ask", [False, True])
def test_a_word_alone_is_not_answered_by_the_question_model(sim, ask):
    # q1 attempt 1 answered "armed" as the autopilot: a word alone is a
    # fragment, on either key.
    _, plugin, xp = sim
    spoken = say_two(plugin, "armed", ("None", {}, 0.6), ("ask_autopilot", {}, 0.98), ask=ask)
    assert not events(plugin, "question_answered")
    assert xp.writes == [] and xp.commands_fired == []
    if ask:
        assert spoken == ["I can't answer that"]


def test_a_reading_alone_is_still_answered(sim):
    # "heading" alone is the bare-reading rule's, not the question model's.
    _, plugin, xp = sim
    say_two(
        plugin,
        "heading",
        ("set_autopilot_mode", {"mode": "heading"}, 0.9),
        ("ask_flight_data", {"reading": "heading"}, 0.99),
        ask=True,
    )
    assert answered(plugin)["intent"] == "ask_flight_data"
    assert xp.writes == [] and xp.commands_fired == []


def test_the_session_report_lists_an_answer_with_its_question_not_as_a_problem(sim):
    # 147 answers in the Tier 4 run filled "Problems"; and the command list
    # showed the command model's reading ("what's the heading" -> heading mode).
    _, plugin, xp = sim
    xp.datarefs[G + "heading_AHARS_deg_mag_pilot"] = 270.3
    ask(plugin, "what's the heading", "ask_flight_data", {"reading": "heading"})
    plugin.XPluginStop()
    text = test_plugin._report_module().report(plugin.tel.events_path)
    commands = text.split("Problems")[0]
    said = "\"what's the heading\" -> ask_flight_data {'reading': 'heading'}"
    assert f"{said}: 'Heading two seven zero'" in commands
    assert "Problems: none" in text
