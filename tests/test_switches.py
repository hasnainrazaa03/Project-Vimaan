"""Tier 3 switches: lights, cabin signs, yaw damper, speed brake, autobrake.

The plans (ML/vimaan_nlu/switches.py) and the plugin running them on the fake
sim, with each aircraft's own switches modelled as its scripts behave
(docs/AIRCRAFT_PROFILES.md, "Tier 3 findings")."""

import pytest
import test_plugin
from test_plugin import aircraft, events, run_selectors, say, selector
from vimaan_nlu import switches
from vimaan_nlu.aircraft import AircraftInfo, resolve


@pytest.fixture
def sim(monkeypatch, tmp_path):
    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    yield module, plugin, xp
    plugin.XPluginStop()


SW = "sim/cockpit2/switches/"
JET, PISTON = 7, 1


def run(plugin, xp, *applies, loops=40):
    return run_selectors(plugin, xp, *applies, loops=loops)


# --- the plans --------------------------------------------------------------


def test_every_plan_reaches_a_listed_setting():
    keys = {f"light:{n}" for n in switches.LIGHTS} | {f"sign:{n}" for n in switches.SIGNS}
    keys |= {"yaw_damper", "speed_brake", "autobrake"}
    for plans in (switches.GENERIC, switches.B58, switches.C90, switches.B738, switches.A333):
        assert set(plans) <= keys
        for table in plans.values():
            for setting, plan in (table or {}).items():
                assert setting in {"on", "off", "auto"} | set(switches.SPEEDBRAKE_SETTINGS) | set(
                    switches.AUTOBRAKE_SETTINGS
                )
                assert plan and all(not isinstance(s, switches.Check) for s in plan[1:]), (
                    "a Check only starts a plan"
                )


def test_lookup_tells_absent_from_unavailable():
    assert switches.lookup(switches.B738, "sign:no smoking", "on") == ("absent", ())
    assert switches.lookup({}, "light:landing", "auto")[0] == "unavailable"
    assert switches.lookup(switches.A333, "light:strobe", "auto")[0] == "ok"
    assert switches.lookup({}, "light:beacon", "on")[0] == "ok"  # generic


def test_generic_autobrake_only_on_jets():
    def profile(engine):
        return resolve(AircraftInfo(engine_types=(engine,)), lambda _n: False)

    assert profile(PISTON).switch_plan("autobrake", "2")[0] == "absent"
    assert profile(JET).switch_plan("autobrake", "2")[0] == "ok"


def test_a_button_reads_any_of_its_values():
    b = switches.Button("x", "y", (0, 5))
    assert switches.reads(5.0, b.want) and switches.reads(0.01, b.want)
    assert not switches.reads(3.0, b.want) and not switches.reads(None, b.want)


# --- any aircraft (X-Plane's own commands) ---------------------------------


def test_beacon_on_then_already_on(sim):
    _, plugin, xp = sim
    assert say(plugin, "beacon on", "toggle_lights", {"light": "beacon", "switch": "on"}) == []
    assert run(plugin, xp) == ["Beacon on"]
    assert xp.commands_fired == ["sim/lights/beacon_lights_on"]
    assert say(plugin, "beacon on", "toggle_lights", {"light": "beacon", "switch": "on"}) == [
        "Beacon already on"
    ]


def test_lights_ask_what_they_cannot_do(sim):
    _, plugin, _ = sim
    assert say(plugin, "lights", "toggle_lights", {"light": "landing", "switch": "auto"}) == [
        "Landing lights on or off?"
    ]
    assert say(plugin, "lights", "toggle_lights", {"light": "logo", "switch": "on"}) == [
        "This aircraft has no logo lights"
    ]
    assert say(plugin, "lights", "toggle_lights", {"light": "landing"}) == [
        "Landing lights on or off?"
    ]


def test_all_lights_on_runs_every_light_this_aircraft_has(sim):
    _, plugin, xp = sim
    xp.datarefs[SW + "beacon_on"] = 1  # already on: not pressed again
    assert say(plugin, "all lights on", "toggle_lights", {"light": "all", "switch": "on"}) == []
    assert run(plugin, xp) == ["All lights on"]
    assert xp.commands_fired == [
        "sim/lights/landing_lights_on",
        "sim/lights/taxi_lights_on",
        "sim/lights/strobe_lights_on",
        "sim/lights/nav_lights_on",
    ]


def test_a_switch_that_never_moves_is_reported(sim):
    _, plugin, xp = sim
    del xp.effects["sim/lights/strobe_lights_on"]
    say(plugin, "strobes on", "toggle_lights", {"light": "strobe", "switch": "on"})
    assert run(plugin, xp) == ["Strobes not set: the switch did not move."]


def test_seat_belt_sign_toggles_only_when_needed(sim):
    _, plugin, xp = sim
    say(plugin, "seat belt sign on", "toggle_signs", {"sign": "seat belt", "switch": "on"})
    assert run(plugin, xp) == ["Seat belt sign on"]
    assert xp.commands_fired == ["sim/systems/seatbelt_sign_toggle"]
    assert say(plugin, "sign", "toggle_signs", {"sign": "seat belt", "switch": "on"}) == [
        "Seat belt sign already on"
    ]
    assert say(plugin, "sign", "toggle_signs", {"sign": "seat belt", "switch": "auto"}) == [
        "Seat belt sign on or off?"
    ]


def test_yaw_damper(sim):
    _, plugin, xp = sim
    say(plugin, "yaw damper on", "toggle_yaw_damper", {"state": "on"})
    assert run(plugin, xp) == ["Yaw damper on"]
    assert xp.datarefs[SW + "yaw_damper_on"] == 1


def test_speed_brake_on_an_aircraft_without_one(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/aircraft/parts/acf_sbrkEQ"] = 0
    assert say(plugin, "speed brake", "set_speed_brake", {"speedbrake": "armed"}) == [
        "This aircraft has no speed brakes"
    ]
    assert xp.writes == []


def test_speed_brake_armed_and_up_is_extended(sim):
    _, plugin, xp = sim
    say(plugin, "speed brake armed", "set_speed_brake", {"speedbrake": "armed"})
    assert run(plugin, xp) == ["Speed brake armed"]
    assert xp.datarefs["sim/cockpit2/controls/speedbrake_ratio"] == -0.5
    say(plugin, "speed brake up", "set_speed_brake", {"speedbrake": "up"})  # on the ground
    assert run(plugin, xp) == ["Speed brake extended"]
    assert xp.datarefs["sim/cockpit2/controls/speedbrake_ratio"] == 1.0


def test_speed_brake_low_in_the_air_asks_to_confirm(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 600 * 0.3048
    assert say(plugin, "speed brake", "set_speed_brake", {"speedbrake": "extend"}) == [
        "Confirm speed brake, low altitude. Say override, or repeat to confirm."
    ]
    assert xp.writes == []


def test_autobrake_on_a_piston_twin_and_on_a_jet(sim):
    _, plugin, xp = sim
    assert say(plugin, "auto brake two", "set_autobrake", {"autobrake_level": "2"}) == [
        "This aircraft has no auto brake"
    ]
    aircraft(plugin, xp, engine_type=JET)
    say(plugin, "auto brake two", "set_autobrake", {"autobrake_level": "2"})
    assert run(plugin, xp) == ["Auto brake two"]
    assert xp.datarefs[SW + "auto_brake_level"] == 3
    say(plugin, "auto brake r t o", "set_autobrake", {"autobrake": "r t o"})
    assert run(plugin, xp) == ["Auto brake R T O"]
    assert xp.datarefs[SW + "auto_brake_level"] == 0
    assert say(plugin, "auto brake", "set_autobrake", {}) == [
        "Auto brake off, one, two, three, or max?"
    ]


# --- Beechcraft Baron 58 -----------------------------------------------------

B58 = ("laminar/b58/surface_ice_switch_up", "laminar/b58/surface_ice_switch_dn")


def test_baron_landing_lights_are_two_switches(sim):
    _, plugin, xp = sim
    assert "b58_switches" in aircraft(plugin, xp, engine_type=PISTON, commands=B58).overrides
    say(plugin, "landing lights on", "toggle_lights", {"light": "landing", "switch": "on"})
    assert run(plugin, xp) == ["Landing lights on"]
    assert xp.commands_fired == ["sim/lights/landing_01_light_on", "sim/lights/landing_02_light_on"]
    assert xp.datarefs[SW + "landing_lights_switch"][:2] == [1.0, 1.0]


def test_baron_wing_light_and_no_signs(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, engine_type=PISTON, commands=B58)
    say(plugin, "wing lights on", "toggle_lights", {"light": "wing", "switch": "on"})
    assert run(plugin, xp) == ["Wing lights on"]
    assert xp.datarefs[SW + "generic_lights_switch"][0] == 1
    assert say(plugin, "sign", "toggle_signs", {"sign": "seat belt", "switch": "on"}) == [
        "There is no seat belt sign switch on this aircraft"
    ]


# --- Beechcraft King Air C90B ------------------------------------------------

C90 = (switches.C90_SIGNS.down, switches.C90_SIGNS.up)


def king_air_signs(xp):
    """The C90's sign switch: 1 NO SMOKING + SEAT BELT, 0 OFF, -1 SEAT BELT.
    Its handler forgets the seat belt sign going up from -1 to 0."""
    pos = switches.C90_SIGNS.position
    turn = selector(xp, pos, switches.C90_SIGNS.up, switches.C90_SIGNS.down, -1, 1)
    last = [xp.datarefs[pos]]

    def apply():
        turn()
        now = xp.datarefs[pos]
        if now != last[0]:
            xp.datarefs[SW + "no_smoking"] = int(now == 1)
            if now != 0 or last[0] != -1:  # the bug: -1 -> 0 leaves it lit
                xp.datarefs[SW + "fasten_seat_belts"] = int(now != 0)
            last[0] = now

    return apply


def test_king_air_seat_belt_sign_off_puts_out_what_its_switch_leaves_lit(sim):
    _, plugin, xp = sim
    assert "c90_switches" in aircraft(plugin, xp, engine_type=9, commands=C90).overrides
    signs = king_air_signs(xp)
    say(plugin, "seat belt sign on", "toggle_signs", {"sign": "seat belt", "switch": "on"})
    assert run(plugin, xp, signs) == ["Seat belt sign on"]
    assert xp.datarefs[switches.C90_SIGNS.position] == -1
    say(plugin, "seat belt sign off", "toggle_signs", {"sign": "seat belt", "switch": "off"})
    assert run(plugin, xp, signs) == ["Seat belt sign off"]
    assert xp.datarefs[switches.C90_SIGNS.position] == 0
    assert xp.datarefs[SW + "fasten_seat_belts"] == 0
    assert xp.commands_fired[-1] == "sim/systems/seatbelt_sign_toggle"


def test_king_air_no_smoking_sign_on_and_off(sim):
    _, plugin, xp = sim
    aircraft(plugin, xp, engine_type=9, commands=C90)
    signs = king_air_signs(xp)
    say(plugin, "no smoking sign on", "toggle_signs", {"sign": "no smoking", "switch": "on"})
    assert run(plugin, xp, signs) == ["No smoking sign on"]
    assert xp.datarefs[switches.C90_SIGNS.position] == 1
    say(plugin, "no smoking sign off", "toggle_signs", {"sign": "no smoking", "switch": "off"})
    assert run(plugin, xp, signs) == ["No smoking sign off"]
    assert xp.datarefs[switches.C90_SIGNS.position] == -1  # the seat belt sign stays


# --- Boeing 737-800 ------------------------------------------------------------

B738 = (
    switches.B738_POSITION.down,
    switches.B738_POSITION.up,
    switches.B738_SEATBELT.down,
    switches.B738_SEATBELT.up,
    switches.B738_TAXI.down,
    switches.B738_TAXI.up,
)


def load_737(plugin, xp):
    assert "b738_switches" in aircraft(plugin, xp, engine_type=JET, commands=B738).overrides
    xp.datarefs[switches.B738_SEATBELT.position] = 0
    xp.datarefs[switches.B738_TAXI.position] = 0
    xp.datarefs[switches.B738_SPEEDBRAKE] = 0.0


def position_switch(xp):
    """-1 STEADY, 0 OFF, 1 STROBE & STEADY; it fires X-Plane's light commands."""
    knob = switches.B738_POSITION
    turn = selector(xp, knob.position, knob.up, knob.down, -1, 1)

    def apply():
        turn()
        pos = xp.datarefs[knob.position]
        xp.datarefs[SW + "strobe_lights_on"] = int(pos == 1)
        xp.datarefs[SW + "navigation_lights_on"] = int(pos != 0)

    return apply


def test_737_strobes_off_leaves_the_nav_lights(sim):
    _, plugin, xp = sim
    load_737(plugin, xp)
    lights = position_switch(xp)
    say(plugin, "strobes on", "toggle_lights", {"light": "strobe", "switch": "on"})
    assert run(plugin, xp, lights) == ["Strobes on"]
    assert xp.datarefs[switches.B738_POSITION.position] == 1
    say(plugin, "strobes off", "toggle_lights", {"light": "strobe", "switch": "off"})
    assert run(plugin, xp, lights) == ["Strobes off"]
    assert xp.datarefs[switches.B738_POSITION.position] == -1  # STEADY: nav still on
    assert say(plugin, "lights", "toggle_lights", {"light": "nav", "switch": "on"}) == [
        "Nav lights already on"
    ]
    say(plugin, "nav lights off", "toggle_lights", {"light": "nav", "switch": "off"})
    assert run(plugin, xp, lights) == ["Nav lights off"]
    assert xp.datarefs[switches.B738_POSITION.position] == 0


def test_737_seat_belt_sign_auto(sim):
    _, plugin, xp = sim
    load_737(plugin, xp)
    knob = switches.B738_SEATBELT
    belt = selector(xp, knob.position, knob.up, knob.down, 0, 2)
    say(plugin, "seat belt sign auto", "toggle_signs", {"sign": "seat belt", "switch": "auto"})
    assert run(plugin, xp, belt) == ["Seat belt sign auto"]
    assert xp.datarefs[knob.position] == 1
    assert xp.commands_fired == [knob.up]  # its "dn" command turns it toward ON
    assert say(plugin, "sign", "toggle_signs", {"sign": "no smoking", "switch": "on"}) == [
        "There is no no smoking sign switch on this aircraft"
    ]


def test_737_landing_and_taxi_lights(sim):
    _, plugin, xp = sim
    load_737(plugin, xp)
    say(plugin, "landing lights on", "toggle_lights", {"light": "landing", "switch": "on"})
    assert run(plugin, xp) == ["Landing lights on"]
    assert xp.datarefs[SW + "landing_lights_switch"][:4] == [1, 1, 1, 1]
    knob = switches.B738_TAXI
    taxi = selector(xp, knob.position, knob.up, knob.down, 0, 2)
    say(plugin, "taxi lights on", "toggle_lights", {"light": "taxi", "switch": "on"})
    assert run(plugin, xp, taxi) == ["Taxi lights on"]
    assert xp.datarefs[knob.position] == 2


def test_737_speed_brake_lever_stops_at_the_flight_detent_in_the_air(sim):
    _, plugin, xp = sim
    load_737(plugin, xp)
    say(plugin, "speed brake armed", "set_speed_brake", {"speedbrake": "armed"})
    assert run(plugin, xp) == ["Speed brake armed"]
    assert xp.datarefs[switches.B738_SPEEDBRAKE] == 0.0889
    xp.datarefs["sim/flightmodel/failures/onground_any"] = 0
    xp.datarefs["sim/flightmodel/position/y_agl"] = 10000 * 0.3048
    say(plugin, "speed brake up", "set_speed_brake", {"speedbrake": "up"})
    assert run(plugin, xp) == ["Speed brake flight detent"]
    assert xp.datarefs[switches.B738_SPEEDBRAKE] == 0.667


# --- Airbus A330-300 -------------------------------------------------------------

A333 = (
    switches.A333_NAV.down,
    switches.A333_NAV.up,
    switches.A333_SEATBELT.down,
    switches.A333_SEATBELT.up,
    switches.A333_NO_SMOKING.down,
    switches.A333_NO_SMOKING.up,
    "laminar/A333/toggle_switch/strobe_pos_on",
    "laminar/A333/toggle_switch/strobe_pos_auto",
    "laminar/A333/toggle_switch/strobe_pos_off",
)


def load_a330(plugin, xp):
    assert "a333_switches" in aircraft(plugin, xp, engine_type=JET, commands=A333).overrides
    for knob in (switches.A333_NAV, switches.A333_SEATBELT, switches.A333_NO_SMOKING):
        xp.datarefs[knob.position] = 0
    for setting, value in (("on", 2), ("auto", 1), ("off", 0)):
        xp.effects[f"laminar/A333/toggle_switch/strobe_pos_{setting}"] = lambda x, v=value: (
            x.datarefs.update({switches.A333_STROBE: v})
        )

    # LO / MED / MAX toggle: the lit one turns the autobrake off.
    def button(level, also=()):
        def press(x):
            now = x.datarefs[SW + "auto_brake_level"]
            x.datarefs[SW + "auto_brake_level"] = 1 if now in (level, *also) else level

        return press

    xp.effects["sim/flight_controls/brakes_1_auto"] = button(3)
    xp.effects["sim/flight_controls/brakes_2_auto"] = button(4)
    xp.effects["sim/flight_controls/brakes_rto_auto"] = button(0, also=(5,))


def test_a330_strobes_auto(sim):
    _, plugin, xp = sim
    load_a330(plugin, xp)
    say(plugin, "strobes auto", "toggle_lights", {"light": "strobe", "switch": "auto"})
    assert run(plugin, xp) == ["Strobes auto"]
    assert xp.datarefs[switches.A333_STROBE] == 1
    assert xp.commands_fired == ["laminar/A333/toggle_switch/strobe_pos_auto"]


def test_a330_logo_is_the_nav_switch_and_no_yaw_damper(sim):
    _, plugin, xp = sim
    load_a330(plugin, xp)
    knob = switches.A333_NAV
    nav = selector(xp, knob.position, knob.up, knob.down, 0, 2)
    say(plugin, "logo lights on", "toggle_lights", {"light": "logo", "switch": "on"})
    assert run(plugin, xp, nav) == ["Logo lights on"]
    assert xp.datarefs[knob.position] == 1
    assert say(plugin, "yaw damper", "toggle_yaw_damper", {"state": "on"}) == [
        "This aircraft has no yaw damper switch"
    ]


def test_a330_autobrake_buttons_toggle(sim):
    _, plugin, xp = sim
    load_a330(plugin, xp)
    say(plugin, "auto brake low", "set_autobrake", {"autobrake": "low"})
    assert run(plugin, xp) == ["Auto brake low"]
    assert xp.datarefs[SW + "auto_brake_level"] == 3
    assert say(plugin, "auto brake", "set_autobrake", {"autobrake_level": "1"}) == [
        "Auto brake already low"
    ]
    assert say(plugin, "auto brake", "set_autobrake", {"autobrake_level": "3"}) == [
        "This aircraft has no auto brake three"
    ]
    say(plugin, "auto brake off", "set_autobrake", {"autobrake": "off"})
    assert run(plugin, xp) == ["Auto brake off"]
    assert xp.datarefs[SW + "auto_brake_level"] == 1
    assert xp.commands_fired == ["sim/flight_controls/brakes_1_auto"] * 2  # the lit one again


def test_a330_seat_belt_and_no_smoking_signs(sim):
    _, plugin, xp = sim
    load_a330(plugin, xp)
    belt, smoke = switches.A333_SEATBELT, switches.A333_NO_SMOKING
    turns = (
        selector(xp, belt.position, belt.up, belt.down, 0, 2),
        selector(xp, smoke.position, smoke.up, smoke.down, 0, 2),
    )
    say(plugin, "sign", "toggle_signs", {"sign": "seat belt", "switch": "on"})
    assert run(plugin, xp, *turns) == ["Seat belt sign on"]
    say(plugin, "sign", "toggle_signs", {"sign": "no smoking", "switch": "auto"})
    assert run(plugin, xp, *turns) == ["No smoking sign auto"]
    assert (xp.datarefs[belt.position], xp.datarefs[smoke.position]) == (2, 1)
    assert events(plugin, "knob_step_end")[-1]["label"] == "No smoking sign"
