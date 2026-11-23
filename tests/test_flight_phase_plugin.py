"""Phase F.1 in the plugin: the aircraft sampled once a second, recorded,
and the phase of flight logged when it changes. Observe only."""

import pytest
import test_plugin
from test_plugin import events

AGL = "sim/flightmodel/position/y_agl"  # metres
GS = "sim/flightmodel/position/groundspeed"  # m/s
ON_GROUND = "sim/flightmodel/failures/onground_any"
VS = "sim/flightmodel/position/vh_ind_fpm"
PAUSED = "sim/time/paused"


@pytest.fixture
def sim(monkeypatch, tmp_path):
    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    clock = {"now": 100.0}
    plugin.tel._clock = lambda: clock["now"]
    plugin.tel._t0 = 100.0
    # Loading ran the flight loop once (one sample, on the fake sim's ground).
    plugin.samples_at_start = len(events(plugin, "flight"))
    yield module, plugin, xp, clock
    plugin.XPluginStop()


def seconds(plugin, clock, n):
    """``n`` one-second samples, as the flight loop takes them."""
    for _ in range(n):
        plugin._flight_next = 0.0
        plugin._service_flight()
        clock["now"] += 1.0


def samples(plugin):
    return events(plugin, "flight")[plugin.samples_at_start :]


def on_ground(xp, kt=0.0):
    xp.datarefs.update({ON_GROUND: 1, AGL: 0.0, GS: kt * 0.514444, VS: 0.0, PAUSED: 0})


def flying(xp, agl_ft, vs_fpm=0.0):
    xp.datarefs.update({ON_GROUND: 0, AGL: agl_ft * 0.3048, GS: 120.0, VS: vs_fpm, PAUSED: 0})


def test_the_phase_is_logged_when_it_changes_and_every_sample_recorded(sim):
    _, plugin, xp, clock = sim
    on_ground(xp)
    seconds(plugin, clock, 3)
    on_ground(xp, kt=12.0)
    seconds(plugin, clock, 5)
    changes = events(plugin, "flight_phase")
    assert [(e["previous"], e["phase"]) for e in changes] == [
        ("unknown", "parked"),
        ("parked", "taxi"),
    ]
    assert changes[1]["since"] == 3.0  # dated from when it began
    flight = samples(plugin)
    assert len(flight) == 8 and flight[-1]["phase"] == "taxi" and flight[-1]["gs_kt"] == 12.0
    assert flight[0]["gear"] == 1.0  # the fake sim's gear is down
    assert xp.writes == [] and xp.commands_fired == []  # observe only


def test_samples_go_to_the_events_file_only_not_the_log(sim):
    _, plugin, xp, clock = sim
    on_ground(xp)
    seconds(plugin, clock, 3)
    log = plugin.tel.log_path.read_text()
    assert "[flight_phase]" in log and "[flight]" not in log


def test_nothing_is_sampled_while_paused(sim):
    _, plugin, xp, clock = sim
    on_ground(xp)
    xp.datarefs[PAUSED] = 1
    seconds(plugin, clock, 5)
    assert samples(plugin) == []


def test_a_new_flight_starts_the_phase_again(sim):
    module, plugin, xp, clock = sim
    on_ground(xp)
    seconds(plugin, clock, 3)
    # An air start: the aircraft is put at 5,000 ft. Without the reset that
    # would read as a takeoff (initial climb).
    plugin.XPluginReceiveMessage(None, module.xp.MSG_PLANE_LOADED, 0)
    flying(xp, 5000.0)
    before = len(samples(plugin))
    xp.datarefs[PAUSED] = 1  # loading: paused time doesn't count
    seconds(plugin, clock, 3)
    xp.datarefs[PAUSED] = 0
    # Running, the first samples are skipped: an air start's airspeed reads
    # 0 for a couple of seconds, which would be a stall.
    seconds(plugin, clock, module.FLIGHT_SETTLE_SAMPLES)
    assert len(samples(plugin)) == before
    seconds(plugin, clock, 2)
    assert [e["phase"] for e in events(plugin, "flight_phase")] == ["parked", "cruise"]
    assert events(plugin, "flight_reset")
    assert plugin._bridge_state()["phase"] == "cruise"


def test_the_session_report_shows_the_phases_and_replays_them(sim):
    _, plugin, xp, clock = sim
    on_ground(xp)
    seconds(plugin, clock, 3)
    on_ground(xp, kt=12.0)
    seconds(plugin, clock, 10)
    plugin.XPluginStop()
    text = test_plugin._report_module().report(plugin.tel.events_path)
    assert f"Flight phases ({13 + plugin.samples_at_start} samples, one a second" in text
    assert "parked -> taxi" in text
    assert "replayed through today's tracker: the same phases" in text


def test_each_sample_carries_its_speed_margins_and_the_report_sums_them(sim):
    # F.2: the fake sim is a Baron (Vs 84, Vso 74, Vle 152, Vne 223 kt).
    _, plugin, xp, clock = sim
    flying(xp, 3000.0)
    xp.datarefs["sim/cockpit2/gauges/indicators/airspeed_kts_pilot"] = 160.0
    xp.datarefs["sim/flightmodel2/gear/deploy_ratio"] = [1.0] * 10  # gear down, 8 kt over Vle
    seconds(plugin, clock, 3)
    (first, *_) = samples(plugin)
    assert first["margins"]["gear"] == -8.0
    assert first["margins"]["stall"] == 76.0  # clean (flaps up): Vs 84
    assert first["margins"]["vne"] == 63.0 and first["margins"]["mmo"] is None
    plugin.XPluginStop()
    text = test_plugin._report_module().report(plugin.tel.events_path)
    assert "speed margins, lowest in each phase" in text
    assert "beyond a limit: gear 3 s" in text


def test_the_report_says_where_todays_tracker_reads_a_flight_differently():
    report = test_plugin._report_module()
    logged = [(10.0, "cruise"), (193.0, "approach"), (202.0, "cruise")]
    replayed = [(10.0, "cruise"), (192.0, "descent"), (202.0, "cruise")]
    assert report._phase_differences(logged, replayed) == ["t=193s approach -> descent"]
    assert report._phase_differences(logged, logged) == []
    assert report._phase_differences(logged, logged[:1]) == ["t=193s approach cruise -> -"]


def test_the_safety_rules_read_the_phase(sim):
    # F.4: on an approach "gear up" is held even above 1,000 ft (the fake
    # sim's gear is down: configured, so the approach starts at once).
    _, plugin, xp, clock = sim
    plugin.phase.reset()  # an air start on the approach, not a takeoff
    flying(xp, 1800.0, vs_fpm=-700.0)
    seconds(plugin, clock, 2)
    assert plugin.phase.phase == "approach"
    spoken = test_plugin.say(plugin, "gear up", "toggle_landing_gear", {"state": "up"})
    assert spoken[0].startswith("Confirm gear up")
    hold = events(plugin, "safety_hold")[-1]
    assert hold["rule"] == "gear_up_low_altitude"
    assert hold["context"]["phase"] == "approach"
    assert hold["context"]["stall_clean_kt"] == 84.0  # the fake Baron's Vs, for flaps up


IAS = "sim/cockpit2/gauges/indicators/airspeed_kts_pilot"


def test_callouts_are_off_until_switched_on(sim):
    # F.4: "Speed" within 5 kt of the stall (the fake Baron: Vs 84 kt).
    module, plugin, xp, clock = sim
    plugin.phase.reset()
    flying(xp, 3000.0)
    xp.datarefs[IAS] = 86.0
    seconds(plugin, clock, 5)
    assert events(plugin, "callout") == []  # opt-in: off by default
    assert "Speed" not in xp.spoken
    plugin._on_callouts_cmd(None, module.xp.CommandBegin, None)
    assert "Vimaan callouts on" in xp.spoken
    seconds(plugin, clock, 3)
    (c,) = events(plugin, "callout")  # once, not every second
    assert c["callout"] == "speed" and c["said"] == "Speed" and c["margins"]["stall"] == 2.0
    assert xp.spoken.count("Speed") == 1
    plugin._on_callouts_cmd(None, module.xp.CommandBegin, None)
    assert "Vimaan callouts off" in xp.spoken
    assert events(plugin, "callouts_setting")[-1]["enabled"] is False


def test_the_report_lists_the_callouts_said(sim):
    module, plugin, xp, clock = sim
    plugin.phase.reset()
    plugin._on_callouts_cmd(None, module.xp.CommandBegin, None)
    flying(xp, 3000.0)
    xp.datarefs[IAS] = 86.0
    seconds(plugin, clock, 35)  # said again after 30 s while it holds
    plugin.XPluginStop()
    text = test_plugin._report_module().report(plugin.tel.events_path)
    assert "callouts: Speed 2x (first t=" in text


def test_fixed_gear_is_recorded_as_no_gear_so_level_flight_is_not_an_approach(sim):
    # The C172 and SR22 (fixed gear) read as on an approach in level flight
    # below 2,500 ft: the gear, always down, counted as set up to land (L.2.3).
    _, plugin, xp, clock = sim
    xp.datarefs["sim/aircraft/gear/acf_gear_retract"] = 0
    test_plugin.load_aircraft(plugin, xp, 1)  # a piston single
    plugin.samples_at_start = len(events(plugin, "flight"))
    flying(xp, 1300.0)
    seconds(plugin, clock, 6)
    flight = samples(plugin)
    assert flight[-1]["gear"] is None and flight[-1]["phase"] == "cruise"
    xp.datarefs["sim/cockpit2/controls/flap_handle_request_ratio"] = 0.33  # flaps still count
    seconds(plugin, clock, 5)
    assert samples(plugin)[-1]["phase"] == "approach"


def test_retractable_gear_down_still_counts_as_set_up_to_land(sim):
    _, plugin, xp, clock = sim
    xp.datarefs["sim/aircraft/gear/acf_gear_retract"] = 1
    test_plugin.load_aircraft(plugin, xp, 1)
    plugin.samples_at_start = len(events(plugin, "flight"))
    flying(xp, 1300.0)
    seconds(plugin, clock, 6)
    flight = samples(plugin)
    assert flight[-1]["gear"] == 1.0 and flight[-1]["phase"] == "approach"
