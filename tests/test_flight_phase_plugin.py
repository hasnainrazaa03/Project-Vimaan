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
