"""Phase L.4 checklists in the plugin: read out, each item checked against
the sim, wrong ones offered and set through the usual handler and safety
rules, and suggested at phase changes when switched on.

The fake sim starts as a Baron-like piston twin parked with everything off;
the 737 and the A330 are loaded with their own switches (test_switches)."""

import time

import pytest
import test_plugin
import test_switches
from test_plugin import events, say
from vimaan_nlu.flight_phase import FlightSample

SW = "sim/cockpit2/switches/"
TAIL = "Say confirm, or cancel."


@pytest.fixture
def sim(monkeypatch, tmp_path):
    test_plugin.NLU.clear()
    module, plugin, xp = test_plugin._load_plugin(monkeypatch, tmp_path)
    # No pauses: each item as soon as the last is said.
    monkeypatch.setattr(module.checklists, "GAP_SEC", 0.0)
    monkeypatch.setattr(module.checklists, "WORD_SEC", 0.0)
    monkeypatch.setattr(module, "VERIFY_DELAY_SEC", 0.0)  # read-backs at once
    plugin._speaking_until = 0.0  # what it said at startup, at the real pace
    yield module, plugin, xp
    plugin.XPluginStop()


def start(plugin, name):
    return say(plugin, f"{name} checklist", "run_checklist", {"checklist": name})


def read(plugin, xp, loops=40):
    """Run the flight loop until the checklist ends or a new offer waits."""
    run = plugin._checklist
    waiting = run["offer"] if run is not None else None
    out = []
    for _ in range(loops):
        before = len(xp.spoken)
        plugin._flight_loop()
        out += xp.spoken[before:]
        run = plugin._checklist
        if run is None:
            break
        new = run["offer"]
        if new is not None and new is not waiting and not new["answered"]:
            break
    return out


def offer(text):
    return f"{text} {TAIL}"


def test_a_checklist_is_read_checked_and_a_wrong_item_set_on_confirm(sim):
    _, plugin, xp = sim
    assert start(plugin, "before landing") == ["Before landing checklist"]
    assert read(plugin, xp) == [
        "Gear down, checked",
        "Flaps up. Not set for landing",  # checked only
        offer("Landing lights off. Set landing lights on?"),
    ]
    assert xp.datarefs[SW + "landing_lights_on"] == 0  # nothing set unasked
    assert say(plugin, "confirm") == []  # the switch is stepped, then read back
    assert read(plugin, xp) == [
        "Landing lights on",  # the handler's own read-back, then the summary
        "Before landing checklist done, one item open: flaps",
    ]
    assert xp.datarefs[SW + "landing_lights_on"] == 1
    assert plugin._checklist is None
    end = events(plugin, "checklist_end")[-1]
    assert end["reason"] == "done" and end["open"] == ["flaps"]
    assert events(plugin, "checklist_offer")[-1]["result"] == "set"


def test_cancel_declines_an_offer_and_the_checklist_goes_on(sim):
    _, plugin, xp = sim
    start(plugin, "approach")
    assert read(plugin, xp) == [offer("Seat belt sign off. Set the seat belt sign on?")]
    assert say(plugin, "cancel") == ["Cancelled"]
    assert read(plugin, xp) == [offer("Landing lights off. Set landing lights on?")]
    say(plugin, "confirm")
    assert read(plugin, xp) == [
        "Landing lights on",
        "Approach checklist done, one item open: seat belt sign",
    ]
    assert xp.datarefs[SW + "fasten_seat_belts"] == 0


def test_cancel_between_items_stops_the_checklist(sim):
    _, plugin, xp = sim
    start(plugin, "before takeoff")
    assert say(plugin, "cancel") == ["Checklist cancelled"]
    assert read(plugin, xp) == [] and plugin._checklist is None
    assert events(plugin, "checklist_end")[-1]["reason"] == "cancelled"
    assert say(plugin, "cancel") == ["Nothing to cancel"]


def test_no_answer_leaves_the_item_open(sim, monkeypatch):
    module, plugin, xp = sim
    monkeypatch.setattr(module.checklists, "OFFER_WAIT_SEC", 0.0)
    start(plugin, "approach")
    lines = []
    for _ in range(10):
        lines += read(plugin, xp, loops=1)
    assert lines == [
        offer("Seat belt sign off. Set the seat belt sign on?"),
        offer("Landing lights off. Set landing lights on?"),
        "Approach checklist done, 2 items open: seat belt sign and landing lights",
    ]
    assert say(plugin, "confirm") == ["Nothing to override"]


def test_the_pilot_saying_the_command_sets_the_item(sim):
    _, plugin, xp = sim
    start(plugin, "approach")
    read(plugin, xp)
    say(plugin, "seat belt sign on", "toggle_signs", {"sign": "seat belt", "switch": "on"})
    assert read(plugin, xp) == [
        "Seat belt sign on",
        offer("Landing lights off. Set landing lights on?"),
    ]
    say(plugin, "landing lights on", "toggle_lights", {"light": "landing", "switch": "on"})
    assert read(plugin, xp) == ["Landing lights on", "Approach checklist complete"]


def test_another_command_leaves_the_offer_open(sim):
    _, plugin, xp = sim
    start(plugin, "approach")
    read(plugin, xp)
    say(plugin, "heading two seven zero", "set_autopilot_heading", {"degrees": "270"})
    assert read(plugin, xp)[0] == offer("Landing lights off. Set landing lights on?")
    assert plugin._checklist["open"] == ["seat belt sign"]


def test_an_offer_held_by_a_safety_rule_waits_for_the_hold(sim):
    _, plugin, xp = sim
    start(plugin, "after takeoff")  # on the ground: gear up is held
    assert read(plugin, xp) == [offer("Gear down. Raise the gear?")]
    held = say(plugin, "confirm")
    assert held and held[-1].endswith("Say override, or repeat to confirm.")
    assert read(plugin, xp, loops=5) == []  # nothing said over the hold
    assert say(plugin, "cancel") == ["Cancelled"]
    assert "sim/flight_controls/landing_gear_up" not in xp.commands_fired
    assert read(plugin, xp) == [
        "Flaps up, checked",
        "After takeoff checklist done, one item open: gear",
    ]


def test_an_overridden_hold_sets_the_item(sim):
    _, plugin, xp = sim
    start(plugin, "after takeoff")
    read(plugin, xp)
    say(plugin, "confirm")
    say(plugin, "override")
    assert "sim/flight_controls/landing_gear_up" in xp.commands_fired
    assert read(plugin, xp)[-1] == "After takeoff checklist complete"


def test_items_the_aircraft_lacks_are_skipped(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/aircraft/gear/acf_gear_retract"] = 0
    xp.datarefs["sim/aircraft/parts/acf_sbrkEQ"] = 0
    start(plugin, "after landing")
    start_event = events(plugin, "checklist_start")[-1]
    assert "speed brakes" in start_event["skipped"]
    assert "auto brake" in start_event["skipped"]  # a piston: no auto brake
    assert start(plugin, "before landing") == ["Before landing checklist"]
    assert events(plugin, "checklist_start")[-1]["skipped"] == ["gear"]
    assert read(plugin, xp)[0] == "Flaps up. Not set for landing"


def test_an_unreadable_item_is_left_open(sim):
    _, plugin, xp = sim
    del xp.datarefs["sim/cockpit2/autopilot/servos_on"]
    start(plugin, "before takeoff")
    for _ in range(12):
        read(plugin, xp)
        if plugin._checklist is None:
            break
        say(plugin, "cancel")
    end = events(plugin, "checklist_end")[-1]
    assert "autopilot" in end["open"]
    item = [e for e in events(plugin, "checklist_item") if e["item"] == "autopilot"][-1]
    assert item["said"] == "I can't read the autopilot"


def test_the_737_offers_the_first_notch_and_rto(sim):
    _, plugin, xp = sim
    test_switches.load_737(plugin, xp)
    start(plugin, "before takeoff")
    assert read(plugin, xp) == [offer("Flaps up. Set the first notch of flaps?")]
    say(plugin, "confirm")
    assert "sim/flight_controls/flaps_down" in xp.commands_fired
    for _ in range(3):  # transponder, landing lights, strobes
        read(plugin, xp)
        say(plugin, "cancel")
    assert read(plugin, xp) == [offer("Auto brake off. Set auto brake R T O?")]
    say(plugin, "confirm")
    assert read(plugin, xp)[0] == "Auto brake R T O"
    assert xp.datarefs[SW + "auto_brake_level"] == 0  # RTO


def test_the_a330_auto_brake_is_offered_by_its_own_name(sim):
    _, plugin, xp = sim
    test_switches.load_a330(plugin, xp)
    xp.datarefs[SW + "auto_brake_level"] = 1  # off
    start(plugin, "approach")
    for _ in range(4):
        read(plugin, xp)
        if plugin._checklist["items"][plugin._checklist["i"] - 1].name == "auto brake":
            break
        say(plugin, "cancel")
    assert xp.spoken[-1] == offer("Auto brake off. Set auto brake low?")


def test_shutdown_with_the_engines_running(sim):
    _, plugin, xp = sim
    xp.datarefs["sim/flightmodel/controls/parkbrake"] = 1.0
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][:2] = [1, 1]
    xp.datarefs[SW + "beacon_on"] = 1
    start(plugin, "shutdown")
    assert read(plugin, xp) == [
        "Parking brake set, checked",
        offer("Both engines running. Shut down the engines?"),
    ]
    say(plugin, "cancel")
    assert read(plugin, xp)[0] == "Beacon on, until the engines are off"
    assert events(plugin, "checklist_end")[-1]["open"] == ["engines", "beacon"]


def test_asked_again_starts_afresh_and_a_new_flight_stops_it(sim):
    _, plugin, xp = sim
    start(plugin, "approach")
    read(plugin, xp)
    assert start(plugin, "before landing") == ["Before landing checklist"]
    assert events(plugin, "checklist_end")[-1]["reason"] == "replaced"
    assert plugin._checklist["name"] == "before landing"
    plugin._new_flight()
    assert plugin._checklist is None
    assert events(plugin, "checklist_end")[-1]["reason"] == "new_flight"


def test_which_checklist(sim):
    _, plugin, xp = sim
    assert say(plugin, "checklist", "run_checklist", {}) == ["Which checklist?"]
    assert plugin._checklist is None and xp.commands_fired == []


def sample(gear=1.0):
    return FlightSample(
        t=0.0,
        on_ground=True,
        agl_ft=0.0,
        gs_kt=0.0,
        ias_kt=0.0,
        vs_fpm=0.0,
        gear=gear,
        flaps=0.0,
        altitude_ft=0.0,
    )


def taxi_out(plugin, xp):
    """Parked with the engines off, an engine start, then taxiing."""
    spoken = len(xp.spoken)
    plugin.phase.phase = "parked"
    plugin._suggest_checklist(sample())
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 1
    plugin._suggest_checklist(sample())
    plugin.phase.phase = "taxi"
    plugin._suggest_checklist(sample())
    return xp.spoken[spoken:]


def test_suggestions_are_off_unless_switched_on(sim):
    _, plugin, xp = sim
    assert taxi_out(plugin, xp) == []
    assert "before takeoff" in plugin.suggestions.said  # tracked all the same
    assert events(plugin, "checklist_suggested") == []


def test_a_suggested_checklist_starts_on_confirm(sim):
    _, plugin, xp = sim
    plugin._on_checklist_suggestions_cmd(None, test_plugin.make_xp().CommandBegin, None)
    assert xp.spoken[-1] == "Checklist suggestions on"
    assert taxi_out(plugin, xp) == [f"Before takeoff checklist? {TAIL}"]
    assert say(plugin, "confirm") == ["Before takeoff checklist"]
    assert plugin._checklist["name"] == "before takeoff"


def test_a_suggestion_cancelled_or_unanswered(sim, monkeypatch):
    module, plugin, xp = sim
    plugin._suggest_checklists = True
    taxi_out(plugin, xp)
    assert say(plugin, "cancel") == ["Cancelled"]
    assert plugin._suggested is None and plugin._checklist is None
    plugin.suggestions.reset()
    monkeypatch.setattr(module.checklists, "OFFER_WAIT_SEC", 0.0)
    xp.datarefs["sim/flightmodel/engine/ENGN_running"][0] = 0
    taxi_out(plugin, xp)
    plugin._flight_loop()
    assert plugin._suggested is None
    assert say(plugin, "confirm") == ["Nothing to override"]


def test_no_suggestion_over_a_running_checklist(sim):
    _, plugin, xp = sim
    plugin._suggest_checklists = True
    start(plugin, "shutdown")
    assert taxi_out(plugin, xp) == []
    assert events(plugin, "checklist_suggested")[-1]["said"] is False


def test_never_said_over_what_vimaan_is_saying(sim):
    # An answer, a read-back or a callout said meanwhile: the next item waits.
    _, plugin, xp = sim
    start(plugin, "before landing")
    plugin._speaking_until = time.perf_counter() + 60.0
    assert read(plugin, xp, loops=5) == []
    plugin._speaking_until = 0.0
    assert read(plugin, xp)[0] == "Gear down, checked"


def test_a_suggestion_waits_for_a_callout(sim):
    _, plugin, xp = sim
    plugin._suggest_checklists = True
    plugin._speaking_until = time.perf_counter() + 60.0
    assert taxi_out(plugin, xp) == []
    plugin._flight_loop()
    assert xp.spoken[-1] != f"Before takeoff checklist? {TAIL}"
    plugin._speaking_until = 0.0
    plugin._flight_loop()
    assert xp.spoken[-1] == f"Before takeoff checklist? {TAIL}"
    assert say(plugin, "confirm") == ["Before takeoff checklist"]
