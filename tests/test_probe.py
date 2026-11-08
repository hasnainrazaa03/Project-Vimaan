"""vimaan_nlu.probe: classifying, restoring and saving probe steps."""

import json

import pytest
from vimaan_nlu import probe
from vimaan_nlu.aircraft import DEFAULT_COMMANDS

STEP = {s.role: s for s in probe.STEPS}
# Autopilot on in heading and altitude hold, the state the probe starts from.
BASE = {
    "servos_on": 1,
    "lateral_mode": 1,
    "vertical_mode": 6,
    "heading_status": 2,
    "nav_status": 0,
    "approach_status": 0,
    "backcourse_status": 0,
    "altitude_armed": 0,
    "vs_status": 0,
    "autothrottle_enabled": 0,
    "airspeed_is_mach": 0,
    "speed_dial": 150.0,
    "baro": 30.12,
}


def test_every_step_names_a_known_role_or_state():
    for step in probe.STEPS:
        assert step.where in ("air", "ground")
        assert set(step.watch) <= set(probe.STATE), step.role
        assert step.write is None or step.write in probe.STATE
        if step.write is None:
            assert step.role in DEFAULT_COMMANDS, step.role
    assert set(probe.LATERAL_ROLES.values()) | set(probe.VERTICAL_ROLES.values()) <= set(
        DEFAULT_COMMANDS
    )


@pytest.mark.parametrize(
    ("on_ground", "agl", "gs", "where"),
    [
        (True, 0.0, 0.0, "ground"),
        (True, 0.0, 12.0, None),  # taxiing
        (False, 3000.0, 140.0, "air"),
        (False, 800.0, 120.0, None),  # too low
        (None, None, None, None),  # unknown: never
    ],
)
def test_situation(on_ground, agl, gs, where):
    assert probe.situation(on_ground, agl, gs) == where


def test_air_steps_need_the_autopilot_and_ground_steps_the_brake():
    assert probe.ready_for(STEP["mode_heading"], {**BASE, "servos_on": 0}, True)
    assert probe.ready_for(STEP["mode_heading"], BASE, None) is None
    assert probe.ready_for(STEP["flap_handle"], BASE, False)
    assert probe.ready_for(STEP["flap_handle"], BASE, True) is None


def test_mode_that_changes_works_and_one_that_doesnt_is_no_change():
    vs = STEP["mode_vertical_speed"]
    after = {**BASE, "vertical_mode": 4, "vs_status": 2}
    assert probe.classify(vs, BASE, after, found=True) == "works"
    assert probe.classify(vs, BASE, BASE, found=True) == "no_change"
    assert probe.classify(vs, BASE, BASE, found=False) == "missing"


def test_a_write_works_only_if_the_aircraft_keeps_it():
    dial = STEP["dial_speed"]
    assert probe.write_value(dial, BASE, None) == 155.0
    assert probe.classify(dial, BASE, {**BASE, "speed_dial": 155.0}, True, 155.0) == "works"
    # The aircraft's own logic put it back: the dial isn't ours to set.
    assert probe.classify(dial, BASE, BASE, True, 155.0) == "no_change"


def test_flap_step_moves_one_notch():
    assert probe.write_value(STEP["flap_handle"], {"flap_handle": 0.0}, 4) == 0.25
    assert probe.write_value(STEP["flap_handle"], {"flap_handle": 0.0}, None) is None


def test_baro_std():
    std = STEP["baro_std"]
    assert probe.classify(std, BASE, {**BASE, "baro": 29.92}, True) == "works"
    assert probe.classify(std, BASE, BASE, True) == "no_change"
    already = {**BASE, "baro": 29.92}
    assert probe.classify(std, already, already, True) == "inconclusive"


def test_restored_compares_modes_and_the_written_value():
    dial = STEP["dial_speed"]
    assert probe.restored(dial, BASE, dict(BASE))
    assert not probe.restored(dial, BASE, {**BASE, "speed_dial": 155.0})
    assert not probe.restored(STEP["mode_heading"], BASE, {**BASE, "lateral_mode": 0})


def test_mode_fix_presses_the_previous_mode():
    # Pressing V/S left pitch mode behind; ALT was on before.
    now = {**BASE, "vertical_mode": 3}
    assert probe.mode_fix(BASE, now) == ["mode_altitude_hold"]
    assert probe.mode_fix(BASE, {**BASE, "lateral_mode": 0, "vertical_mode": 3}) == [
        "mode_heading",
        "mode_altitude_hold",
    ]
    # Roll mode before: there is no button for "roll", so nothing to press.
    assert probe.mode_fix({**BASE, "lateral_mode": 0}, {**BASE, "lateral_mode": 1}) == []


def test_results_merge_ground_and_air_runs(tmp_path):
    path = probe.results_path(str(tmp_path / "aircraft"), "Beechcraft_Baron_58__Baron_58")
    probe.save_results(path, {"where": "ground", "steps": {"flap_handle": "works"}})
    probe.save_results(
        path,
        {"where": "air", "steps": {"mode_vertical_speed": "no_change", "baro_std": "inconclusive"}},
    )
    assert probe.load_results(path) == {
        "flap_handle": "works",
        "mode_vertical_speed": "no_change",
    }
    assert json.loads(open(path).read())["where"] == "air"


def test_missing_or_broken_results_file_means_no_results(tmp_path):
    assert probe.load_results(str(tmp_path / "none.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert probe.load_results(str(bad)) == {}


def test_aircraft_rounding_still_counts_as_restored():
    # The 737 put the altimeter back at 29.92 for 29.9212: the probe stopped.
    std = STEP["baro_std"]
    assert probe.restored(std, {**BASE, "baro": 29.9212}, {**BASE, "baro": 29.92})
    assert not probe.restored(std, {**BASE, "baro": 29.92}, {**BASE, "baro": 30.12})
    flaps = STEP["flap_handle"]
    assert probe.restored(flaps, {**BASE, "flap_handle": 0.5}, {**BASE, "flap_handle": 0.505})


def test_mach_dial_tolerance_is_tighter():
    dial = STEP["dial_speed"]
    assert probe.restored(dial, {**BASE, "speed_dial": 250.0}, {**BASE, "speed_dial": 250.3})
    assert not probe.restored(dial, {**BASE, "speed_dial": 0.78}, {**BASE, "speed_dial": 0.80})


def test_autopilot_dropping_mid_step_is_detected():
    heading = STEP["mode_heading"]
    assert probe.autopilot_dropped(heading, BASE, {**BASE, "servos_on": 0})
    assert not probe.autopilot_dropped(heading, BASE, BASE)
    assert not probe.autopilot_dropped(STEP["flap_handle"], BASE, {**BASE, "servos_on": 0})


def test_inconclusive_does_not_replace_a_definite_result(tmp_path):
    path = str(tmp_path / "a.json")
    probe.save_results(path, {"steps": {"baro_std": "works"}})
    probe.save_results(path, {"steps": {"baro_std": "inconclusive", "dial_baro": "inconclusive"}})
    steps = json.loads(open(path).read())["steps"]
    assert steps == {"baro_std": "works", "dial_baro": "inconclusive"}
    probe.save_results(path, {"steps": {"baro_std": "no_change"}})
    assert probe.load_results(path)["baro_std"] == "no_change"  # a new definite one does


def test_resume_point_is_kept_per_situation_and_cleared_by_none(tmp_path):
    path = str(tmp_path / "a.json")
    probe.save_results(path, {"resume": {"air": "mode_vnav"}})
    probe.save_results(path, {"resume": {"ground": "flap_handle"}})
    record = probe.load_record(path)
    assert probe.resume_index(record, "air") == 9
    assert probe.resume_index(record, "ground") == 2
    probe.save_results(path, {"resume": {"air": None}})
    record = probe.load_record(path)
    assert probe.resume_index(record, "air") == 0 and record["resume"] == {"ground": "flap_handle"}
    # A role that isn't a step (any more) starts from the beginning.
    assert probe.resume_index({"resume": {"air": "gone"}}, "air") == 0


def test_counts_cover_every_saved_run_for_one_situation():
    record = {"steps": {"mode_heading": "works", "mode_vnav": "no_change", "flap_handle": "works"}}
    assert probe.counts(record, "air") == {"works": 1, "no_change": 1}


def test_restore_presses_modes_first_then_arm_and_autothrottle():
    now = {**BASE, "vertical_mode": 5, "altitude_armed": 1, "autothrottle_enabled": 1}
    assert probe.restore_presses(BASE, now, 0) == ["mode_altitude_hold"]
    assert probe.restore_presses(BASE, now, 1) == ["mode_altitude_arm", "autothrottle_toggle"]
    assert probe.restore_presses(BASE, now, 2) is None
    # Both "on", in different sub-states: nothing to press.
    assert (
        probe.restore_presses(
            {**BASE, "autothrottle_enabled": 1}, {**BASE, "autothrottle_enabled": 3}, 1
        )
        == []
    )


def test_leftovers_and_their_names():
    dial = STEP["dial_altitude"]
    before = {**BASE, "altitude_dial": 8680.0}
    now = {**before, "altitude_armed": 1}
    assert probe.leftovers(dial, before, now) == ["altitude_armed"]
    assert set(probe.leftovers(dial, before, now)) <= probe.HARMLESS_LEFTOVERS
    worse = {**now, "lateral_mode": 0, "nav_status": 1}
    assert probe.describe(probe.leftovers(dial, before, worse)) == (
        "the lateral mode, nav and altitude arm"
    )


def test_caution_only_for_air_steps_without_selectable_modes():
    hdg = STEP["mode_heading"]
    assert probe.caution(hdg, BASE) is None
    assert probe.caution(hdg, {**BASE, "lateral_mode": 0, "vertical_mode": 3})
    assert probe.caution(hdg, {**BASE, "lateral_mode": 2, "vertical_mode": 4}) is None  # NAV, V/S
    assert probe.caution(STEP["flap_handle"], {}) is None
