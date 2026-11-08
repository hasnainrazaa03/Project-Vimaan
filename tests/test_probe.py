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
