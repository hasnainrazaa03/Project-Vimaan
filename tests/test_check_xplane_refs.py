"""scripts/check_xplane_refs.py: finding the controls aircraft create."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import check_xplane_refs  # noqa: E402


def test_commands_created_by_aircraft_scripts_are_found(tmp_path):
    scripts = tmp_path / "Aircraft" / "Laminar Research" / "Airbus A330-300" / "plugins" / "xlua"
    scripts.mkdir(parents=True)
    (scripts / "A333.autopilot.lua").write_text(
        'cmd = create_command("laminar/A333/autopilot/a_thr_toggle", "A/THR", handler)\n'
        "other = create_command( 'laminar/A333/autopilot/speed_knob_pull' , 'pull', h)\n"
        '-- replace_command("sim/autopilot/servos_toggle", h) is not a creation\n'
    )
    (scripts / "notes.txt").write_text('create_command("not/a/lua/file")')
    assert check_xplane_refs.aircraft_commands(str(tmp_path)) == {
        "laminar/A333/autopilot/a_thr_toggle",
        "laminar/A333/autopilot/speed_knob_pull",
    }


def test_new_modules_are_checked_by_default():
    assert "ML/vimaan_nlu/aircraft.py" in check_xplane_refs.DEFAULT_SOURCES
    assert "ML/vimaan_nlu/probe.py" in check_xplane_refs.DEFAULT_SOURCES
