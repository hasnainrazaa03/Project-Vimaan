"""vimaan_nlu.aircraft: layered profiles that work on any aircraft."""

import pytest
from vimaan_nlu.aircraft import DEFAULT_COMMANDS, OVERRIDES, AircraftInfo, override_names, resolve

# Values as X-Plane 12.4.3 reports them for three Laminar aircraft.
BARON = AircraftInfo(
    icao="BE58",
    acf_path="/X-Plane 12/Aircraft/Laminar Research/Beechcraft Baron 58/Baron_58.acf",
    num_engines=2,
    engine_types=(1, 1),
    flap_detents=2,
    flap_degrees=(0.0, 13.0, 30.0, 27.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
B738 = AircraftInfo(
    icao="B738",
    acf_path="/X-Plane 12/Aircraft/Laminar Research/Boeing 737-800/b738.acf",
    num_engines=2,
    engine_types=(7, 7),
    flap_detents=8,
    flap_degrees=(0.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 18.0, 40.0, 0.0),
)
A333_COMMANDS = {
    "laminar/A333/autopilot/speed_knob_pull",
    "laminar/A333/autopilot/a_thr_toggle",
}


def nothing(_name):
    return False


@pytest.mark.parametrize(
    ("types", "kind", "start"),
    [
        ((1, 1), "piston", "piston"),
        ((0,), "piston", "piston"),
        ((7, 7), "jet", "jet"),
        ((5,), "jet", "jet"),
        ((9, 9), "turboprop", "turboprop"),  # King Air C90B
        ((3,), "electric", "electric"),
        ((), "unknown", "piston"),  # unreadable: start as today, logged unknown
        ((42,), "unknown", "piston"),
        ((1, 7), "mixed", "piston"),
    ],
)
def test_engine_start_follows_engine_type_not_name(types, kind, start):
    info = AircraftInfo(icao="ANY", engine_types=types)
    profile = resolve(info, nothing)
    assert info.engine_kind == kind and profile.engine_start == start


def test_an_aircraft_with_no_custom_controls_gets_the_standard_ones():
    profile = resolve(BARON, nothing)
    assert profile.overrides == ()
    assert dict(profile.commands) == dict(DEFAULT_COMMANDS)
    assert profile.command("autopilot_toggle") == "sim/autopilot/servos_toggle"


def test_override_applies_only_when_its_controls_exist():
    profile = resolve(AircraftInfo(icao="A333"), A333_COMMANDS.__contains__)
    assert profile.overrides == ("airbus_fcu",)
    assert profile.command("autothrottle_toggle") == "laminar/A333/autopilot/a_thr_toggle"
    assert profile.command("autothrottle_on") is None
    assert profile.command("speed_select") == "laminar/A333/autopilot/speed_knob_pull"


def test_name_alone_never_selects_an_override():
    # An aircraft calling itself A333 without the A330's controls (an add-on
    # reusing the code) gets the standard controls.
    profile = resolve(AircraftInfo(icao="A333", author="Laminar Research"), nothing)
    assert profile.overrides == ()
    assert profile.command("autothrottle_toggle") == "sim/autopilot/autothrottle_toggle"


def test_partial_match_does_not_apply():
    has_one = {"laminar/A333/autopilot/speed_knob_pull"}.__contains__
    assert resolve(AircraftInfo(), has_one).overrides == ()


def test_probe_results_decide_support():
    profile = resolve(BARON, nothing, {"mode_vertical_speed": "no_change", "mode_heading": "works"})
    assert profile.supports("mode_heading") is True
    assert profile.supports("mode_vertical_speed") is False
    assert profile.supports("mode_level_change") is None  # nobody has probed it
    assert profile.supports("speed_select") is False  # no such control here


@pytest.mark.parametrize(
    ("info", "notches"),
    [
        (BARON, (0.0, 13.0, 30.0)),
        (B738, (0.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 18.0, 40.0)),
        (AircraftInfo(flap_detents=2, flap_degrees=(0.0, 10.0)), ()),  # too short: unknown
        (AircraftInfo(flap_detents=None), ()),
    ],
)
def test_flap_notch_degrees(info, notches):
    assert info.flap_notch_degrees() == notches


@pytest.mark.parametrize(
    ("path", "key"),
    [
        (BARON.acf_path, "Beechcraft_Baron_58__Baron_58"),
        (
            "C:\\X-Plane 12\\Aircraft\\Laminar Research\\Airbus A330-300\\A330.acf",
            "Airbus_A330-300__A330",
        ),
        ("", "BE58"),
    ],
)
def test_key_is_folder_and_file(path, key):
    assert AircraftInfo(icao="BE58", acf_path=path).key == key


def test_summary_is_json_ready():
    import json

    summary = resolve(B738, nothing).summary()
    assert json.loads(json.dumps(summary))["engine_start"] == "jet"
    assert summary["flap_notch_degrees"][5] == 12.0


def test_every_override_cites_a_source_and_only_known_roles():
    for override in OVERRIDES:
        assert override.source and override.requires
        assert set(override.commands) <= set(DEFAULT_COMMANDS), override.name
    assert "laminar/A333/autopilot/a_thr_toggle" in override_names()


B738_CONTROLS = {
    "laminar/B738/engines/condition_lever_1_idle",
    "laminar/B738/engines/condition_lever_2_idle",
}


@pytest.mark.parametrize(
    ("label", "notch"),
    [(1, 1), (2, 2), (5, 3), (10, 4), (15, 5), (25, 6), (30, 7), (40, 8), (3, None)],
)
def test_737_flap_labels_not_angles(label, notch):
    assert resolve(B738, B738_CONTROLS.__contains__).flap_notch(label) == notch


@pytest.mark.parametrize(
    ("label", "notch"), [(15, 1), (10, 1), (30, 2), (1, 1), (2, 2), (25, None), (40, None)]
)
def test_baron_flaps_by_angle_or_notch_number(label, notch):
    assert resolve(BARON, nothing).flap_notch(label) == notch


def test_labels_ignored_when_the_detent_count_disagrees():
    # A 737 variant with a different handle: fall back to angles / numbers.
    odd = AircraftInfo(flap_detents=5, flap_degrees=(0.0, 1.0, 5.0, 15.0, 30.0, 40.0))
    profile = resolve(odd, B738_CONTROLS.__contains__)
    assert profile.flap_notch(15) == 3 and profile.flap_notch(25) is None


def test_airbus_settings_one_two_three():
    a333 = AircraftInfo(flap_detents=4, flap_degrees=(0.0, 10.0, 15.0, 20.0, 35.0))
    profile = resolve(a333, A333_COMMANDS.__contains__)
    assert [profile.flap_notch(n) for n in (1, 2, 3, 5)] == [1, 2, 3, None]


def test_files_mention_finds_an_autothrottle_in_the_acf_scripts_or_objects(tmp_path):
    from vimaan_nlu.aircraft import files_mention

    def plane(name, files):
        d = tmp_path / name
        for rel, text in files.items():
            f = d / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        return str(d / "plane.acf")

    assert (
        files_mention(plane("jet", {"plane.acf": "P sim/cockpit2/autopilot/AutoThrottle_on"}))
        is True
    )
    md82 = plane(
        "md82", {"plane.acf": "x", "objects/panel.obj": "ANIM sim/autopilot/autothrottle_toggle"}
    )
    assert files_mention(md82) is True
    baron = plane("baron", {"plane.acf": "x", "plugins/xlua/scripts/a.lua": "throttle_ratio"})
    assert files_mention(baron) is False
    # Liveries are skipped; a missing folder is unknown.
    livery = plane("livery", {"plane.acf": "x", "liveries/red/objects/x.obj": "autothrottle"})
    assert files_mention(livery) is False
    assert files_mention(str(tmp_path / "nowhere" / "x.acf")) is None


def test_own_controls_measured_as_no_change_are_tried_not_refused():
    from vimaan_nlu.aircraft import AircraftInfo, resolve

    b738 = {"laminar/B738/autopilot/cmd_a_press", "laminar/B738/autopilot/hdg_sel_press"}
    probed = {"mode_vnav": "no_change", "mode_nav": "no_change"}
    p = resolve(AircraftInfo(), lambda n: n in b738 or n.startswith("sim/"), probed)
    assert p.supports("mode_vnav") is None  # the 737's own VNAV button: try, verify
    assert p.supports("mode_nav") is None  # also its own (VOR/LOC)
    generic = resolve(AircraftInfo(), lambda n: n.startswith("sim/"), probed)
    assert generic.supports("mode_vnav") is False


# ---------------------------------------------------------------------------
# Fixes from the A2 runs (2026-11-12)
# ---------------------------------------------------------------------------
from vimaan_nlu.aircraft import acf_flap_limits, flap_limit_kt  # noqa: E402


def test_overrides_need_their_aircrafts_datarefs_too():
    has_all = lambda _name: True  # noqa: E731
    gone = lambda _name: False  # noqa: E731
    assert resolve(AircraftInfo(), has_all, has_dataref=has_all).overrides
    assert resolve(AircraftInfo(), has_all, has_dataref=gone).overrides == ()
    assert all(o.live for o in OVERRIDES), "every override names a dataref its aircraft owns"


def test_flap_limits_from_an_acf(tmp_path):
    acf = tmp_path / "b738.acf"
    acf.write_bytes(
        b"I\n1200 version\nP acf/_Vfe1_kts 250.000000000\nP acf/_Vfem_kts 162.000000000\n"
        b"P acf/_Vfe1_kts_extra 1\n"
    )
    assert acf_flap_limits(str(acf)) == {"first": 250.0, "full": 162.0}
    assert acf_flap_limits(str(tmp_path / "missing.acf")) == {}


def test_flap_limit_per_notch():
    # The 737: flaps 1..40 over 8 notches, first-notch 250, full 162.
    limits = [flap_limit_kt(n, 8, 250.0, 162.0) for n in range(1, 9)]
    assert limits == [250.0, 237.4, 224.9, 212.3, 199.7, 187.1, 174.6, 162.0]
    assert flap_limit_kt(2, 2, 152.0, 122.0) == 122.0  # Baron full
    assert flap_limit_kt(1, 2, 152.0, 122.0) == 152.0  # Baron approach
    assert flap_limit_kt(1, 2, 152.0, None) == 152.0  # no .acf yet: the published one
    assert flap_limit_kt(1, 2, None, 122.0) == 122.0
    assert flap_limit_kt(0, 2, 152.0, 122.0) is None
