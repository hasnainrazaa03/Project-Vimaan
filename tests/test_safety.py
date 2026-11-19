import pytest

"""Tests for vimaan_nlu.safety (Phase 6B) — interlocks + confirmation gate.

Pure-Python; no simulator. Sim state is supplied via synthetic SafetyContext.
"""

from vimaan_nlu.safety import (
    ConfirmationGate,
    SafetyContext,
    command_token,
    evaluate_safety,
)


class TestGearUpInterlock:
    def test_gear_up_low_altitude_requires_confirmation(self):
        v = evaluate_safety("toggle_landing_gear", {"state": "up"}, SafetyContext(agl_ft=400))
        assert v.requires_confirmation and v.rule == "gear_up_low_altitude"

    def test_gear_up_high_altitude_is_fine(self):
        v = evaluate_safety("toggle_landing_gear", {"state": "up"}, SafetyContext(agl_ft=5000))
        assert not v.requires_confirmation

    def test_gear_down_never_gated(self):
        v = evaluate_safety("toggle_landing_gear", {"state": "down"}, SafetyContext(agl_ft=100))
        assert not v.requires_confirmation

    def test_missing_agl_fails_safe(self):
        v = evaluate_safety("toggle_landing_gear", {"state": "up"}, SafetyContext())
        assert v.requires_confirmation and v.fail_safe


class TestEngineShutdownInterlock:
    def test_engine_off_airborne_requires_confirmation(self):
        v = evaluate_safety("toggle_engine", {"state": "off"}, SafetyContext(agl_ft=3000))
        assert v.requires_confirmation and v.rule == "engine_shutdown_in_flight"

    def test_engine_off_on_ground_is_fine(self):
        ctx = SafetyContext(on_ground=True, agl_ft=0, ias_kt=0)
        v = evaluate_safety("toggle_engine", {"state": "off"}, ctx)
        assert not v.requires_confirmation

    def test_engine_off_unknown_state_fails_safe(self):
        v = evaluate_safety("toggle_engine", {"state": "off"}, SafetyContext())
        assert v.requires_confirmation and v.fail_safe

    def test_engine_start_not_gated(self):
        v = evaluate_safety("toggle_engine", {"state": "on"}, SafetyContext(agl_ft=3000))
        assert not v.requires_confirmation

    def test_airborne_detected_by_speed_when_no_agl(self):
        v = evaluate_safety("toggle_engine", {"state": "off"}, SafetyContext(ias_kt=120))
        assert v.requires_confirmation and not v.fail_safe


class TestPitotHeatOffInterlock:
    PITOT_OFF = {"ice": "pitot", "switch": "off"}

    def test_pitot_heat_off_airborne_requires_confirmation(self):
        v = evaluate_safety("toggle_anti_ice", self.PITOT_OFF, SafetyContext(agl_ft=3000))
        assert v.requires_confirmation and v.rule == "pitot_heat_off_in_flight"

    def test_pitot_heat_off_on_the_ground_is_fine(self):
        ctx = SafetyContext(on_ground=True, agl_ft=0, ias_kt=0)
        assert not evaluate_safety("toggle_anti_ice", self.PITOT_OFF, ctx).requires_confirmation

    def test_unknown_state_fails_safe(self):
        v = evaluate_safety("toggle_anti_ice", self.PITOT_OFF, SafetyContext())
        assert v.requires_confirmation and v.fail_safe

    @pytest.mark.parametrize(
        "slots",
        [
            {"ice": "pitot", "switch": "on"},
            {"ice": "pitot", "switch": "auto"},
            {"ice": "prop", "switch": "off"},
            {"ice": "wing", "switch": "off"},
            {"ice": "engine", "switch": "off"},
        ],
    )
    def test_only_pitot_heat_off_is_held(self, slots):
        v = evaluate_safety("toggle_anti_ice", slots, SafetyContext(agl_ft=3000))
        assert not v.requires_confirmation

    def test_prop_heat_off_does_not_confirm_a_held_pitot_heat_off(self):
        assert command_token("toggle_anti_ice", self.PITOT_OFF) != command_token(
            "toggle_anti_ice", {"ice": "prop", "switch": "off"}
        )


class TestParkingBrakeInterlock:
    def test_parking_brake_at_speed_requires_confirmation(self):
        v = evaluate_safety(
            "toggle_parking_brake", {"state": "on"}, SafetyContext(ground_speed_kt=30)
        )
        assert v.requires_confirmation and v.rule == "parking_brake_at_speed"

    def test_parking_brake_when_stopped_is_fine(self):
        v = evaluate_safety(
            "toggle_parking_brake", {"state": "on"}, SafetyContext(ground_speed_kt=0)
        )
        assert not v.requires_confirmation


class TestNonRiskyCommands:
    def test_heading_change_never_gated(self):
        v = evaluate_safety("set_autopilot_heading", {"degrees": "270"}, SafetyContext())
        assert not v.requires_confirmation

    def test_altitude_change_never_gated(self):
        v = evaluate_safety(
            "set_autopilot_altitude", {"altitude": "12000"}, SafetyContext(agl_ft=100)
        )
        assert not v.requires_confirmation


class TestConfirmationGate:
    def test_repeat_same_command_confirms(self):
        gate = ConfirmationGate()
        token = command_token("toggle_landing_gear", {"state": "up"})
        assert gate.confirm(token) is False  # first time: nothing pending
        gate.arm(token)
        assert gate.confirm(token) is True  # repeat confirms

    def test_confirm_consumes_pending(self):
        gate = ConfirmationGate()
        token = command_token("toggle_engine", {"state": "off"})
        gate.arm(token)
        assert gate.confirm(token) is True
        assert gate.confirm(token) is False  # already consumed

    def test_different_command_does_not_confirm(self):
        gate = ConfirmationGate()
        gate.arm(command_token("toggle_landing_gear", {"state": "up"}))
        other = command_token("toggle_parking_brake", {"state": "on"})
        assert gate.confirm(other) is False

    def test_expires_after_window(self):
        now = {"t": 100.0}
        gate = ConfirmationGate(window_sec=8.0, clock=lambda: now["t"])
        token = command_token("toggle_landing_gear", {"state": "up"})
        gate.arm(token)
        now["t"] = 109.0  # 9s later, past the 8s window
        assert gate.confirm(token) is False
        assert gate.pending_token() is None

    def test_within_window_confirms(self):
        now = {"t": 100.0}
        gate = ConfirmationGate(window_sec=8.0, clock=lambda: now["t"])
        token = command_token("toggle_landing_gear", {"state": "up"})
        gate.arm(token)
        now["t"] = 105.0  # 5s later, inside the window
        assert gate.confirm(token) is True

    def test_command_token_keys_on_intent_instance_and_state(self):
        off_1 = command_token("toggle_engine", {"state": "off", "instance": "1"})
        assert off_1 == "toggle_engine:1:off"
        assert off_1 != command_token("toggle_engine", {"state": "on", "instance": "1"})
        # "engine two off" must not confirm a pending "engine one off"
        assert off_1 != command_token("toggle_engine", {"state": "off", "instance": "2"})
        assert command_token("set_flaps", {"position": "up"}) == "set_flaps::up"

    def test_a_held_switch_is_confirmed_only_by_the_same_switch(self):
        # The switches' action and target are in their own slots.
        gen_1 = command_token(
            "toggle_electrical", {"power": "generator", "instance": "1", "switch": "off"}
        )
        assert gen_1 == "toggle_electrical:1::power=generator:switch=off"
        assert gen_1 != command_token(
            "toggle_electrical", {"power": "avionics", "instance": "1", "switch": "off"}
        )
        assert command_token("set_speed_brake", {"speedbrake": "half"}) != command_token(
            "set_speed_brake", {"speedbrake": "extend"}
        )
        assert command_token("toggle_fuel_pump", {"switch": "off"}) != command_token(
            "toggle_fuel_pump", {"switch": "on"}
        )

    def test_a_stray_slot_does_not_change_the_token(self):
        # "v confirm engine one off" came with a "mode" tag the engine lacks.
        assert (
            command_token("toggle_engine", {"state": "off", "instance": "1", "mode": "v"})
            == "toggle_engine:1:off"
        )


# ---------------------------------------------------------------------------
# Tier 1: flaps above the flap speed
# ---------------------------------------------------------------------------
from vimaan_nlu.safety import SafetyContext as _Ctx  # noqa: E402
from vimaan_nlu.safety import command_token as _token  # noqa: E402
from vimaan_nlu.safety import evaluate_safety as _evaluate  # noqa: E402


def test_flap_setting_above_vfe_needs_confirmation():
    fast = _Ctx(ias_kt=150.0, vfe_kt=122.0)
    slow = _Ctx(ias_kt=100.0, vfe_kt=122.0)
    assert _evaluate("set_flaps", {"flap_setting": "30"}, fast).rule == "flaps_above_vfe"
    assert _evaluate("set_flaps", {"position": "full"}, fast).requires_confirmation
    assert not _evaluate("set_flaps", {"flap_setting": "30"}, slow).requires_confirmation
    # Up, and one notch down, are as before.
    assert not _evaluate("set_flaps", {"position": "up"}, fast).requires_confirmation
    assert not _evaluate("set_flaps", {"position": "down"}, fast).requires_confirmation


def test_unknown_flap_speed_fails_safe():
    verdict = _evaluate("set_flaps", {"flap_setting": "15"}, _Ctx(ias_kt=90.0, vfe_kt=None))
    assert verdict.requires_confirmation and verdict.fail_safe


def test_a_different_flap_setting_does_not_confirm_a_held_one():
    assert _token("set_flaps", {"flap_setting": "15"}) != _token(
        "set_flaps", {"flap_setting": "30"}
    )


@pytest.mark.parametrize(
    ("ctx", "held"),
    [
        (SafetyContext(on_ground=True, agl_ft=3.0), True),
        (SafetyContext(on_ground=False, agl_ft=30.0, vs_fpm=900.0), True),  # just off
        # Climbing out: the flights held gear up at 447-779 ft, five times.
        (SafetyContext(on_ground=False, agl_ft=447.0, vs_fpm=1200.0), False),
        (SafetyContext(on_ground=False, agl_ft=779.0, vs_fpm=350.0), False),
        (SafetyContext(on_ground=False, agl_ft=600.0, vs_fpm=0.0), True),  # level, low
        (SafetyContext(on_ground=False, agl_ft=600.0, vs_fpm=-700.0), True),  # descending
        (SafetyContext(on_ground=False, agl_ft=600.0), True),  # climb rate unknown
        (SafetyContext(on_ground=False, agl_ft=1500.0, vs_fpm=-500.0), False),
        (SafetyContext(), True),  # nothing known: fail safe
    ],
)
def test_gear_up_holds_low_unless_climbing(ctx, held):
    verdict = evaluate_safety("toggle_landing_gear", {"state": "up"}, ctx)
    assert verdict.requires_confirmation is held


# ---- Phase F.4: the rules read the phase of flight -------------------------
@pytest.mark.parametrize(
    ("phase", "vs_fpm", "held"),
    [
        ("approach", -700.0, True),  # a misheard "gear down" on the approach
        ("approach", 0.0, True),  # level on the approach, above 1,000 ft too
        ("landing", -500.0, True),
        ("go_around", 1500.0, False),  # climbing away: one press
        ("climb", 1500.0, False),
        ("cruise", 0.0, False),
        (None, 0.0, False),  # no phase yet: as before (above 1,000 ft, fine)
        ("unknown", 0.0, False),
    ],
)
def test_gear_up_on_an_approach_asks_to_confirm_at_any_height(phase, vs_fpm, held):
    ctx = _Ctx(agl_ft=1800.0, on_ground=False, ias_kt=140.0, vs_fpm=vs_fpm, phase=phase)
    verdict = _evaluate("toggle_landing_gear", {"state": "up"}, ctx)
    assert verdict.requires_confirmation is held
    assert not verdict.fail_safe
    if held:
        assert verdict.message == "Confirm gear up, on approach"
    # Gear down is never held.
    assert not _evaluate("toggle_landing_gear", {"state": "down"}, ctx).requires_confirmation


@pytest.mark.parametrize(
    ("phase", "on_ground", "held"),
    [
        ("takeoff_roll", True, True),  # they'd stop the takeoff
        ("rollout", True, False),  # after landing: what they're for
        ("taxi", True, False),
        ("parked", True, False),
        (None, True, False),  # no phase yet: as before
    ],
)
def test_speed_brake_on_the_takeoff_roll_asks_to_confirm(phase, on_ground, held):
    ctx = _Ctx(agl_ft=0.0, on_ground=on_ground, ias_kt=90.0, phase=phase)
    verdict = _evaluate("set_speed_brake", {"speedbrake": "extend"}, ctx)
    assert verdict.requires_confirmation is held
    if held:
        assert verdict.rule == "speed_brake_low"
        assert verdict.message == "Confirm speed brake, takeoff roll"
    # Armed for landing is never held.
    assert not _evaluate("set_speed_brake", {"speedbrake": "armed"}, ctx).requires_confirmation


@pytest.mark.parametrize(
    ("ias_kt", "on_ground", "stall_clean_kt", "held", "fail_safe"),
    [
        (120.0, False, 139.0, True, False),  # the 737 after takeoff, below Vs
        (145.0, False, 139.0, True, False),  # within 10 kt of it
        (149.0, False, 139.0, False, False),  # 10 kt above: fine
        (220.0, False, 139.0, False, False),
        (100.0, True, 139.0, False, False),  # on the ground: fine
        (None, False, 139.0, True, True),  # no airspeed in the air: fail safe
        (90.0, False, None, False, False),  # no stall speed known: not judged
    ],
)
def test_flaps_up_when_slow_asks_to_confirm(ias_kt, on_ground, stall_clean_kt, held, fail_safe):
    ctx = _Ctx(ias_kt=ias_kt, on_ground=on_ground, agl_ft=900.0, stall_clean_kt=stall_clean_kt)
    verdict = _evaluate("set_flaps", {"position": "up"}, ctx)
    assert verdict.requires_confirmation is held
    assert verdict.fail_safe is fail_safe
    if held:
        assert verdict.rule == "flaps_up_slow"
        assert verdict.message == "Confirm flaps up, slow"
    # Other flap commands don't read the stall speed.
    assert not _evaluate("set_flaps", {"position": "down"}, ctx).requires_confirmation
