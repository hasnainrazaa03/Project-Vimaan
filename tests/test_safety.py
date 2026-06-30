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
        v = evaluate_safety("toggle_engine_1", {"state": "off"}, SafetyContext(agl_ft=3000))
        assert v.requires_confirmation and v.rule == "engine_shutdown_in_flight"

    def test_engine_off_on_ground_is_fine(self):
        ctx = SafetyContext(on_ground=True, agl_ft=0, ias_kt=0)
        v = evaluate_safety("toggle_engine_2", {"state": "off"}, ctx)
        assert not v.requires_confirmation

    def test_engine_off_unknown_state_fails_safe(self):
        v = evaluate_safety("toggle_engine_1", {"state": "off"}, SafetyContext())
        assert v.requires_confirmation and v.fail_safe

    def test_engine_start_not_gated(self):
        v = evaluate_safety("toggle_engine_1", {"state": "on"}, SafetyContext(agl_ft=3000))
        assert not v.requires_confirmation

    def test_airborne_detected_by_speed_when_no_agl(self):
        v = evaluate_safety("toggle_engine_1", {"state": "off"}, SafetyContext(ias_kt=120))
        assert v.requires_confirmation and not v.fail_safe


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
        token = command_token("toggle_engine_1", {"state": "off"})
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

    def test_command_token_keys_on_intent_and_state(self):
        assert command_token("toggle_engine_1", {"state": "off"}) == "toggle_engine_1:off"
        assert command_token("toggle_engine_1", {"state": "on"}) != command_token(
            "toggle_engine_1", {"state": "off"}
        )
