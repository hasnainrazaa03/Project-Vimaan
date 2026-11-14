"""Vimaan AI Co-Pilot — X-Plane plugin entry point.

Hold-to-talk speech command for X-Plane via a joint intent/slot DistilBERT
model. Hold the ``Z`` key (or a yoke/joystick button bound to the
``vimaan/push_to_talk`` command), speak, release.

Process model
-------------
The voice pipeline — microphone, speech-to-text, the model — runs in a
separate worker process (``ML/vimaan_nlu/voice_worker.py``) started with
XPPython3's ``xp.pythonExecutable``. Inside X-Plane, Python background threads
only run when the simulator's main thread releases the GIL, so in-process
capture/STT/NLU cost visible frame hitches (measured in-sim: ~118 ms per
command, plus several ms per frame while transcribing). The plugin now only
sends "start"/"stop" on key down/up and reads replies from a non-blocking pipe
in the flight loop — microseconds per frame. torch never loads into X-Plane.

All XPPython3 (`xp.*`) calls run on the simulator's main thread: dispatch,
safety checks, dataref writes, verification and engine cranking all happen in
the flight loop.

State management
----------------
Toggle-style intents consult the relevant dataref before acting so that
"off" never accidentally turns the system on when it was already off, and
vice-versa. Every dataref/command name here is verified against X-Plane
12.4's DataRefs.txt / Commands.txt (``scripts/check_xplane_refs.py``).

Diagnostics
-----------
Every push-to-talk gets a command id that tags each stage (capture, STT,
NLU, decision, sim writes, verification) in ``~/Vimaan_Logs/``; the worker's
own output goes to ``vimaan_<ts>.worker.log``. See
``ML/vimaan_nlu/telemetry.py`` and ``scripts/vimaan_log_report.py``.
"""

import time

_IMPORT_T0 = time.perf_counter()

import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from typing import Any  # noqa: E402

from XPPython3 import xp  # noqa: E402

# Locate the ML package. In dev it is <repo>/ML (i.e. <plugin_dir>/../ML); when
# installed by scripts/install_plugin.py it sits beside the plugin as
# PythonPlugins/ML. Use the first layout that exists so both work.
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_CANDIDATES = (
    os.path.join(_PLUGIN_DIR, "..", "ML"),  # <repo>/plugin/ -> <repo>/ML (dev)
    os.path.join(_PLUGIN_DIR, "ML"),  # PythonPlugins/PI_*.py + PythonPlugins/ML (installed)
)
_ML_PATH = next(
    (os.path.abspath(p) for p in _ML_CANDIDATES if os.path.isdir(os.path.abspath(p))),
    os.path.abspath(_ML_CANDIDATES[0]),
)
if _ML_PATH not in sys.path:
    sys.path.insert(0, _ML_PATH)

# Only light, stdlib-backed modules load into X-Plane; the heavy ones live in
# the worker process.
from vimaan_nlu import probe, sim_bridge  # noqa: E402
from vimaan_nlu.aircraft import (  # noqa: E402
    AircraftInfo,
    acf_flap_limits,
    files_mention,
    flap_limit_kt,
    resolve,
)
from vimaan_nlu.ambiguity import TARGET_SLOTS, bare_number, chosen, two_or_to  # noqa: E402
from vimaan_nlu.commands import (  # noqa: E402
    BY_INTENT,
    COMMANDS,
    required_slots,
    self_test_probes,
)
from vimaan_nlu.disconnect import (  # noqa: E402
    WATCHED_COMMANDS,
    CommandTrace,
    StickTrace,
    diagnose,
    dropped,
)
from vimaan_nlu.overlay import Overlay  # noqa: E402
from vimaan_nlu.overlay_state import OverlayState  # noqa: E402
from vimaan_nlu.pilot_replies import override_prefix, pilot_reply  # noqa: E402
from vimaan_nlu.readback import spell_digits  # noqa: E402
from vimaan_nlu.safety import (  # noqa: E402
    ConfirmationGate,
    SafetyContext,
    command_token,
    evaluate_safety,
)
from vimaan_nlu.telemetry import Telemetry  # noqa: E402
from vimaan_nlu.validators import (  # noqa: E402
    bare_command,
    clipped_number,
    has_command_word,
    heading_too_short,
    validate_slot,
)
from vimaan_nlu.worker_client import WorkerClient  # noqa: E402

_IMPORT_MS = round((time.perf_counter() - _IMPORT_T0) * 1000, 1)

QUEUE_POLL_INTERVAL_SEC = 0.2
# A command with no reply from the worker after this long is abandoned.
WORKER_REPLY_TIMEOUT_SEC = 20.0
# The worker is restarted after a crash at most this many times per session.
WORKER_MAX_RESTARTS = 3

# Engine start: hold the starter until the engine reports running, up to this.
STARTER_MAX_CRANK_SEC = 20.0
# Turbine starts (jet, turboprop) spool for longer, and fuel goes in only once
# the starter has the core turning: N2 (jet) / N1, the gas generator
# (turboprop), in percent. The starter is held until the engine runs and the
# core is past the cutout.
TURBINE_MAX_CRANK_SEC = 60.0
FUEL_ON_AT_PCT = {"jet": 20.0, "turboprop": 13.0}
STARTER_CUTOUT_PCT = 50.0
CORE_SPEED = {
    "jet": "sim/cockpit2/engine/indicators/N2_percent",
    "turboprop": "sim/cockpit2/engine/indicators/N1_percent",
}
DREF_FUEL_LEVER = "sim/cockpit2/engine/actuators/mixture_ratio"  # 0 cutoff, 1 open
# What the pilot must say with a number to make it a command, per "to or two?" target.
COMMAND_NOUN = {
    "set_autopilot_altitude": "altitude",
    "set_autopilot_heading": "heading",
    "set_flight_level": "flight level",
    "set_autopilot_speed": "speed",
    "set_vertical_speed": "vertical speed",
    "set_flaps": "flaps",
    "set_course": "course",
}
# Delay before reading back a dataref a command or write should have changed.
VERIFY_DELAY_SEC = 1.0
# After a target or radio is set, record the aircraft this many seconds later:
# a changed target is not the same as an aircraft that followed it.
FOLLOW_UP_SEC = (0.0, 10.0, 30.0)
# How often to look for autopilot changes nobody asked Vimaan for.
STATE_WATCH_SEC = 1.0
# The automated-test bridge (vimaan_nlu.sim_bridge): how often it looks for
# its switch file and requests.
BRIDGE_POLL_SEC = 0.25
METRES_PER_FOOT = 0.3048
MPS_PER_KNOT = 0.514444
# Autopilot-off diagnostics (vimaan_nlu.disconnect). The pilot's stick input,
# not the autopilot's: X-Plane keeps the servo output separately.
STICK_INPUT = {
    "pitch": "sim/joystick/yoke_pitch_ratio",
    "roll": "sim/joystick/yoke_roll_ratio",
    "yaw": "sim/joystick/yoke_heading_ratio",
}
SERVO_OUTPUT = {
    "pitch": "sim/joystick/servo_pitch_ratio",
    "roll": "sim/joystick/servo_roll_ratio",
    "yaw": "sim/joystick/servo_heading_ratio",
}
DREF_HAS_JOYSTICK = "sim/joystick/has_joystick"
DREF_NULLZONE = "sim/joystick/joystick_pitch_nullzone"  # one serves all axes
DREF_AP_POWER = "sim/cockpit2/autopilot/autopilot_has_power"
DREF_AP_FAIL = "sim/cockpit2/autopilot/otto_fail_warn"
DREF_AP_DISCONNECT_WARNING = "sim/cockpit2/autopilot/autopilot_disconnect"
DREF_MACH = "sim/flightmodel/misc/machno"
# "Autopilot off" from Vimaan this recently is Vimaan's doing.
VIMAAN_OFF_SEC = 5.0
# A new flight or aircraft resets the autopilot: no callout for this long.
RELOAD_QUIET_SEC = 10.0
# After an aircraft loads, look again for its overrides (its scripts'
# datarefs may come a moment later).
OVERRIDE_RECHECK_SEC = (5.0, 15.0)
# After arming an autothrottle switch, engage this much later (the 737's
# scripts take the switch on their next frames).
ATHR_ARM_SETTLE_SEC = 0.5

# What the aircraft is doing, logged as aircraft_state.
STATE_FLOATS = {
    "heading_actual": "sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot",
    "altitude_actual": "sim/cockpit2/gauges/indicators/altitude_ft_pilot",
    "vs_actual": "sim/cockpit2/gauges/indicators/vvi_fpm_pilot",
    "ias_actual": "sim/cockpit2/gauges/indicators/airspeed_kts_pilot",
    "heading_target": "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot",
    "altitude_target": "sim/cockpit2/autopilot/altitude_dial_ft",
}
STATE_RADIOS = {
    "com1_active": "sim/cockpit2/radios/actuators/com1_frequency_hz_833",
    "com1_standby": "sim/cockpit2/radios/actuators/com1_standby_frequency_hz_833",
    "com2_active": "sim/cockpit2/radios/actuators/com2_frequency_hz_833",
    "com2_standby": "sim/cockpit2/radios/actuators/com2_standby_frequency_hz_833",
}
AUTOPILOT_STATE = {
    "servos_on": "sim/cockpit2/autopilot/servos_on",
    "autopilot_on": "sim/cockpit2/autopilot/autopilot_on",
    "flight_director": "sim/cockpit2/autopilot/flight_director_mode",
    "lateral_mode": "sim/cockpit2/autopilot/heading_mode",
    "vertical_mode": "sim/cockpit2/autopilot/altitude_mode",
    "heading_status": "sim/cockpit2/autopilot/heading_status",
    "nav_status": "sim/cockpit2/autopilot/nav_status",
    "approach_status": "sim/cockpit2/autopilot/approach_status",
    "altitude_hold_status": "sim/cockpit2/autopilot/altitude_hold_status",
    "altitude_armed": "sim/cockpit2/autopilot/altitude_hold_armed",
    "vs_status": "sim/cockpit2/autopilot/vvi_status",
    "speed_status": "sim/cockpit2/autopilot/speed_status",
}
DREF_AIRCRAFT_ICAO = "sim/aircraft/view/acf_ICAO"
# What X-Plane reports about the loaded aircraft (vimaan_nlu.aircraft, layer 2).
AIRCRAFT_STRINGS = {
    "author": "sim/aircraft/view/acf_author",
    "description": "sim/aircraft/view/acf_descrip",
}
DREF_ENGINE_TYPE = "sim/aircraft/prop/acf_en_type"
DREF_FLAP_DETENTS = "sim/aircraft/controls/acf_flap_detents"
DREF_FLAP_DEGREES = "sim/aircraft/controls/acf_flap_dn"
DREF_AUTOPILOT_TYPE = "sim/aircraft/autopilot/preconfigured_ap_type"
DREF_SINGLE_AXIS_AP = "sim/aircraft/autopilot/single_axis_autopilot"
V_SPEEDS = {
    "vso_kt": "sim/aircraft/view/acf_Vso",
    "vfe_kt": "sim/aircraft/view/acf_Vfe",
    "vno_kt": "sim/aircraft/view/acf_Vno",
    "vne_kt": "sim/aircraft/view/acf_Vne",
    "mmo": "sim/aircraft/view/acf_Mmo",
}

# Commands (phrasing, values, bindings, handler, safety, self-test) are
# declared once in vimaan_nlu/commands.py; these are views of that table.
# Binary systems (autopilot, flight director): state dataref, dedicated on
# command, toggle. X-Plane has no "servos off" / "fdir off" commands, so "off"
# uses the toggle, but only after reading the state, so it can never turn a
# system on.
# intent -> instance ("1"/"2") -> binding.
TOGGLE_BINDINGS: dict[str, dict[str, dict[str, Any]]] = {
    c.intent: dict(c.params["instances"]) for c in COMMANDS if c.handler == "_toggle_binary"
}
# Engines: starter held until the engine reports running. instance -> binding.
ENGINE_BINDINGS: dict[str, dict[str, Any]] = dict(BY_INTENT["toggle_engine"].params["instances"])

DREF_HEADING = "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"
DREF_ALTITUDE = "sim/cockpit2/autopilot/altitude_dial_ft"
# kHz, supports 8.33 kHz spacing: 118.750 MHz -> 118750.
DREF_COM_KHZ: dict[str, str] = dict(BY_INTENT["set_com_frequency"].params["datarefs"])
# Tier 2 radios. NAV frequencies are in 10 kHz units: 110.50 MHz -> 11050.
DREF_COM_STANDBY_KHZ: dict[str, str] = dict(BY_INTENT["set_com_standby"].params["datarefs"])
CMD_COM_FLIP: dict[str, str] = dict(BY_INTENT["swap_com_frequency"].params["commands"])
DREF_NAV_10KHZ: dict[str, str] = dict(BY_INTENT["set_nav_frequency"].params["datarefs"])
DREF_NAV_STANDBY_10KHZ: dict[str, str] = dict(BY_INTENT["set_nav_standby"].params["datarefs"])
CMD_NAV_FLIP: dict[str, str] = dict(BY_INTENT["swap_nav_frequency"].params["commands"])
DREF_OBS: dict[str, str] = dict(BY_INTENT["set_course"].params["datarefs"])
DREF_XPDR_CODE: str = BY_INTENT["set_transponder_code"].params["dataref"]
CMD_XPDR: dict[str, str] = dict(BY_INTENT["set_transponder_mode"].params["commands"])
DREF_XPDR_MODE: str = BY_INTENT["set_transponder_mode"].params["mode_dataref"]
DREF_XPDR_IDENT = "sim/cockpit2/radios/indicators/transponder_id"
# X-Plane's transponder_mode: off 0, standby 1, on (mode A) 2, alt (mode C) 3,
# test 4, ground (mode S) 5, TA only 6, TA/RA 7. What each spoken mode sets,
# and the modes that already are it. "On" means replying: it never turns
# altitude reporting (3, 6, 7) down to mode A.
XPDR_SETS = {"off": 0, "standby": 1, "on": 2, "alt": 3}
XPDR_ALREADY = {"off": {0}, "standby": {1}, "on": {2, 3, 5, 6, 7}, "alt": {3, 6, 7}}
XPDR_SPOKEN = {"off": "off", "standby": "standby", "on": "on", "alt": "altitude"}
XPDR_ALREADY_SPOKEN = {
    "off": "Transponder already off",
    "standby": "Transponder already in standby",
    "on": "Transponder already on",
    "alt": "Transponder already reporting altitude",
}
# Transponder selectors turned by clicks (737, A330).
KNOB_STEP_MAX_PRESSES = 12
KNOB_STEP_MAX_SEC = 10.0
KNOB_STUCK_FRAMES = 30  # a click that moves nothing for this many frames
DREF_GEAR_HANDLE = "sim/cockpit2/controls/gear_handle_down"
DREF_GEAR_RETRACTABLE = "sim/aircraft/gear/acf_gear_retract"
DREF_FLAP_HANDLE = "sim/cockpit2/controls/flap_handle_request_ratio"
# Tier 1 targets.
DREF_SPEED_DIAL = "sim/cockpit2/autopilot/airspeed_dial_kts_mach"
DREF_SPEED_IS_MACH = "sim/cockpit2/autopilot/airspeed_is_mach"
DREF_VVI_DIAL = "sim/cockpit2/autopilot/vvi_dial_fpm"
DREF_VVI_STATUS = "sim/cockpit2/autopilot/vvi_status"
DREF_ALTITUDE_ACTUAL = "sim/cockpit2/gauges/indicators/altitude_ft_pilot"
DREF_AUTOTHROTTLE = "sim/cockpit2/autopilot/autothrottle_enabled"
DREF_BARO = (
    "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_pilot",
    "sim/cockpit2/gauges/actuators/barometer_setting_in_hg_copilot",
)
HPA_TO_INHG = 0.0295300
DREF_ALT_ARMED = "sim/cockpit2/autopilot/altitude_hold_armed"
DREF_SERVOS_ON = "sim/cockpit2/autopilot/servos_on"
DREF_VERTICAL_MODE = "sim/cockpit2/autopilot/altitude_mode"
DREF_ALT_HOLD_STATUS = "sim/cockpit2/autopilot/altitude_hold_status"
VERTICAL_MOVING_MODES = (4, 5)  # V/S, level change: modes that climb or descend
# Arming altitude capture waits for the climb/descent mode to settle: the 737
# arms it by itself when V/S engages, and the generic arm command toggles.
ARM_CAPTURE_DELAY_SEC = 1.2
# Vimaan's own capture (option b): press altitude hold this far ahead of the
# target, scaled to the climb rate, never closer than the minimum.
CAPTURE_LEAD_SEC = 6.0
CAPTURE_MIN_BAND_FT = 60.0
CAPTURE_MAX_SEC = 20 * 60
# Altimeter by knob steps (aircraft that put written values back).
BARO_TOLERANCE_INHG = 0.005
BARO_STEP_MAX_PRESSES = 400
BARO_STEP_MAX_SEC = 30.0
# Autopilot mode -> (profile role, status dataref; 0 off, 1 armed, 2 active).
AUTOPILOT_MODES = {
    "heading": ("mode_heading", "sim/cockpit2/autopilot/heading_status"),
    "nav": ("mode_nav", "sim/cockpit2/autopilot/nav_status"),
    "approach": ("mode_approach", "sim/cockpit2/autopilot/approach_status"),
    "back course": ("mode_back_course", "sim/cockpit2/autopilot/backcourse_status"),
    "altitude hold": ("mode_altitude_hold", "sim/cockpit2/autopilot/altitude_hold_status"),
    "vertical speed": ("mode_vertical_speed", "sim/cockpit2/autopilot/vvi_status"),
    "level change": ("mode_level_change", "sim/cockpit2/autopilot/speed_status"),
    "v nav": ("mode_vnav", "sim/cockpit2/autopilot/vnav_status"),
    # LNAV on X-Plane's generic autopilot is GPS steering (to verify per aircraft).
    "l nav": ("mode_gpss", "sim/cockpit2/autopilot/gpss_status"),
}
MODE_SPOKEN = {"nav": "Nav", "v nav": "V NAV", "l nav": "L NAV"}
# Modes where armed (status 1) isn't on. Altitude hold armed means "capture
# the selected altitude later" (after level change on the 737 and A330);
# "altitude hold" asks to hold now. Nav and approach armed are on: waiting
# to capture is what they do.
ACTIVE_ONLY_MODES = ("altitude hold",)
FMS_MODES = ("v nav", "l nav")
# cockpit2/controls/parking_brake_ratio is deprecated in 12.4 (X-Plane logs
# "has been replaced" on every read); this is its replacement.
DREF_PARKING_BRAKE = "sim/flightmodel/controls/parkbrake"
DREF_VERTICAL_SPEED_FPM = "sim/flightmodel/position/vh_ind_fpm"
DREF_ENGINE_RUNNING = "sim/flightmodel/engine/ENGN_running"
DREF_ENGINE_MIXTURE = "sim/flightmodel/engine/ENGN_mixt"
DREF_IGNITION_KEY = "sim/cockpit2/engine/actuators/ignition_key"
DREF_NUM_ENGINES = "sim/aircraft/engine/acf_num_engines"
DREF_BATTERY = "sim/cockpit2/electrical/battery_on"
DREF_PAUSED = "sim/time/paused"

# Slots each actionable intent needs before it can be dispatched (self-test).
REQUIRED_SLOTS: dict[str, tuple[str, ...]] = required_slots()


def _round(value, digits):
    return None if value is None else round(value, digits)


def mhz_text(units, per_mhz):
    """A radio dataref's units as MHz text: 121500 kHz -> "121.5",
    11050 (10 kHz) -> "110.5", 118025 -> "118.025"."""
    digits = 3 if per_mhz == 1000 else 2
    text = f"{units / per_mhz:.{digits}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


class PythonInterface:
    # Intents predicted below this confidence are not dispatched; the copilot
    # asks the pilot to repeat instead of acting on a low-certainty guess.
    CONFIDENCE_FLOOR = 0.55

    # Load the model as dynamic-int8 (Phase 4B). ~halves model RAM/disk with no
    # measured intent-F1 change. Falls back to fp32 if the runtime has no
    # quantized backend. Set False to force fp32. Ignored when BACKEND="onnx".
    USE_QUANTIZED_MODEL = True

    # Inference backend (Phase 4C): "torch" (default) or "onnx". Override at
    # runtime with the VIMAAN_BACKEND env var. The ONNX backend requires
    # onnxruntime and a prior `python3 ML/export_onnx.py` to produce
    # <model>/onnx/model.onnx.
    BACKEND = os.environ.get("VIMAAN_BACKEND", "torch").lower()

    # Speech-to-text backend: "vosk" (offline, streaming, restricted to the
    # command words; default), "google" (online), "whisper". Override with
    # VIMAAN_STT_BACKEND. If Vosk can't load, the worker falls back to Google
    # and logs stt_fallback. See docs/OFFLINE_STT.md.
    STT_BACKEND = os.environ.get("VIMAAN_STT_BACKEND", "vosk").lower()
    # Each command's audio is saved under ~/Vimaan_Logs/audio/ so real flights
    # become a recogniser test set. VIMAAN_SAVE_AUDIO=0 turns it off.
    SAVE_AUDIO = os.environ.get("VIMAAN_SAVE_AUDIO", "1") != "0"
    # In-sim overlay (live transcript, what was understood, outcome):
    # "auto" (show on activity, then fade), "pinned" or "off". Cycle in-sim
    # with the vimaan/toggle_overlay command.
    OVERLAY_MODE = os.environ.get("VIMAAN_OVERLAY", "auto").lower()
    WHISPER_MODEL = os.environ.get("VIMAAN_WHISPER_MODEL", "base.en")

    # Intents the model can emit that map to no sim command. "None" is a
    # deliberate out-of-scope reject sink (it improves precision on real
    # commands); the conversational intents get a short spoken acknowledgement
    # instead of the misleading "Command not found". A None value => stay
    # silent.
    NON_ACTIONABLE_INTENTS = {c.intent: c.reply for c in COMMANDS if not c.actionable}

    # Seconds a risky command stays "armed" waiting for the pilot to repeat it
    # as confirmation (Phase 6B safety interlocks).
    CONFIRM_WINDOW_SEC = 12.0
    # After "Altitude 1000 or 21000 feet?", a bare-number reply within this
    # long is taken as the answer, if it is one of the two values offered.
    QUESTION_WINDOW_SEC = 12.0

    # Canned (utterance, expected intent) pairs for the Ctrl-Z self-test
    # (Phase 6C): declared per command in the table, run through text -> NLU
    # (STT bypassed, nothing actuated). A probe passes only if the intent is
    # the expected one AND every slot its handler needs was extracted.
    SELF_TEST_PROBES = self_test_probes()

    # What the pilot hears when a command ends without reaching the sim.
    OUTCOME_SPEECH = {
        "silent_mic": "Microphone appears silent",
        "not_understood": "I could not understand you",
        "stt_service_error": "Recognition service failed",
        "mic_error": "Microphone error",
        "worker_error": "An error occurred",
        "timeout": "Voice engine timed out",
        "too_short": None,
    }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self):
        self.Name = "Vimaan AI CoPilot"
        self.Sig = "plugin.vimaan.aicopilot.bymhr"
        self.Desc = "Voice command interface with joint intent + slot NLU for X-Plane"

        self.tel = Telemetry(mirror=xp.log)
        self.tel.event(
            "session_start",
            log_file=str(self.tel.log_path),
            ml_path=_ML_PATH,
            import_ms=_IMPORT_MS,
            backend=self.BACKEND,
            stt_backend=self.STT_BACKEND,
            **self.tel.environment(),
        )
        self.ready = False
        self.worker: Any = None
        self._worker_started_at: float | None = None
        self._worker_restarts = 0
        # The push-to-talk command in flight: recording, then awaiting a reply.
        self._active: dict[str, Any] | None = None
        self._self_test_pending = False
        self._self_test_started = 0.0
        # Intents the loaded model knows (from the worker's ready message);
        # None until then, or from a worker that doesn't say.
        self._model_intents: set[str] | None = None
        self._flight_loop_id = None
        self.hotkeyPress = None
        self.hotkeyRelease = None
        self.hotkeySelfTest = None
        self._commands: list[tuple[Any, Any]] = []
        self._current_cmd: str | None = None
        # Deferred main-thread work: dataref read-backs and starter cranks.
        self._verifications: list[dict[str, Any]] = []
        self._cranks: dict[str, dict[str, Any]] = {}
        self._follow_ups: list[dict[str, Any]] = []
        self._autopilot_state: dict[str, Any] | None = None
        self._aircraft: tuple[str, str] | None = None  # (ICAO, .acf path) last loaded
        # How Vimaan drives the loaded aircraft; standard controls until one loads.
        self.profile = resolve(AircraftInfo(), lambda _name: False)
        self._next_state_watch = 0.0
        self._last_action: dict[str, Any] | None = None
        self._question: dict[str, Any] | None = None  # the open "to or two?" question
        self._held: dict[str, Any] | None = None  # the command a safety rule is holding
        self._probe: dict[str, Any] | None = None  # a running aircraft probe
        self._deferred: list[dict[str, Any]] = []  # main-thread actions due later
        self._baro_step: dict[str, Any] | None = None  # altimeter set by knob steps
        self._knob_step: dict[str, Any] | None = None  # selectors turned by clicks
        self._capture: dict[str, Any] | None = None  # Vimaan's own altitude capture
        # Option (b): Vimaan captures altitude where the aircraft can't. Off
        # unless VIMAAN_CAPTURE=1, or switched on with its key.
        self._capture_enabled = os.environ.get("VIMAAN_CAPTURE") == "1"
        # Whether the loaded aircraft's own files use an autothrottle:
        # (acf path, True/False/None), scanned off the main thread.
        self._athr_scan: tuple[str, bool | None] | None = None
        # The loaded aircraft's flap speed limits from its .acf, read with
        # the scan: (acf path, {"first": kt, "full": kt}).
        self._flap_limits: tuple[str, dict[str, float]] | None = None
        # The automated-test bridge (Phase A): off unless its switch file exists.
        self._bridge_on = False
        self._bridge_next = 0.0
        self._say_results = sim_bridge.SayResults()
        # Autopilot-off diagnostics: stick history, autopilot-off commands
        # seen (whoever fired them), and the command Vimaan is firing now.
        self._stick = StickTrace()
        self._off_commands = CommandTrace()
        self._hooks: list[Any] = []  # command refs with our watch handler
        self._hooked: set[str] = set()
        self._firing: str | None = None
        self._quiet_until = 0.0  # no disconnect callouts until then (reload)

        # Safety interlocks (Phase 6B): risky commands must be repeated to confirm.
        self._confirm_gate = ConfirmationGate(window_sec=self.CONFIRM_WINDOW_SEC)
        self.intent_to_command = self._setup_intent_handlers()
        # The overlay follows the telemetry stream: everything it shows is
        # also in events.jsonl (except live partial transcripts).
        self.overlay_state = OverlayState(mode=self.OVERLAY_MODE)
        self.tel.subscribe(self.overlay_state.on_event)
        self.tel.subscribe(self._say_results.on_event)
        self.overlay: Overlay | None = None
        self._start_worker()

    # ------------------------------------------------------------------
    # Voice worker process
    # ------------------------------------------------------------------
    @staticmethod
    def _python_executable():
        """The interpreter to run the worker with. ``sys.executable`` is
        X-Plane itself here; XPPython3 exposes the right one. (Its standalone
        ``bin/python3.12`` is unsigned for arm64 and killed by macOS.)"""
        exe = getattr(xp, "pythonExecutable", None)
        if exe and os.path.exists(exe):
            return exe
        app = os.path.join(
            sys.exec_prefix, "Resources", "Python.app", "Contents", "MacOS", "Python"
        )
        return app if os.path.exists(app) else sys.executable

    def _worker_argv(self):
        argv = [
            self._python_executable(),
            "-u",
            "-m",
            "vimaan_nlu.voice_worker",
            "--stt-backend",
            self.STT_BACKEND,
            "--whisper-model",
            self.WHISPER_MODEL,
            "--backend",
            self.BACKEND,
        ]
        if not self.USE_QUANTIZED_MODEL:
            argv.append("--no-quantize")
        if self.SAVE_AUDIO:
            argv += [
                "--save-audio-dir",
                str(self.tel.log_dir / "audio"),
                "--audio-prefix",
                f"{self.tel.session_id}_",
            ]
        return argv

    def _start_worker(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = _ML_PATH + os.pathsep + env.get("PYTHONPATH", "")
        env["HF_HUB_OFFLINE"] = "1"  # the model is local; never touch the network for it
        env["TOKENIZERS_PARALLELISM"] = "false"
        log_path = self.tel.log_dir / f"vimaan_{self.tel.session_id}.worker.log"
        argv = self._worker_argv()
        self.worker = WorkerClient(argv, log_path, env=env, cwd=_ML_PATH)
        try:
            self.worker.start()
        except Exception as exc:
            self.tel.error("worker_spawn_failed", exc, argv=argv)
            self.worker = None
            return
        self._worker_started_at = time.perf_counter()
        self.tel.event("worker_spawn", pid=self.worker.pid, argv=argv, log=str(log_path))

    def _check_worker(self):
        """Detect a dead worker; restart it a few times, then give up loudly."""
        if self.worker is None or self.worker.alive:
            return
        self.tel.event(
            "worker_exit",
            logging.ERROR,
            returncode=self.worker.returncode,
            log_tail=self.worker.log_tail(25),
        )
        self.ready = False
        if self._active is not None:
            self._end_command("worker_error")
        self.worker.stop()
        if self._worker_restarts < WORKER_MAX_RESTARTS:
            self._worker_restarts += 1
            self._say("Voice engine restarting")
            self._start_worker()
        else:
            self.worker = None
            self._say("Voice engine failed. Check the log.")

    def _setup_intent_handlers(self):
        return {c.intent: getattr(self, c.handler) for c in COMMANDS if c.handler}

    # ------------------------------------------------------------------
    # XPPython3 plugin hooks
    # ------------------------------------------------------------------
    def XPluginStart(self):
        # Hold Z to talk: key down starts recording, key up ends it.
        self.hotkeyPress = xp.registerHotKey(
            xp.VK_Z, xp.DownFlag, "Vimaan Push-to-Talk", self.OnPressCallback
        )
        self.hotkeyRelease = xp.registerHotKey(
            xp.VK_Z, xp.UpFlag, "Vimaan Push-to-Talk release", self.OnReleaseCallback
        )
        # Ctrl-Z: STT-bypass self-test (Phase 6C).
        self.hotkeySelfTest = xp.registerHotKey(
            xp.VK_Z, xp.DownFlag | xp.ControlFlag, "Vimaan Self-Test", self._run_self_test
        )
        # Bindable commands, so push-to-talk can live on a yoke button.
        self._register_command("vimaan/push_to_talk", "Vimaan: push to talk", self._on_ptt_command)
        self._register_command("vimaan/self_test", "Vimaan: run self test", self._on_self_test_cmd)
        self._register_command(
            "vimaan/toggle_overlay", "Vimaan: overlay auto / pinned / off", self._on_overlay_cmd
        )
        # The aircraft probe actuates the autopilot, so it is a key the pilot
        # binds and presses, one step per press; never a voice command.
        self._register_command(
            "vimaan/probe_aircraft",
            "Vimaan: probe this aircraft (one step per press)",
            self._on_probe_cmd,
        )
        self._register_command(
            "vimaan/probe_stop",
            "Vimaan: stop the probe (press twice to start over)",
            self._on_probe_stop_cmd,
        )
        self._register_command(
            "vimaan/probe_skip", "Vimaan: probe, skip to the next step", self._on_probe_skip_cmd
        )
        self._register_command(
            "vimaan/probe_back", "Vimaan: probe, go back one step", self._on_probe_back_cmd
        )
        self._register_command(
            "vimaan/toggle_capture",
            "Vimaan: altitude capture by Vimaan on/off (aircraft that can't capture)",
            self._on_capture_cmd,
        )
        self._hook_off_commands()
        self._start_overlay()

        # Flight loop: worker replies, dispatch, verification, engine cranks.
        self._flight_loop_id = xp.createFlightLoop(self._flight_loop, phase=1)
        xp.scheduleFlightLoop(self._flight_loop_id, QUEUE_POLL_INTERVAL_SEC, 1)

        self.tel.event("plugin_start", worker=self.worker is not None, sim=self._sim_info())
        if self.worker is None:
            self._say("Vimaan could not start its voice engine. Check the log.")
        # "Ready" is announced when the worker has loaded the model.
        return self.Name, self.Sig, self.Desc

    def XPluginStop(self):
        # Never leave a starter engaged.
        for engine in list(self._cranks):
            self._end_crank(engine, "plugin_stop")
        if self.overlay is not None:
            self.overlay.stop()
            self.overlay = None
        for cmd_ref, handler in self._commands:
            xp.unregisterCommandHandler(cmd_ref, handler, 1, None)
        self._commands = []
        for cmd_ref, name in self._hooks:
            xp.unregisterCommandHandler(cmd_ref, self._on_off_command, 1, name)
        self._hooks = []
        self._hooked = set()
        if self.worker is not None:
            self.worker.stop()
        for hotkey in (self.hotkeyPress, self.hotkeyRelease, self.hotkeySelfTest):
            if hotkey:
                xp.unregisterHotKey(hotkey)
        if self._flight_loop_id:
            xp.destroyFlightLoop(self._flight_loop_id)
            self._flight_loop_id = None
        self.tel.event("plugin_stop")
        self.tel.close()

    def XPluginEnable(self):
        self.tel.event("plugin_enable")
        return 1

    def XPluginDisable(self):
        self.tel.event("plugin_disable")

    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        # A new flight or aircraft resets the autopilot: that's not a disconnect.
        if inMessage == xp.MSG_AIRPORT_LOADED or (
            inMessage == xp.MSG_PLANE_LOADED and inParam == 0
        ):
            self._autopilot_state = None
            self._quiet_until = time.perf_counter() + RELOAD_QUIET_SEC

    # ------------------------------------------------------------------
    # Autopilot-off diagnostics (vimaan_nlu.disconnect)
    # ------------------------------------------------------------------
    def _hook_off_commands(self):
        """Watch every autopilot-off command, whoever fires it. Aircraft
        commands exist only once that aircraft loads, so this runs again on
        each aircraft change."""
        for name in WATCHED_COMMANDS:
            if name in self._hooked:
                continue
            cmd_ref = xp.findCommand(name)
            if not cmd_ref:
                continue
            xp.registerCommandHandler(cmd_ref, self._on_off_command, 1, name)
            self._hooks.append((cmd_ref, name))
            self._hooked.add(name)

    def _on_off_command(self, cmd_ref, phase, name):
        if phase == xp.CommandBegin:
            self._off_commands.add(time.perf_counter(), name, by_vimaan=self._firing == name)
        return 1  # never swallow it: X-Plane and the aircraft still act on it

    def _overspeed(self):
        """ "335 kt, limit 330" when above the aircraft's Vne (Vmo on jets) or
        Mmo, else None."""
        info = self.profile.info
        ias = self._read_float(STATE_FLOATS["ias_actual"])
        if ias is not None and info.vne_kt and ias > info.vne_kt:
            return f"{ias:.0f} kt, limit {info.vne_kt:.0f}"
        mach = self._read_float(DREF_MACH)
        if mach is not None and info.mmo and mach > info.mmo:
            return f"Mach {mach:.2f}, limit {info.mmo:.2f}"
        return None

    def _sample_stick(self):
        now = time.perf_counter()
        if self._stick.due(now):
            self._stick.add(now, {axis: self._read_float(n) for axis, n in STICK_INPUT.items()})

    def _joystick_setup(self):
        """Logged per aircraft: a null zone of 0 lets a resting stick fly the
        aircraft, and on airliners take the autopilot off."""
        nullzone = self._read_float(DREF_NULLZONE)
        return {
            "has_joystick": self._read_int(DREF_HAS_JOYSTICK),
            "nullzone": None if nullzone is None else round(nullzone, 3),
            "stick": {a: _round(self._read_float(n), 3) for a, n in STICK_INPUT.items()},
        }

    def _on_autopilot_drop(self, previous, state, now):
        last = self._last_action
        # Only an "off" from Vimaan is its doing ("flight director off" takes
        # the servos with it in X-Plane). A pilot disconnect seconds after
        # "autopilot on" was missed until 2026-11-12.
        vimaan_off = bool(
            last
            and last["intent"] in ("toggle_autopilot", "toggle_flight_director")
            and last.get("state") == "off"
            and now - last["t"] <= VIMAAN_OFF_SEC
        )
        commands = self._off_commands.recent(now)
        peak = self._stick.peak(now)
        overspeed = self._overspeed()
        found = diagnose(
            stick_peak=peak,
            commands=commands,
            has_power=self._read_int(DREF_AP_POWER),
            fail_warn=self._read_int(DREF_AP_FAIL),
            vimaan_turned_off=vimaan_off,
            probe_running=self._probe is not None,
            reloading=now < self._quiet_until,
            overspeed=overspeed,
        )
        floats = {k: _round(self._read_float(n), 1) for k, n in STATE_FLOATS.items()}
        self.tel.event(
            "autopilot_disconnect",
            logging.INFO if found["say"] is None else logging.WARNING,
            cause=found["cause"],
            detail=found["detail"],
            said=found["say"],
            changed={k: [previous[k], state[k]] for k in state if previous[k] != state[k]},
            commands=commands,
            stick_peak=peak,
            servo={a: _round(self._read_float(n), 3) for a, n in SERVO_OUTPUT.items()},
            has_power=self._read_int(DREF_AP_POWER),
            fail_warn=self._read_int(DREF_AP_FAIL),
            disconnect_warning=self._read_int(DREF_AP_DISCONNECT_WARNING),
            overspeed=overspeed,
            last_vimaan_cmd=last["cmd"] if last else None,
            last_vimaan_intent=last["intent"] if last else None,
            sec_since_vimaan_action=round(now - last["t"], 1) if last else None,
            **self._joystick_setup(),
            **floats,
        )
        if found["say"]:
            previous_cmd, self._current_cmd = self._current_cmd, None
            try:
                self._say(found["say"])
            finally:
                self._current_cmd = previous_cmd

    def _start_overlay(self):
        if self.overlay_state.mode == "off":
            self.tel.event("overlay_start", backend=None, mode="off")
            return
        overlay = Overlay(xp, self.overlay_state, log=self.tel.log)
        try:
            backend = overlay.start()
        except Exception as exc:
            self.tel.error("overlay_failed", exc)
            return
        self.overlay = overlay
        self.tel.event("overlay_start", backend=backend, mode=self.overlay_state.mode)

    def _on_overlay_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            mode = self.overlay_state.cycle_mode()
            if self.overlay is None and mode != "off":
                self._start_overlay()
            self.overlay_state.notice(f"Overlay {mode}")
            self.tel.event("overlay_mode", mode=mode)
        return 1

    def _register_command(self, name, description, handler):
        cmd_ref = xp.createCommand(name, description)
        xp.registerCommandHandler(cmd_ref, handler, 1, None)
        self._commands.append((cmd_ref, handler))

    def _sim_info(self):
        info: dict[str, Any] = {}
        try:
            info["versions"] = xp.getVersions()
        except Exception as exc:
            info["versions_error"] = repr(exc)
        info["num_engines"] = self._read_int(DREF_NUM_ENGINES)
        info["gear_retractable"] = self._read_int(DREF_GEAR_RETRACTABLE)
        return info

    # ------------------------------------------------------------------
    # Push-to-talk: the plugin only sends start/stop; the worker records
    # ------------------------------------------------------------------
    def _on_ptt_command(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self.OnPressCallback(refcon, source="command")
        elif phase == xp.CommandEnd:
            self.OnReleaseCallback(refcon, source="command")
        return 1

    def _on_self_test_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self._run_self_test()
        return 1

    def OnPressCallback(self, inRefcon, source="hotkey"):
        t_press = time.perf_counter()
        if self._active is not None:
            if self._active.get("t_release") is not None:
                self.tel.event("ptt_ignored", reason="still_processing", source=source)
            return  # otherwise key auto-repeat while held
        if not self.ready:
            self.tel.event("ptt_ignored", logging.WARNING, reason="not_ready", source=source)
            self._say("Vimaan is not ready yet")
            return
        cmd = self.tel.new_command_id()
        if not self.worker.send({"op": "start", "cmd": cmd}):
            self.tel.event("ptt_ignored", logging.ERROR, reason="worker_gone", source=source)
            return
        self._active = {"cmd": cmd, "t_press": t_press, "t_release": None, "source": source}
        if self._flight_loop_id:
            xp.scheduleFlightLoop(self._flight_loop_id, -1.0, 1)  # replies from next frame
        self.tel.event(
            "ptt_press",
            cmd=cmd,
            source=source,
            handler_ms=round((time.perf_counter() - t_press) * 1000, 3),
        )

    def OnReleaseCallback(self, inRefcon, source="hotkey"):
        active = self._active
        if active is None or active["t_release"] is not None:
            return
        t = time.perf_counter()
        active["t_release"] = t
        self.worker.send({"op": "stop", "cmd": active["cmd"]})
        self.tel.event(
            "ptt_release",
            cmd=active["cmd"],
            source=source,
            held_ms=round((t - active["t_press"]) * 1000, 1),
            handler_ms=round((time.perf_counter() - t) * 1000, 3),
        )

    def _end_command(self, outcome, **fields):
        """Close the active command without dispatching (error / timeout)."""
        active, self._active = self._active, None
        if active is None:
            return
        speech = self.OUTCOME_SPEECH.get(outcome, "An error occurred")
        if speech:
            self._say(speech)
        self.tel.command_done(active["cmd"], outcome, active["t_press"], **fields)

    # ------------------------------------------------------------------
    # Flight loop (main thread)
    # ------------------------------------------------------------------
    def _flight_loop(self, *_):
        t0 = time.perf_counter()
        try:
            if self.worker is not None:
                for msg in self.worker.poll():
                    self._on_worker_message(msg)
                self._check_worker()
            active = self._active
            if (
                active is not None
                and active["t_release"] is not None
                and time.perf_counter() - active["t_release"] > WORKER_REPLY_TIMEOUT_SEC
            ):
                self._end_command("timeout")
            self._service_cranks()
            self._service_deferred()
            self._service_baro_step()
            self._service_knob_step()
            self._service_capture()
            self._service_verifications()
            self._service_follow_ups()
            self._service_probe()
            self._service_bridge()
            self._sample_stick()
            self._watch_state()
            if self.overlay is not None:
                self.overlay.update()
        except Exception as exc:  # never let the flight loop die
            self.tel.error("flight_loop_error", exc)
        self.tel.tick(
            busy_ms=(time.perf_counter() - t0) * 1000,
            pending_verifications=len(self._verifications),
            paused=self._read_int(DREF_PAUSED),
            overlay_draw_ms_max=None
            if self.overlay is None
            else round(self.overlay.draw_ms_max, 3),
        )
        if (
            self._active is not None
            or self._self_test_pending
            or self._baro_step is not None
            or self._knob_step is not None
        ):
            return -1.0  # every frame while a reply is due or a knob is stepping
        return QUEUE_POLL_INTERVAL_SEC

    def _on_worker_message(self, msg):
        kind = msg.get("type")
        cmd = msg.get("cmd")
        active = self._active
        mine = active is not None and cmd == active["cmd"]
        if kind == "ready":
            self.ready = True
            intents = msg.get("intents")
            self._model_intents = set(intents) if intents else None
            self.tel.event(
                "stage",
                stage="model_load",
                ms=msg.get("load_ms"),
                **{k: v for k, v in msg.items() if k not in ("type", "load_ms")},
            )
            requested, self.STT_BACKEND = self.STT_BACKEND, msg.get("stt_backend", self.STT_BACKEND)
            if self.STT_BACKEND != requested:
                self.tel.event(
                    "stt_fallback",
                    logging.WARNING,
                    requested=requested,
                    using=self.STT_BACKEND,
                    **(msg.get("stt") or {}),
                )
            boot = time.perf_counter() - (self._worker_started_at or time.perf_counter())
            self.tel.event("worker_ready", boot_sec=round(boot, 2), restarts=self._worker_restarts)
            self._say("Vimaan AI CoPilot ready")
        elif kind == "fatal":
            self.tel.event("worker_fatal", logging.ERROR, **msg)
            self.ready = False
            self._say("Vimaan could not load its model. Check the log.")
        elif kind == "partial" and mine:
            self.overlay_state.on_partial(cmd, msg.get("text", ""))
        elif kind == "recording" and mine:
            self.tel.event("mic_open", cmd=cmd, open_ms=msg.get("open_ms"))
        elif kind == "captured" and active is not None and mine:
            self.tel.event(
                "stage",
                stage="capture",
                cmd=cmd,
                ms=round((time.perf_counter() - active["t_press"]) * 1000, 1),
                **{k: v for k, v in msg.items() if k not in ("type", "cmd")},
            )
        elif kind == "result" and active is not None and mine:
            self._active = None
            if msg.get("stt_ms") is not None:
                self.tel.event(
                    "stage",
                    stage="stt",
                    cmd=cmd,
                    ms=msg["stt_ms"],
                    backend=self.STT_BACKEND,
                    text=msg["text"],
                )
            self._execute_command(
                {
                    "cmd": cmd,
                    "text": msg["text"],
                    "prediction": msg,
                    "t_press": active["t_press"],
                    "t_release": active["t_release"],
                }
            )
        elif kind == "error" and mine:
            if msg.get("stage") == "stt":
                self.tel.event(
                    "stage_ended",
                    stage="stt",
                    cmd=cmd,
                    ms=msg.get("stt_ms"),
                    reason=msg.get("error"),
                )
            level = logging.INFO if msg.get("outcome") == "too_short" else logging.WARNING
            self.tel.event("worker_error", level, **msg)
            self._end_command(msg.get("outcome", "worker_error"))
        elif kind == "selftest":
            self._finish_self_test(msg.get("results", []))
        elif kind == "log":
            self.tel.log(f"[worker] {msg.get('msg')}", logging.WARNING)
        else:
            self.tel.event("worker_message_ignored", logging.DEBUG, msg=msg)

    # ------------------------------------------------------------------
    # Dispatch (main thread only)
    # ------------------------------------------------------------------
    def _execute_command(self, job):
        cmd, text, result = job["cmd"], job["text"], job["prediction"]
        self._current_cmd = cmd
        outcome = "failed"
        try:
            self.tel.event(
                "stage",
                stage="nlu",
                cmd=cmd,
                ms=result.get("nlu_ms"),
                normalized=result.get("normalized"),
                intent=result["intent"],
                confidence=round(result["confidence"], 3),
                slots=result["slots"],
            )
            intent_pred, slots = result["intent"], result["slots"]

            # "Override" / "cancel" answer Vimaan; they name no system, so
            # they are matched on the words and never reach the model.
            reply = pilot_reply(text)
            if reply is not None:
                outcome = self._pilot_reply(cmd, reply)
                return
            # "Override landing gear up" while gear up is held: the override,
            # said with the command. Not when it names a different command.
            normalized = result.get("normalized") or text
            if override_prefix(text) is not None and self._names_held(
                intent_pred, slots, result["confidence"], normalized
            ):
                self.tel.event("override_with_command", cmd=cmd, intent=intent_pred, text=text)
                outcome = self._pilot_reply(cmd, "override")
                return

            # A number on its own ("one thousand") names no command: it can
            # only be the answer to an open "to or two?" question.
            if bare_number(text) is not None:
                outcome = self._answer_question(cmd, text)
                return

            # A command's name alone ("landing gear"): ask which way rather
            # than stay silent. The handler asks when the choice is missing.
            # Only when the model found no command: a confident reading wins
            # ("heading" is also a word for the autopilot-mode command).
            bare = bare_command(normalized)
            found_none = (
                intent_pred in self.NON_ACTIONABLE_INTENTS
                or result["confidence"] < self.CONFIDENCE_FLOOR
            )
            if bare is not None and bare != intent_pred and found_none:
                self.tel.event("bare_command", cmd=cmd, intent=bare, predicted=intent_pred)
                outcome = self._dispatch(cmd, bare, {})
                return

            if result["confidence"] < self.CONFIDENCE_FLOOR:
                outcome = "low_confidence"
                self._say("Please repeat that")
                return

            if intent_pred in self.NON_ACTIONABLE_INTENTS:
                outcome = "non_actionable"
                ack = self.NON_ACTIONABLE_INTENTS[intent_pred]
                if ack:
                    self._say(ack)
                return

            handler = self.intent_to_command.get(intent_pred)
            if handler is None:
                outcome = "no_handler"
                self._say("Command not found")
                return

            # A fragment ("seventy", "city eight on") can be classified
            # confidently; only act when the pilot named the thing.
            if not has_command_word(intent_pred, normalized):
                outcome = "rejected"
                self.tel.event("implausible", cmd=cmd, reason="no_command_word", intent=intent_pred)
                self._say("Please repeat that")
                return
            # "altitude thousand": the number's start went unheard.
            if clipped_number(text):
                outcome = "rejected"
                self.tel.event("implausible", cmd=cmd, reason="clipped_number", intent=intent_pred)
                self._say("Say the whole number")
                return
            if intent_pred == "set_autopilot_heading" and heading_too_short(slots.get("degrees")):
                outcome = "rejected"
                self.tel.event(
                    "implausible", cmd=cmd, reason="one_digit_heading", value=slots.get("degrees")
                )
                self._say("Say the heading as three digits")
                return
            # "to" and "two" sound the same, so don't guess between two targets.
            either = two_or_to(intent_pred, text)
            if either and self._implausible_reading(intent_pred, either[1]):
                either = None  # "speed two eighty" on a jet: 80 kt is below its stall speed
            if either:
                outcome = "ambiguous"
                self._ask(cmd, intent_pred, *either)
                return
            outcome = self._dispatch(cmd, intent_pred, slots)
        except Exception as exc:
            self.tel.error("execute_error", exc, cmd=cmd, text=text)
            self._say("Command execution failed")
        finally:
            self._current_cmd = None
            extra = {}
            if job.get("t_release") is not None:
                extra["speech_to_action_ms"] = round(
                    (time.perf_counter() - job["t_release"]) * 1000, 1
                )
            self.tel.command_done(cmd, outcome, job["t_press"], text=text, **extra)

    def _ask(self, cmd, intent, as_two, as_to):
        """Ask which target was meant, and keep the question open for a reply."""
        self.tel.event("ambiguous", cmd=cmd, intent=intent, as_two=as_two, as_to=as_to)
        self._question = {
            "cmd": cmd,
            "intent": intent,
            "offered": (as_to, as_two),
            "expires": time.perf_counter() + self.QUESTION_WINDOW_SEC,
        }
        self._say(self._which_value(intent, as_two, as_to))

    def _answer_question(self, cmd, text):
        """A bare number: act on it only when it is one of the two values just
        offered. Anything else gets told what to say, never silence (in flight
        "one thousand" and "zero" were ignored without a word)."""
        q = self._question
        value = bare_number(text)
        if q is None or time.perf_counter() >= q["expires"]:
            self._question = None
            self.tel.event("implausible", cmd=cmd, reason="bare_number", value=value)
            self._say("Say the full command")
            return "rejected"
        # The reply can carry the same doubt: "two twenty" for 020 or 220.
        if two_or_to(q["intent"], text):
            self.tel.event("ambiguous", cmd=cmd, intent=q["intent"], reply_to=q["cmd"])
            as_to, as_two = q["offered"]
            self._say(self._which_value(q["intent"], as_two, as_to))
            return "ambiguous"
        choice = chosen(q["offered"], text)
        if choice is None:
            self.tel.event(
                "implausible",
                cmd=cmd,
                reason="not_offered",
                value=value,
                offered=list(q["offered"]),
            )
            self._say(f"Say it with the word {COMMAND_NOUN[q['intent']]}")
            return "rejected"
        slots = {TARGET_SLOTS[q["intent"]]: choice}
        self.tel.event("answered", cmd=cmd, reply_to=q["cmd"], intent=q["intent"], slots=slots)
        return self._dispatch(cmd, q["intent"], slots)

    def _names_held(self, intent, slots, confidence, normalized):
        """Whether an utterance that starts with "override" is about the held
        command: the same command (no slot contradicting it, so "override
        gear down" doesn't confirm a held gear up), or no other command at
        all (a garbled rest, as "override landing get up" was in flight)."""
        held = self._held if self._confirm_gate.pending_token() else None
        if held is None:
            return False
        if intent == held["intent"]:
            return all(held["slots"].get(k) in (None, v) for k, v in (slots or {}).items())
        names_other = (
            intent in self.intent_to_command
            and confidence >= self.CONFIDENCE_FLOOR
            and has_command_word(intent, normalized)
        )
        return not names_other

    def _pilot_reply(self, cmd, reply):
        """Override the held command, or cancel what is pending."""
        held = self._held if self._confirm_gate.pending_token() else None
        if reply == "override":
            if held is None:
                self.tel.event("override_ignored", cmd=cmd, reason="nothing_held")
                self._say("Nothing to override")
                return "rejected"
            self._confirm_gate.clear()
            self._held = None
            self.tel.event(
                "safety_override",
                logging.WARNING,
                cmd=cmd,
                held_cmd=held["cmd"],
                rule=held["rule"],
                intent=held["intent"],
                slots=held["slots"],
            )
            self._say("Override")
            skip = (*held["skip"], held["rule"])
            return self._dispatch(cmd, held["intent"], held["slots"], skip=skip)
        dropped = []
        if held is not None:
            dropped.append(held["intent"])
        if self._capture is not None:
            dropped.append("altitude capture")
            self._end_capture(False, "cancelled", speak=False)
        if self._question is not None and time.perf_counter() < self._question["expires"]:
            dropped.append(self._question["intent"])
        self._confirm_gate.clear()
        self._held = None
        self._question = None
        if not dropped:
            self._say("Nothing to cancel")
            return "no_op"
        self.tel.event("cancelled", cmd=cmd, dropped=dropped)
        self._say("Cancelled")
        return "cancelled"

    def _dispatch(self, cmd, intent, slots, skip=()):
        """Aircraft check, safety interlock, then the handler. Returns the outcome.
        ``skip``: safety rules the pilot overrode for this command."""
        # Commands this aircraft physically can't do are rejected before
        # the safety gate, which would otherwise ask the pilot to confirm
        # "gear up" on a fixed-gear Cessna.
        not_applicable = self._not_applicable(intent, slots)
        if not_applicable:
            self.tel.event("not_applicable", cmd=cmd, intent=intent, reason=not_applicable)
            self._say(not_applicable)
            return "rejected"

        # Safety interlock: gate irreversible/dangerous commands on a
        # confirmation (repeat the command) when risky in the current flight
        # context. Fails safe if the relevant datarefs are unavailable.
        ctx = self._read_safety_context(slots)
        verdict = evaluate_safety(intent, slots, ctx, skip=skip)
        token = command_token(intent, slots)
        if verdict.requires_confirmation and not self._confirm_gate.confirm(token):
            self._confirm_gate.arm(token)
            # ``skip`` carries rules already overridden for this command, so a
            # second rule's hold can't bring the first one back.
            self._held = {
                "cmd": cmd,
                "intent": intent,
                "slots": dict(slots),
                "rule": verdict.rule,
                "skip": tuple(skip),
            }
            self.tel.event(
                "safety_hold",
                logging.WARNING,
                cmd=cmd,
                rule=verdict.rule,
                fail_safe=verdict.fail_safe,
                context=ctx.__dict__,
            )
            self._say(f"{verdict.message}. Say override, or repeat to confirm.")
            return "safety_hold"
        handler = self.intent_to_command[intent]
        with self.tel.stage("dispatch", cmd=cmd, intent=intent) as res:
            outcome = handler(slots, intent) or "executed"
            res["outcome"] = outcome
        if outcome in ("executed", "pending"):
            # What the autopilot watch blames a mode change on: every command
            # that reached the sim, not only the ones that set a target.
            self._last_action = {
                "cmd": cmd,
                "intent": intent,
                "state": slots.get("state"),
                "t": time.perf_counter(),
            }
        # A different command that actually ran cancels a pending
        # confirmation and the open "to or two?" question; a misheard or
        # incomplete one ("engine") must not.
        if outcome == "executed":
            self._confirm_gate.clear()
            self._held = None
            self._question = None
        return outcome

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _run_self_test(self, *_):
        """Ctrl-Z: run canned utterances through text -> NLU in the worker (STT
        bypassed, nothing actuated) and check each resolves to the expected
        intent with every slot its handler needs. Spoken summary on reply."""
        if not self.ready:
            self._say("Vimaan is not ready yet")
            return
        if self._self_test_pending:
            return
        self._self_test_pending = True
        self._self_test_started = time.perf_counter()
        self._say("Running self test")
        self.worker.send({"op": "selftest", "probes": [list(p) for p in self._self_test_probes()]})
        if self._flight_loop_id:
            xp.scheduleFlightLoop(self._flight_loop_id, -1.0, 1)

    def _self_test_probes(self):
        """The probes for commands the loaded model was trained on: v19 knows
        nothing of Tier 2, and "15 of 23 passed" would hide a real failure."""
        known = self._model_intents
        return tuple(p for p in self.SELF_TEST_PROBES if known is None or p[1] in known)

    def _finish_self_test(self, results):
        self._self_test_pending = False
        elapsed_ms = round((time.perf_counter() - self._self_test_started) * 1000, 1)
        self._self_test_started = 0.0
        passed = 0
        for r in results:
            intent, slots = r.get("intent"), r.get("slots") or {}
            missing = [s for s in REQUIRED_SLOTS.get(intent, ()) if not slots.get(s)]
            ok = (
                "error" not in r
                and intent == r["expected"]
                and intent in self.intent_to_command
                and not missing
            )
            passed += ok
            self.tel.event(
                "self_test_probe",
                logging.INFO if ok else logging.WARNING,
                text=r["text"],
                expected=r["expected"],
                intent=intent,
                confidence=round(r.get("confidence") or 0.0, 3),
                slots=slots,
                missing_slots=missing,
                error=r.get("error"),
                ok=ok,
            )
        total = len(self._self_test_probes())
        self.tel.event(
            "stage",
            stage="self_test",
            ms=elapsed_ms,
            passed=passed,
            total=total,
            not_in_model=len(self.SELF_TEST_PROBES) - total,
        )
        self._say(f"Self test complete. {passed} of {total} passed.")

    # ------------------------------------------------------------------
    # Sim I/O helpers (every read/write/command is logged with the command id)
    # ------------------------------------------------------------------
    def _say(self, text, origin=None):
        """Speak, and log it. ``origin="probe"`` keeps a probe announcement out
        of the overlay's view of the last voice command."""
        extra = {"origin": origin} if origin else {}
        self.tel.event("speak", logging.DEBUG, cmd=self._current_cmd, text=text, **extra)
        xp.speakString(text)

    @staticmethod
    def _find(name):
        return xp.findDataRef(name)

    def _read_float(self, name):
        dref = self._find(name)
        if not dref:
            return None
        try:
            return float(xp.getDataf(dref))
        except Exception:
            return None

    def _read_int(self, name):
        dref = self._find(name)
        if not dref:
            return None
        try:
            return int(xp.getDatai(dref))
        except Exception:
            return None

    def _read_array(self, name, index, is_float=False):
        """Read element ``index`` of an int[]/float[] dataref, or None."""
        dref = self._find(name)
        if not dref:
            return None
        values: list = []
        try:
            if is_float:
                xp.getDatavf(dref, values, index, 1)
            else:
                xp.getDatavi(dref, values, index, 1)
            return values[0] if values else None
        except Exception as exc:
            self.tel.event("dataref_read_error", logging.WARNING, dataref=name, error=repr(exc))
            return None

    @staticmethod
    def _which_value(intent, as_two, as_to):
        three_digits = {
            "set_autopilot_heading": "Heading",
            "set_flight_level": "Flight level",
            "set_course": "Course",
        }
        if intent in three_digits:
            name = three_digits[intent]
            a, b = (spell_digits(f"{int(v):03d}") for v in (as_to, as_two))
            return f"{name} {a} or {b}? Say all three digits."
        unit = {"set_autopilot_altitude": " feet", "set_vertical_speed": " feet per minute"}
        name = COMMAND_NOUN[intent].capitalize()
        return (
            f"{name} {as_to} or {as_two}{unit.get(intent, '')}? Say it again without the word to."
        )

    def _write(self, name, value, is_float=True):
        """Write a dataref and read it straight back. Returns True on success.
        Same-frame only: ``_hold_check`` confirms the value later."""
        dref = self._find(name)
        if not dref:
            self.tel.event(
                "sim_write",
                logging.ERROR,
                cmd=self._current_cmd,
                dataref=name,
                value=value,
                ok=False,
                reason="dataref_not_found",
            )
            return False
        if is_float:
            xp.setDataf(dref, float(value))
            readback: Any = xp.getDataf(dref)
            ok = abs(readback - float(value)) < 0.5
        else:
            xp.setDatai(dref, int(value))
            readback = xp.getDatai(dref)
            ok = readback == int(value)
        self.tel.event(
            "sim_write",
            logging.INFO if ok else logging.WARNING,
            cmd=self._current_cmd,
            dataref=name,
            value=value,
            readback=readback,
            ok=ok,
        )
        return ok

    def _fire_command(self, command_path):
        if not command_path:
            return False
        cmd_ref = xp.findCommand(command_path)
        if not cmd_ref:
            self.tel.event(
                "sim_command",
                logging.ERROR,
                cmd=self._current_cmd,
                command=command_path,
                ok=False,
                reason="command_not_found",
            )
            return False
        self._firing = command_path  # our own, to the autopilot-off watch
        try:
            xp.commandOnce(cmd_ref)
        finally:
            self._firing = None
        self.tel.event("sim_command", cmd=self._current_cmd, command=command_path, ok=True)
        return True

    def _hold_check(self, label, name, value, is_float=True, modulo=None, tolerance=0.5):
        """Did a written value stay? The read-back in ``_write`` is in the same
        frame, so it can't tell if the sim or aircraft replaced it. Also records
        the aircraft now and shortly after, to compare target with response."""
        read = self._read_float if is_float else self._read_int

        def current():
            actual = read(name)
            return actual % modulo if actual is not None and modulo else actual

        self._expect(label, current, value % modulo if modulo else value, tolerance=tolerance)
        now = time.perf_counter()
        for after in FOLLOW_UP_SEC:
            self._follow_ups.append(
                {"cmd": self._current_cmd, "label": label, "after_sec": after, "due": now + after}
            )
        self._service_follow_ups()

    def _aircraft_state(self):
        state: dict[str, Any] = {}
        for key, name in STATE_FLOATS.items():
            value = self._read_float(name)
            state[key] = None if value is None else round(value, 1)
        for key, name in {**STATE_RADIOS, **AUTOPILOT_STATE}.items():
            state[key] = self._read_int(name)
        return state

    def _service_follow_ups(self):
        now = time.perf_counter()
        due = [f for f in self._follow_ups if f["due"] <= now]
        if not due:
            return
        self._follow_ups = [f for f in self._follow_ups if f["due"] > now]
        state = self._aircraft_state()
        for f in due:
            self.tel.event(
                "aircraft_state", cmd=f["cmd"], after=f["label"], after_sec=f["after_sec"], **state
            )

    def _watch_state(self):
        """Log autopilot changes as they happen, including ones Vimaan didn't
        make, with the time since Vimaan's last action. The log can't tell a
        panel button from the aircraft dropping out; this narrows it down."""
        now = time.perf_counter()
        if now < self._next_state_watch:
            return
        self._next_state_watch = now + STATE_WATCH_SEC
        self._check_aircraft()
        state = {key: self._read_int(name) for key, name in AUTOPILOT_STATE.items()}
        previous, self._autopilot_state = self._autopilot_state, state
        if previous is None or previous == state:
            return
        last = self._last_action
        self.tel.event(
            "autopilot_state_change",
            changed={k: [previous[k], state[k]] for k in state if previous[k] != state[k]},
            last_vimaan_cmd=last["cmd"] if last else None,
            last_vimaan_intent=last["intent"] if last else None,
            sec_since_vimaan_action=round(now - last["t"], 1) if last else None,
            **state,
        )
        if dropped(previous, state):
            self._on_autopilot_drop(previous, state, now)

    # ------------------------------------------------------------------
    # Automated-test bridge (vimaan_nlu.sim_bridge, Phase A)
    # ------------------------------------------------------------------
    def _service_bridge(self):
        """Switch on or off with the switch file, hand back finished results,
        and take new requests. Off: one file check every quarter second."""
        now = time.perf_counter()
        if now < self._bridge_next:
            return
        self._bridge_next = now + BRIDGE_POLL_SEC
        flag, inbox, outbox = sim_bridge.paths(self.tel.log_dir)
        on = flag.exists()
        if on != self._bridge_on:
            self._bridge_on = on
            self.tel.event("test_bridge", on=on, folder=str(flag.parent))
            self._say("Test bridge on" if on else "Test bridge off")
        if not on:
            return
        # A command still has work due: read-backs, or a deferred step (the
        # 737's autothrottle engages after its arm switch).
        checking = {v["cmd"] for v in self._verifications} | {d["cmd"] for d in self._deferred}
        checking |= {j["cmd"] for j in (self._baro_step, self._knob_step) if j is not None}
        for result in self._say_results.finished(now, checking):
            sim_bridge.write_json(outbox / f"{result['id']}.json", result)
        for req in sim_bridge.take_requests(inbox):
            result = self._bridge_request(req)
            if result is not None:  # a say answers when its command is done
                sim_bridge.write_json(outbox / f"{req['id']}.json", {"id": req["id"], **result})

    def _bridge_request(self, req):
        why = sim_bridge.problem(req)
        if why:
            return {"ok": False, "error": why}
        op = req["op"]
        if op not in ("state", "read"):
            fields = {k: v for k, v in req.items() if k not in ("id", "op")}
            self.tel.event("test_request", request=req["id"], op=op, **fields)
        try:
            if op == "state":
                return {"ok": True, **self._bridge_state()}
            if op == "say":
                error = self._bridge_say(req)
                return None if error is None else {"ok": False, "error": error}
            if op == "read":
                return {"ok": True, "values": {n: self._read_any(n) for n in req["datarefs"]}}
            if op == "write":
                return self._write_any(req["dataref"], req["value"])
            if op == "command":
                return self._bridge_command(req["name"], req.get("phase", "once"))
            if op == "place":
                xp.placeUserAtLocation(
                    float(req["latitude"]),
                    float(req["longitude"]),
                    float(req["elevation_ft"]) * METRES_PER_FOOT,
                    float(req["heading_true"]),
                    float(req["speed_kt"]) * MPS_PER_KNOT,
                )
            elif op == "airport":
                xp.placeUserAtAirport(req["icao"])
            elif op == "aircraft":
                xp.setUsersAircraft(req["path"])
            # A new position resets the autopilot: not a disconnect to call out.
            self._autopilot_state = None
            self._quiet_until = time.perf_counter() + RELOAD_QUIET_SEC
            return {"ok": True}
        except Exception as exc:
            self.tel.error("test_request_failed", exc, request=req["id"], op=op)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _bridge_say(self, req):
        """Start a command as if the pilot had pressed push-to-talk and said
        it: a WAV goes through recognition, text to the model only."""
        if not self.ready or self.worker is None:
            return "Vimaan is not ready"
        if self._active is not None or self._say_results.busy:
            return "busy"
        cmd = self.tel.new_command_id()
        if req.get("wav"):
            msg, said = {"op": "wav", "cmd": cmd, "path": req["wav"]}, {"wav": req["wav"]}
        else:
            msg, said = {"op": "text", "cmd": cmd, "text": req["text"]}, {"text": req["text"]}
        now = time.perf_counter()
        self._active = {"cmd": cmd, "t_press": now, "t_release": now, "source": "test"}
        self._say_results.start(req["id"], cmd, now)
        self.tel.event("ptt_press", cmd=cmd, source="test", request=req["id"], **said)
        self.tel.event("ptt_release", cmd=cmd, source="test", held_ms=0.0)
        if not self.worker.send(msg):
            self._end_command("worker_error")
            return None  # the result reports the error
        if self._flight_loop_id:
            xp.scheduleFlightLoop(self._flight_loop_id, -1.0, 1)
        return None

    def _bridge_command(self, name, phase):
        """A command from the bridge counts as the pilot's, not Vimaan's."""
        ref = xp.findCommand(name)
        if not ref:
            return {"ok": False, "error": f"no command {name}"}
        {"once": xp.commandOnce, "begin": xp.commandBegin, "end": xp.commandEnd}[phase](ref)
        self.tel.event("sim_command", cmd=None, command=name, ok=True, origin="test", phase=phase)
        return {"ok": True}

    def _bridge_state(self):
        info = self.profile.info
        agl_m = self._read_float("sim/flightmodel/position/y_agl")
        return {
            "session": self.tel.session_id,
            "events": str(self.tel.events_path),
            "ready": self.ready,
            "busy": self._active is not None or self._say_results.busy,
            "probe": self._probe is not None,
            # What the loaded model knows (a runner refuses a checklist the
            # model wasn't trained for); None from a worker that doesn't say.
            "model_intents": None if self._model_intents is None else sorted(self._model_intents),
            "aircraft": {
                "icao": info.icao,
                "key": info.key,
                "acf": info.acf_path,
                "description": info.description,
            },
            "paused": self._read_int(DREF_PAUSED),
            "on_ground": self._read_int("sim/flightmodel/failures/onground_any"),
            "agl_ft": None if agl_m is None else round(agl_m / METRES_PER_FOOT, 1),
            "flying": {k: _round(self._read_float(n), 1) for k, n in STATE_FLOATS.items()},
            "autopilot": {k: self._read_int(n) for k, n in AUTOPILOT_STATE.items()},
        }

    @staticmethod
    def _dataref_kind(ref):
        """'int', 'float', 'ints', 'floats', 'data' or None, from X-Plane's type bits."""
        types = xp.getDataRefTypes(ref)
        if types & 1 and not types & 6:
            return "int"
        if types & 6:
            return "float"
        if types & 16:
            return "ints"
        if types & 8:
            return "floats"
        return "data" if types & 32 else None

    def _read_any(self, name):
        """Any dataref by name, "name[i]" for one element of an array; a whole
        array comes back as a list (first 64 values)."""
        base, index = sim_bridge.parse_dataref(name)
        ref = self._find(base)
        if not ref:
            return None
        kind = self._dataref_kind(ref)
        values: list = []
        if kind in ("ints", "floats"):
            get = xp.getDatavi if kind == "ints" else xp.getDatavf
            if index is not None:
                get(ref, values, index, 1)
                return values[0] if values else None
            get(ref, values, 0, min(64, get(ref, None, 0, -1)))
            return values
        if index is not None:
            return None
        if kind == "int":
            return xp.getDatai(ref)
        if kind == "float":
            return xp.getDataf(ref)
        return xp.getDatas(ref) if kind == "data" else None

    def _write_any(self, name, value):
        base, index = sim_bridge.parse_dataref(name)
        ref = self._find(base)
        if not ref:
            return {"ok": False, "error": f"no dataref {base}"}
        if not xp.canWriteDataRef(ref):
            return {"ok": False, "error": f"{base} is read-only"}
        kind = self._dataref_kind(ref)
        if kind == "ints" and index is not None:
            xp.setDatavi(ref, [int(value)], index, 1)
        elif kind == "floats" and index is not None:
            xp.setDatavf(ref, [float(value)], index, 1)
        elif kind == "int" and index is None:
            xp.setDatai(ref, int(value))
        elif kind == "float" and index is None:
            xp.setDataf(ref, float(value))
        else:
            return {"ok": False, "error": f"can't write {name} ({kind})"}
        back = self._read_any(name)
        ok = back is not None and abs(float(back) - float(value)) < 1e-3
        self.tel.event(
            "sim_write", cmd=None, dataref=name, value=value, readback=back, ok=ok, origin="test"
        )
        return {"ok": ok, "readback": back}

    def _read_string(self, name):
        dref = self._find(name)
        if not dref:
            return ""
        try:
            return (xp.getDatas(dref) or "").strip("\x00 ")
        except Exception as exc:
            self.tel.event("dataref_read_error", logging.WARNING, dataref=name, error=repr(exc))
            return ""

    def _read_values(self, name, count, is_float=False):
        """The first ``count`` elements of an array dataref ([] if unreadable)."""
        dref = self._find(name)
        if not dref or not count:
            return []
        values: list = []
        try:
            (xp.getDatavf if is_float else xp.getDatavi)(dref, values, 0, count)
        except Exception as exc:
            self.tel.event("dataref_read_error", logging.WARNING, dataref=name, error=repr(exc))
            return []
        return values[:count]

    @staticmethod
    def _acf_path():
        try:
            return xp.getNthAircraftModel(0)[1] or ""
        except Exception:
            return ""

    @staticmethod
    def _has_command(name):
        return bool(name) and bool(xp.findCommand(name))

    def _check_aircraft(self):
        """Rebuild the profile when a different aircraft is loaded."""
        identity = (self._read_string(DREF_AIRCRAFT_ICAO), self._acf_path())
        if identity[0] and identity != self._aircraft:
            self._aircraft = identity
            self._load_aircraft(*identity)

    def _aircraft_info(self, icao, acf_path):
        """Layer 2: what X-Plane reports about the loaded aircraft."""
        num_engines = self._read_int(DREF_NUM_ENGINES)
        detents = self._read_int(DREF_FLAP_DETENTS)
        single_axis = self._read_int(DREF_SINGLE_AXIS_AP)
        retractable = self._read_int(DREF_GEAR_RETRACTABLE)
        return AircraftInfo(
            icao=icao,
            acf_path=acf_path,
            **{key: self._read_string(name) for key, name in AIRCRAFT_STRINGS.items()},
            num_engines=num_engines,
            engine_types=tuple(self._read_values(DREF_ENGINE_TYPE, num_engines or 0)),
            flap_detents=detents,
            flap_degrees=tuple(self._read_values(DREF_FLAP_DEGREES, 10, is_float=True)),
            gear_retractable=None if retractable is None else bool(retractable),
            autopilot_type=self._read_int(DREF_AUTOPILOT_TYPE),
            single_axis_autopilot=None if single_axis is None else bool(single_axis),
            **{key: self._read_float(name) for key, name in V_SPEEDS.items()},
        )

    def _load_aircraft(self, icao, acf_path):
        info = self._aircraft_info(icao, acf_path)
        self.profile = self._resolve(info)
        self.tel.event("aircraft", **self.profile.summary())
        # An aircraft's scripts may register their datarefs after the load is
        # seen: look again shortly after.
        for delay in OVERRIDE_RECHECK_SEC:
            self._later(delay, lambda ident=(icao, acf_path): self._recheck_overrides(ident))
        self._hook_off_commands()
        self.tel.event("joystick", **self._joystick_setup())
        self._capture = None
        self._baro_step = None
        self._knob_step = None
        self._athr_scan = (acf_path, None)
        threading.Thread(
            target=self._scan_aircraft_files, args=(acf_path,), daemon=True, name="vimaan-acf-scan"
        ).start()

    def _resolve(self, info):
        return resolve(
            info, self._has_command, self._probe_results(info.key), has_dataref=self._dataref_live
        )

    def _recheck_overrides(self, identity):
        """Re-resolve if the aircraft is the same but its overrides changed
        (its scripts registered their datarefs after the load was seen)."""
        if identity != self._aircraft:
            return
        profile = self._resolve(self.profile.info)
        if profile.overrides != self.profile.overrides:
            self.profile = profile
            self.tel.event("aircraft", reason="overrides_changed", **profile.summary())

    @staticmethod
    def _dataref_live(name):
        """Registered right now: an aircraft's own datarefs go when it unloads."""
        ref = xp.findDataRef(name)
        return ref is not None and bool(xp.isDataRefGood(ref))

    def _scan_aircraft_files(self, acf_path):
        """Background thread: file reads only, no X-Plane calls."""
        try:
            shown = files_mention(acf_path)
        except Exception:  # never let a scan take the plugin down
            shown = None
        try:
            limits = acf_flap_limits(acf_path)
        except Exception:
            limits = {}
        if self._athr_scan and self._athr_scan[0] == acf_path:
            self._athr_scan = (acf_path, shown)
            self._flap_limits = (acf_path, limits)
            self.tel.event(
                "aircraft_files", acf=acf_path, autothrottle_shown=shown, flap_limits=limits
            )

    def _flap_target_notch(self, slots):
        """The notch a "flaps full" / "flaps 15" asks for (None if unknown)."""
        detents = self.profile.info.flap_detents
        if not detents:
            return None
        if slots.get("position") == "full":
            return detents
        value, ok, _ = validate_slot("flap_setting", slots.get("flap_setting"))
        return self.profile.flap_notch(int(float(value))) if ok else None

    def _flap_limit_for(self, slots):
        """The speed limit for the flap setting asked for. X-Plane publishes
        only the first notch's (acf_Vfe); the full-flap one comes from the
        .acf, read in the background on load. Unknown notch: the lowest."""
        info = self.profile.info
        scan = self._flap_limits
        limits = scan[1] if scan and scan[0] == info.acf_path else {}
        first = limits.get("first") or info.vfe_kt or None
        full = limits.get("full")
        notch = self._flap_target_notch(slots)
        if notch is None:
            known = [v for v in (first, full) if v]
            return min(known) if known else None
        return flap_limit_kt(notch, info.flap_detents or 0, first, full)

    def _autothrottle_shown(self):
        scan = self._athr_scan
        return scan[1] if scan and scan[0] == self.profile.info.acf_path else None

    def _aircraft_name(self):
        """Short name for the overlay: the description if it's short, else ICAO."""
        info = self.profile.info
        return info.description if 0 < len(info.description) <= 28 else info.icao

    def _probe_folder(self):
        return str(self.tel.log_dir / "aircraft")

    def _probe_results(self, key):
        """Layer 4: what the probe measured on this aircraft, if anything."""
        return probe.load_results(probe.results_path(self._probe_folder(), key))

    # ------------------------------------------------------------------
    # Aircraft probe (vimaan_nlu.probe): pilot-paced, one step per press
    # ------------------------------------------------------------------
    PROBE_SPOKEN = {
        "works": "works",
        "no_change": "no change",
        "missing": "not on this aircraft",
        "inconclusive": "inconclusive",
    }

    def _on_probe_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self._probe_guarded(self._probe_press)
        return 1

    def _on_probe_stop_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self._probe_guarded(self._probe_stop)
        return 1

    def _on_probe_skip_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self._probe_guarded(lambda: self._probe_move(1))
        return 1

    def _on_probe_back_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self._probe_guarded(lambda: self._probe_move(-1))
        return 1

    def _probe_guarded(self, action):
        try:
            action()
        except Exception as exc:  # never leave a probe half-run
            self.tel.error("probe_error", exc)
            self._probe_finish("error", "Probe stopped by an error. Check the autopilot.")

    def _probe_path(self):
        return probe.results_path(self._probe_folder(), self.profile.info.key)

    def _probe_state(self):
        state = {}
        for key, (name, is_float) in probe.STATE.items():
            value = (self._read_float if is_float else self._read_int)(name)
            state[key] = round(value, 4) if is_float and value is not None else value
        return state

    def _probe_begin(self):
        ctx = self._read_safety_context()
        where = probe.situation(ctx.on_ground, ctx.agl_ft, ctx.ground_speed_kt)
        if where is None:
            self._say(
                "The probe needs the aircraft stopped on the ground, "
                "or airborne above fifteen hundred feet"
            )
            return
        steps = probe.steps_for(where)
        found = {
            role: bool(name) and self._has_command(name)
            for role, name in self.profile.commands.items()
        }
        # An unfinished run carries on where it stopped, even after a restart.
        start = probe.resume_index(probe.load_record(self._probe_path()), where)
        self._probe = {
            "where": where,
            "steps": steps,
            "index": start,
            "phase": "ready",
            "results": {},
            "details": [],
            "found": found,
            "cautioned": False,
            "stopping": False,
        }
        self.tel.event(
            "probe_start",
            where=where,
            steps=len(steps),
            start=start,
            aircraft=self.profile.info.key,
            name=self._aircraft_name(),
            next=steps[start].label,
            found=found,
        )
        if start:
            self._say(
                f"Probe ready. Continuing at {steps[start].label}, step {start + 1} of "
                f"{len(steps)}. Press again to run it.",
                origin="probe",
            )
        else:
            self._say(
                f"Probe ready. {len(steps)} {where} steps. Press again for {steps[0].label}.",
                origin="probe",
            )

    def _probe_press(self):
        p = self._probe
        if p is None:
            self._probe_begin()
            return
        if p["phase"] != "ready":
            self._say("Probe busy", origin="probe")
            return
        step = p["steps"][p["index"]]
        before = self._probe_state()
        brake = self._read_float(DREF_PARKING_BRAKE)
        why = probe.ready_for(step, before, None if brake is None else brake > 0.5)
        if why:
            self._probe_blocked(step, why)
            return
        warning = probe.caution(step, before)
        if warning and not p["cautioned"]:
            p["cautioned"] = True  # once per run: the next press runs it anyway
            self._probe_blocked(step, f"{warning}. Select them, or press again to run it anyway.")
            return
        written, name = None, None
        if step.write is None:
            name = self.profile.command(step.role)
            if not (name and self._has_command(name)):
                self._probe_done(step, "missing", before, before)
                return
            self._fire_command(name)
        else:
            written = probe.write_value(step, before, self.profile.info.flap_detents)
            if written is None or not self._write(probe.STATE[step.write][0], written):
                self._probe_done(step, "inconclusive", before, before)
                return
        p.update(
            phase="acting",
            step=step,
            before=before,
            written=written,
            name=name,
            attempt=0,
            due=time.perf_counter() + probe.SETTLE_SEC,
        )
        self.tel.event(
            "probe_act",
            role=step.role,
            label=step.label,
            index=p["index"],
            total=len(p["steps"]),
            command=name,
            written=written,
        )

    def _probe_blocked(self, step, message):
        """A press that didn't run its step, and why (shown on the overlay)."""
        p = self._probe
        assert p is not None
        self.tel.event(
            "probe_blocked",
            label=step.label,
            index=p["index"],
            total=len(p["steps"]),
            message=message,
        )
        self._say(message, origin="probe")

    def _probe_move(self, delta):
        """Skip ahead or go back one step without running anything, to reach
        a single step. Wraps around, so every step can be reached."""
        p = self._probe
        if p is None:
            self._say("Press the probe key first", origin="probe")
            return
        if p["phase"] != "ready":
            self._say("Probe busy", origin="probe")
            return
        total = len(p["steps"])
        p["index"] = (p["index"] + delta) % total
        step = p["steps"][p["index"]]
        self.tel.event(
            "probe_move",
            label=step.label,
            index=p["index"],
            total=total,
            direction="skip" if delta > 0 else "back",
        )
        verb = "Skipped to" if delta > 0 else "Back to"
        self._say(f"{verb} {step.label}, step {p['index'] + 1} of {total}.", origin="probe")

    def _probe_stop(self):
        p = self._probe
        if p is None:
            self._probe_reset()
        elif p["phase"] != "ready":
            # Mid-step the control is pressed and not yet put back: finish it.
            p["stopping"] = True
            self._say("Stopping after this step", origin="probe")
        else:
            self._probe_finish("stopped", "Probe stopped. Results saved." + self._probe_hint(p))

    def _probe_reset(self):
        """The stop key with no probe running: forget where an unfinished run
        stopped, so the next one starts from the first step."""
        path = self._probe_path()
        if not probe.load_record(path).get("resume"):
            self._say("No probe running", origin="probe")
            return
        probe.save_results(path, {"resume": {"air": None, "ground": None}})
        self.tel.event("probe_reset", path=path)
        self._say("Probe reset. The next run starts from the first step.", origin="probe")

    @staticmethod
    def _probe_hint(p):
        """Where the next press of the probe key carries on, if anywhere."""
        if p["index"] >= len(p["steps"]):
            return " Every step is measured."
        return f" The probe key continues with {p['steps'][p['index']].label}."

    def _service_probe(self):
        p = self._probe
        if p is None or p["phase"] == "ready" or time.perf_counter() < p["due"]:
            return
        self._probe_guarded(lambda: self._advance_probe(p))

    def _advance_probe(self, p):
        step, before = p["step"], p["before"]
        now = self._probe_state()
        if probe.autopilot_dropped(step, before, now):
            self._probe_pause(p, step, before, now)
            return
        if p["phase"] == "acting":
            p["verdict"] = probe.classify(step, before, now, True, p["written"])
            p["after"] = now
            if step.write is not None:
                self._write(probe.STATE[step.write][0], before[step.write])
            elif step.role == "baro_std":
                # STD is a toggle on some aircraft (737 EFIS) and a plain
                # "set 29.92" on others: press it again, then put the old
                # setting back, which covers both.
                self._fire_command(p["name"])
                if before.get("baro") is not None:
                    self._write(probe.STATE["baro"][0], before["baro"])
            elif probe.changes(before, now):
                self._fire_command(p["name"])  # press it again
            p.update(phase="restoring", due=time.perf_counter() + probe.SETTLE_SEC)
            return
        if probe.restored(step, before, now):
            self._probe_done(step, p["verdict"], before, p["after"])
            return
        while (roles := probe.restore_presses(before, now, p["attempt"])) is not None:
            p["attempt"] += 1
            names = [self.profile.command(role) for role in roles]
            names = [n for n in names if n and self._has_command(n)]
            if names:
                for name in names:
                    self._fire_command(name)
                p["due"] = time.perf_counter() + probe.SETTLE_SEC
                return
        left = probe.leftovers(step, before, now)
        if set(left) <= probe.HARMLESS_LEFTOVERS:
            note = f"{probe.describe(left)} stayed on, which is harmless"
            self._probe_done(step, p["verdict"], before, p["after"], now=now, note=note)
            return
        self._probe_record(step, p["verdict"], before, p["after"], now=now)
        p["index"] += 1  # measured: carry on after it, not with it
        self._probe_finish(
            "restore_failed",
            f"Probe stopped. After {step.label}, {probe.describe(left)} did not go back. "
            f"Check the autopilot.{self._probe_hint(p)}",
        )

    def _probe_pause(self, p, step, before, now):
        """The autopilot dropped out mid-step: that's the aircraft (or the
        controls), not the control under test. Nothing is recorded; the same
        step runs again on the next press, once the pilot has re-engaged.
        A control already pressed is put back first (a mode the flight
        director still shows would otherwise carry into the next steps)."""
        if p["phase"] == "acting":
            if step.write is not None:
                self._write(probe.STATE[step.write][0], before[step.write])
            elif p["name"] and probe.undo_needed(step, before, now):
                self._fire_command(p["name"])
        p["phase"] = "ready"
        self.tel.event(
            "probe_paused",
            logging.WARNING,
            role=step.role,
            label=step.label,
            index=p["index"],
            total=len(p["steps"]),
            changed=probe.changes(before, now),
        )
        if p["stopping"]:
            self._probe_finish(
                "stopped",
                f"Probe stopped. The autopilot disconnected during {step.label}."
                + self._probe_hint(p),
            )
            return
        self._say(
            f"The autopilot disconnected during {step.label}. "
            "Re-engage it, then press the probe key to retry.",
            origin="probe",
        )

    def _probe_record(self, step, verdict, before, after, now=None, note=None):
        """Log a measured step and save it straight away, with the place the
        run has reached, so a stop or a crash loses nothing."""
        p = self._probe
        assert p is not None  # only called while a probe runs
        p["results"][step.role] = verdict
        detail = {
            "role": step.role,
            "verdict": verdict,
            "changed": probe.changes(before, after),
            "restored": now is None,
        }
        if now is not None:
            detail["left_changed"] = probe.changes(before, now)
        if note:
            detail["note"] = note
        p["details"].append(detail)
        steps, index = p["steps"], p["index"]
        following = steps[index + 1] if index + 1 < len(steps) else None
        going_on = now is None or note is not None
        self.tel.event(
            "probe_step",
            logging.INFO if going_on else logging.WARNING,
            label=step.label,
            index=index,
            total=len(steps),
            next=following.label if following and going_on else None,
            **detail,
        )
        self._probe_save(
            p,
            steps={step.role: verdict},
            details=[detail],
            resume=following.role if following else None,
            end="in_progress",
        )

    def _probe_save(self, p, *, resume, end, steps=None, details=None):
        path = self._probe_path()
        record = {
            "aircraft": self.profile.summary(),
            "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session": self.tel.session_id,
            "where": p["where"],
            "end": end,
            "steps": steps or {},
            "details": details or [],
            "commands_found": p["found"],
            "resume": {p["where"]: resume},
        }
        try:
            probe.save_results(path, record)
        except OSError as exc:
            self.tel.error("probe_save_error", exc, path=path)
        return path

    def _probe_done(self, step, verdict, before, after, now=None, note=None):
        self._probe_record(step, verdict, before, after, now=now, note=note)
        p = self._probe
        assert p is not None
        p["index"] += 1
        p["phase"] = "ready"
        label = step.label[:1].upper() + step.label[1:]  # "V NAV" stays "V NAV"
        spoken = f"{label}: {self.PROBE_SPOKEN[verdict]}."
        if note:
            spoken += f" {note[:1].upper()}{note[1:]}."
        if p["index"] >= len(p["steps"]):
            self._say(spoken, origin="probe")
            self._probe_finish("complete", "Probe complete. Results saved.")
        elif p["stopping"]:
            self._say(spoken, origin="probe")
            self._probe_finish("stopped", "Probe stopped. Results saved." + self._probe_hint(p))
        else:
            self._say(f"{spoken} Next, {p['steps'][p['index']].label}.", origin="probe")

    def _probe_finish(self, reason, spoken):
        p, self._probe = self._probe, None
        if p is None:
            return
        steps = p["steps"]
        resume = None
        if reason != "complete" and p["index"] < len(steps):
            resume = steps[p["index"]].role
        path = self._probe_save(p, resume=resume, end=reason)
        record = probe.load_record(path)
        self.tel.event(
            "probe_end",
            reason=reason,
            path=path,
            steps=p["results"],
            counts=probe.counts(record, p["where"]),
            resume=resume,
            message=spoken,
        )
        self.profile = resolve(self.profile.info, self._has_command, probe.load_results(path))
        self._say(spoken, origin="probe")

    def _expect(self, label, read, expected, delay=None, tolerance=None, fail=None):
        """After a command, check the sim actually changed; say so if it didn't
        (in-sim the A330 ignored "autopilot on" while Vimaan had said "on").
        ``fail``: what to say instead of "<label> did not respond"."""
        self._verifications.append(
            {
                "cmd": self._current_cmd,
                "label": label,
                "read": read,
                "expected": expected,
                "tolerance": tolerance,
                "fail": fail,
                "due": time.perf_counter() + (VERIFY_DELAY_SEC if delay is None else delay),
            }
        )

    def _service_verifications(self):
        now = time.perf_counter()
        due = [v for v in self._verifications if v["due"] <= now]
        self._verifications = [v for v in self._verifications if v["due"] > now]
        for v in due:
            actual = v["read"]()
            if v.get("tolerance") is not None and actual is not None:
                ok = abs(actual - v["expected"]) <= v["tolerance"]
            else:
                ok = actual == v["expected"]
            self.tel.event(
                "verify",
                logging.INFO if ok else logging.WARNING,
                cmd=v["cmd"],
                label=v["label"],
                expected=v["expected"],
                actual=actual,
                ok=ok,
            )
            if not ok:
                self._say(v.get("fail") or f"{v['label'].replace('_', ' ')} did not respond")

    def _not_applicable(self, intent, slots=None):
        """Why this aircraft can't perform ``intent``, or None if it can."""
        if intent == "toggle_landing_gear" and self._read_int(DREF_GEAR_RETRACTABLE) == 0:
            return "This aircraft has fixed gear"
        if intent == "toggle_engine":
            instance = (slots or {}).get("instance", "1")
            num_engines = self._read_int(DREF_NUM_ENGINES)
            if instance == "both" and num_engines is not None and num_engines < 2:
                return "This aircraft has one engine"
            eng = ENGINE_BINDINGS.get(instance)
            if eng is not None and num_engines is not None and eng["index"] >= num_engines:
                return f"This aircraft has no {eng['label'].lower()}"
        return None

    def _read_safety_context(self, slots=None):
        """Sample flight state for the safety interlocks. Missing datarefs stay
        None so the interlocks fail safe (treat the situation as risky)."""
        agl_m = self._read_float("sim/flightmodel/position/y_agl")
        gs_ms = self._read_float("sim/flightmodel/position/groundspeed")
        ias = self._read_float("sim/cockpit2/gauges/indicators/airspeed_kts_pilot")
        on_ground_i = self._read_int("sim/flightmodel/failures/onground_any")
        return SafetyContext(
            agl_ft=None if agl_m is None else agl_m * 3.28084,
            ground_speed_kt=None if gs_ms is None else gs_ms * 1.94384,
            ias_kt=ias,
            on_ground=None if on_ground_i is None else bool(on_ground_i),
            vs_fpm=self._read_float(DREF_VERTICAL_SPEED_FPM),
            # The limit for the flap setting asked for, when one is.
            vfe_kt=self._flap_limit_for(slots)
            if slots and (slots.get("position") == "full" or slots.get("flap_setting"))
            else self.profile.info.vfe_kt or None,
        )

    # ------------------------------------------------------------------
    # Intent handlers: take (slots, intent), return an outcome string
    # ("executed", "no_op", "rejected", "failed", "pending").
    # ------------------------------------------------------------------
    def _set_heading(self, slots, intent):
        degrees = slots.get("degrees")
        if not degrees:
            self._say("Heading value missing")
            return "rejected"
        # Validators enforce the documented ranges before we touch a dataref,
        # so a mis-decoded number can't be written to the sim.
        value, ok, reason = validate_slot("degrees", degrees)
        if not ok:
            self.tel.event(
                "slot_rejected", logging.WARNING, slot="degrees", value=degrees, reason=reason
            )
            self._say("Invalid heading value")
            return "rejected"
        v = float(value)
        if not self._write(DREF_HEADING, v):
            self._say("Cannot set heading")
            return "failed"
        self._hold_check("Heading", DREF_HEADING, v, modulo=360)
        self._say(f"Heading {spell_digits(f'{int(v):03d}')}")
        return "executed"

    def _set_altitude(self, slots, intent):
        altitude = slots.get("altitude")
        if not altitude:
            self._say("Altitude value missing")
            return "rejected"
        value, ok, reason = validate_slot("altitude", altitude)
        if not ok:
            self.tel.event(
                "slot_rejected", logging.WARNING, slot="altitude", value=altitude, reason=reason
            )
            self._say("Altitude out of range")
            return "rejected"
        v = float(value)
        if not self._write(DREF_ALTITUDE, v):
            self._say("Cannot set altitude")
            return "failed"
        self._hold_check("Altitude", DREF_ALTITUDE, v)
        extra = ""
        if self._capture is not None:
            self._end_capture(False, "new target", speak=False)
        if (
            self._read_int(DREF_SERVOS_ON) == 1
            and self._read_int(DREF_VERTICAL_MODE) in VERTICAL_MOVING_MODES
        ):
            vs = self._read_float(DREF_VERTICAL_SPEED_FPM)
            if vs is not None and abs(vs) >= 100:
                extra = self._plan_capture(vs).replace(f", to {int(v)}", "")
        self._say(f"Altitude {int(v)} feet{extra}")
        return "executed"

    def _set_flight_level(self, slots, intent):
        flight_level = slots.get("flight_level")
        if not flight_level:
            self._say("Flight level missing")
            return "rejected"
        value, ok, reason = validate_slot("flight_level", flight_level)
        if not ok:
            self.tel.event(
                "slot_rejected",
                logging.WARNING,
                slot="flight_level",
                value=flight_level,
                reason=reason,
            )
            self._say("Flight level out of range")
            return "rejected"
        if not self._write(DREF_ALTITUDE, float(value) * 100):
            self._say("Cannot set flight level")
            return "failed"
        self._hold_check("Flight level", DREF_ALTITUDE, float(value) * 100)
        self._say(f"Flight level {spell_digits(f'{int(float(value)):03d}')}")
        return "executed"

    # ------------------------------------------------------------------
    # Tier 1: speed, vertical speed, modes, autothrottle, altimeter
    # ------------------------------------------------------------------
    def _implausible_reading(self, intent, as_to):
        """A "to" reading this aircraft can't fly, so there's nothing to ask."""
        vso = self.profile.info.vso_kt
        if intent != "set_autopilot_speed" or vso is None or vso <= 0:
            return False
        return float(as_to) < vso

    def _set_autopilot_speed(self, slots, intent):
        info = self.profile.info
        if slots.get("mach"):
            value, ok, _ = validate_slot("mach", slots["mach"])
            want_mach = 1
            if ok and info.mmo and float(value) > info.mmo:
                self._say(f"Mach {value} is above this aircraft's limit")
                return "rejected"
        elif slots.get("speed"):
            value, ok, _ = validate_slot("speed", slots["speed"])
            want_mach = 0
            if ok and info.vso_kt and float(value) < info.vso_kt:
                self._say(f"Speed {int(float(value))} is below this aircraft's stall speed")
                return "rejected"
            if ok and info.vne_kt and float(value) > info.vne_kt:
                self._say(f"Speed {int(float(value))} is above this aircraft's never exceed speed")
                return "rejected"
        else:
            self._say("Speed value missing")
            return "rejected"
        if not ok:
            self.tel.event("slot_rejected", logging.WARNING, slot="speed", value=slots)
            self._say("Speed out of range")
            return "rejected"
        target = float(value)
        is_mach = self._read_int(DREF_SPEED_IS_MACH)
        if is_mach is not None and is_mach != want_mach:
            toggle = self.profile.command("knots_mach_toggle")
            if not (toggle and self._fire_command(toggle)):
                self._say("Cannot switch between knots and Mach")
                return "failed"
        if not self._write(DREF_SPEED_DIAL, target):
            self._say("Cannot set speed")
            return "failed"
        select = self.profile.command("speed_select")
        if select:
            self._fire_command(select)  # Airbus: pull for a selected speed
        self._hold_check("Speed", DREF_SPEED_DIAL, target, tolerance=0.005 if want_mach else 0.5)
        if want_mach:
            spoken = f"Mach point {spell_digits(f'{target:.2f}'.split('.')[1])}"
        else:
            spoken = f"Speed {spell_digits(str(int(target)))}"
        self._say(spoken)
        return "executed"

    def _set_vertical_speed(self, slots, intent):
        value, ok, _ = validate_slot("vs", slots.get("vs"))
        if not ok:
            self._say("Vertical speed out of range")
            return "rejected"
        direction = slots.get("direction")
        if direction not in ("up", "down"):
            # Unsaid: towards the selected altitude, when that is clear.
            target_alt = self._read_float(DREF_ALTITUDE)
            actual = self._read_float(DREF_ALTITUDE_ACTUAL)
            if target_alt is None or actual is None or abs(target_alt - actual) < 100:
                self._say("Climb or descend?")
                return "rejected"
            direction = "up" if target_alt > actual else "down"
        if self.profile.supports("mode_vertical_speed") is False:
            self._say("This aircraft has no vertical speed mode")
            return "rejected"
        fpm = float(value) if direction == "up" else -float(value)
        # Engage first: the generic V/S command captures the *current* rate,
        # so the dial is written after it.
        if self._read_int(DREF_VVI_STATUS) != 2:
            mode = self.profile.command("mode_vertical_speed")
            if not (mode and self._fire_command(mode)):
                self._say("Cannot engage vertical speed")
                return "failed"
            self._expect("Vertical speed mode", lambda: self._read_int(DREF_VVI_STATUS), 2)
        if not self._write(DREF_VVI_DIAL, fpm):
            self._say("Cannot set vertical speed")
            return "failed"
        select = self.profile.command("vertical_select")
        if select:
            self._fire_command(select)
        self._hold_check("Vertical speed", DREF_VVI_DIAL, fpm, tolerance=50.0)
        verb = "Climb" if fpm > 0 else "Descend"
        self._say(f"{verb} {int(abs(fpm))} feet per minute{self._plan_capture(fpm)}")
        return "executed"

    # ------------------------------------------------------------------
    # Altitude capture (Tier 1, B6)
    # ------------------------------------------------------------------
    def _plan_capture(self, fpm):
        """After a climb or descent starts: level off at the selected altitude.
        Returns what to add to the read-back.

        1. The aircraft's own capture (altitude arm), unless the probe said it
           does nothing. Armed after a short delay: the 737 arms it by itself
           when V/S engages, and the generic arm command is a toggle.
        2. Else Vimaan's own capture, if switched on (option b).
        3. Else say how to level off (option a)."""
        target = self._read_float(DREF_ALTITUDE)
        actual = self._read_float(DREF_ALTITUDE_ACTUAL)
        if target is None or actual is None or (target - actual) * fpm <= 0:
            return ""  # the selected altitude isn't ahead: nothing to capture
        if abs(target - actual) < 100:
            return ""
        arm = self.profile.command("mode_altitude_arm")
        if (
            self.profile.supports("mode_altitude_arm") is not False
            and arm
            and self._has_command(arm)
        ):
            self._later(ARM_CAPTURE_DELAY_SEC, self._arm_capture)
            return f", to {int(target)}"
        if self._capture_enabled:
            self._start_capture(target, fpm)
            return f". Leveling off at {int(target)}"
        return f". This aircraft can't capture altitude. Say altitude hold at {int(target)}"

    def _arm_capture(self):
        if self._read_int(DREF_ALT_ARMED) == 1:
            return  # armed by the aircraft (737) or already
        if self._read_int(DREF_VERTICAL_MODE) not in VERTICAL_MOVING_MODES:
            return  # the pilot changed mode in the meantime
        arm = self.profile.command("mode_altitude_arm")
        if arm and self._fire_command(arm):
            self._expect(
                "Altitude capture",
                lambda: self._read_int(DREF_ALT_ARMED),
                1,
                fail="Altitude capture did not arm. Say altitude hold at the target.",
            )

    def _start_capture(self, target, fpm):
        self._capture = {
            "cmd": self._current_cmd,
            "target": target,
            "up": fpm > 0,
            "mode": self._read_int(DREF_VERTICAL_MODE),
            "deadline": time.perf_counter() + CAPTURE_MAX_SEC,
        }
        self.tel.event("capture_start", cmd=self._current_cmd, target=target, fpm=fpm)

    def _service_capture(self):
        c = self._capture
        if c is None:
            return
        alt = self._read_float(DREF_ALTITUDE_ACTUAL)
        vs = self._read_float(DREF_VERTICAL_SPEED_FPM)
        dial = self._read_float(DREF_ALTITUDE)
        why = None
        if self._read_int(DREF_SERVOS_ON) != 1:
            why = "autopilot off"
        elif dial is None or abs(dial - c["target"]) > 50:
            why = "target moved"
        elif self._read_int(DREF_VERTICAL_MODE) != c["mode"]:
            why = "mode changed"
        elif time.perf_counter() > c["deadline"]:
            why = "timed out"
        if why:
            return self._end_capture(False, why)
        if alt is None or vs is None:
            return
        remaining = (c["target"] - alt) if c["up"] else (alt - c["target"])
        band = max(CAPTURE_MIN_BAND_FT, abs(vs) * CAPTURE_LEAD_SEC / 60.0)
        if remaining <= band:
            hold = self.profile.command("mode_altitude_hold")
            if hold and self._fire_command(hold):
                self._expect("Altitude hold", lambda: self._read_int(DREF_ALT_HOLD_STATUS), 2)
            self._end_capture(True, "captured", alt)

    def _end_capture(self, ok, why, alt=None, speak=True):
        c, self._capture = self._capture, None
        if c is None:
            return
        self.tel.event(
            "capture_end",
            logging.INFO if ok else logging.WARNING,
            cmd=c["cmd"],
            ok=ok,
            why=why,
            target=c["target"],
            altitude=None if alt is None else round(alt),
        )
        if speak:
            self._say(
                f"Altitude hold, {int(c['target'])}" if ok else f"Altitude capture cancelled, {why}"
            )

    def _on_capture_cmd(self, cmd_ref, phase, refcon):
        if phase == xp.CommandBegin:
            self._capture_enabled = not self._capture_enabled
            if not self._capture_enabled and self._capture is not None:
                self._end_capture(False, "switched off", speak=False)
            self.tel.event("capture_setting", enabled=self._capture_enabled)
            self._say(f"Vimaan altitude capture {'on' if self._capture_enabled else 'off'}")
        return 1

    def _later(self, delay, action):
        self._deferred.append(
            {"due": time.perf_counter() + delay, "action": action, "cmd": self._current_cmd}
        )

    def _service_deferred(self):
        now = time.perf_counter()
        due = [d for d in self._deferred if d["due"] <= now]
        if not due:
            return
        self._deferred = [d for d in self._deferred if d["due"] > now]
        for d in due:
            previous, self._current_cmd = self._current_cmd, d["cmd"]
            try:
                d["action"]()
            except Exception as exc:
                self.tel.error("deferred_error", exc, cmd=d["cmd"])
            finally:
                self._current_cmd = previous

    def _set_autopilot_mode(self, slots, intent):
        mode = slots.get("mode")
        if mode not in AUTOPILOT_MODES:
            self._say("Which mode?")
            return "rejected"
        state = slots.get("state") or "on"
        role, status_dref = AUTOPILOT_MODES[mode]
        status_dref = self.profile.status_dataref(role, status_dref)
        name = MODE_SPOKEN.get(mode, mode.capitalize())
        command = self.profile.command(role)
        if not command or not self._has_command(command) or self.profile.supports(role) is False:
            self._say(f"{name} mode is not available on this aircraft")
            return "rejected"
        status = self._read_int(status_dref)
        if status is None and state == "off":
            self._say(f"Cannot read {name} mode state")  # a blind press could turn it on
            return "failed"
        on_at = 2 if mode in ACTIVE_ONLY_MODES else 1
        if status is not None and (status >= on_at) == (state == "on"):
            self._say(f"{name} already {state}")
            return "no_op"
        if not self._fire_command(command):
            self._say(f"Cannot select {name}")
            return "failed"
        want_on = state == "on"
        # The FMS modes engage only with a flight plan: the 737's LNAV and
        # VNAV buttons did nothing in the probe without one.
        fms = mode in FMS_MODES and (
            role in self.profile.overridden or self.profile.info.autopilot_type == 1
        )
        self._expect(
            f"{name} mode",
            lambda: None if (r := self._read_int(status_dref)) is None else r >= on_at,
            want_on,
            fail=f"{name} did not engage. It needs a flight plan in the F M C."
            if fms and want_on
            else None,
        )
        self._say(f"{name} {'on' if want_on else 'off'}")
        return "executed"

    def _toggle_autothrottle(self, slots, intent):
        state = slots.get("state")
        if state not in ("on", "off"):
            self._say("Autothrottle on or off?")
            return "rejected"
        own = "autothrottle_toggle" in self.profile.overridden
        # X-Plane's own autothrottle runs on any aircraft (the Baron and King
        # Air probes said "works"); only an aircraft whose files use one has it.
        if not own and (
            self._autothrottle_shown() is False
            or self.profile.supports("autothrottle_toggle") is False
        ):
            self._say("This aircraft has no autothrottle")
            return "rejected"
        # autothrottle_enabled: -1 off, 0 armed, 1-3 holding. The A330's A/THR
        # is on when armed; elsewhere on means holding a speed.
        on_from = self.profile.settings.get("athr_on_from", 1)
        hint = self.profile.settings.get("athr_hint")
        raw = self._read_int(DREF_AUTOTHROTTLE)
        current = None if raw is None else raw >= on_from
        want_on = state == "on"
        if current is not None and current == want_on:
            self._say(f"Autothrottle already {state}")
            return "no_op"
        command = self.profile.command("autothrottle_on" if want_on else "autothrottle_off")
        if not (command and self._has_command(command)):
            if current is None:
                self._say("Cannot read autothrottle state")
                return "failed"
            command = self.profile.command("autothrottle_toggle")
        arm = self.profile.command("autothrottle_arm")
        hard_off = raw is not None and raw < 0 and on_from >= 1  # (the A330 arms at 0)
        if want_on and hard_off and self._has_command(arm):
            # Hard off (-1: the 737's A/T ARM switch off): arm it, then engage
            # once the aircraft has taken the switch (the next frames).
            if not self._fire_command(arm):
                self._say("Cannot arm the autothrottle")
                return "failed"
            self._expect(
                "Autothrottle armed",
                lambda: None if (r := self._read_int(DREF_AUTOTHROTTLE)) is None else r >= 0,
                True,
            )
            self._say("Arming the autothrottle")
            self._later(
                ATHR_ARM_SETTLE_SEC, lambda: self._engage_autothrottle(command, on_from, hint)
            )
            return "executed"
        if not (command and self._fire_command(command)):
            self._say("No autothrottle on this aircraft")
            return "rejected"
        self._expect_autothrottle(want_on, on_from, hint)
        self._say(f"Autothrottle {state}")
        return "executed"

    def _engage_autothrottle(self, command, on_from, hint):
        if not (command and self._fire_command(command)):
            self._say("No autothrottle on this aircraft")
            return
        self._expect_autothrottle(True, on_from, hint)
        self._say("Autothrottle on")

    def _expect_autothrottle(self, want_on, on_from, hint):
        self._expect(
            "Autothrottle",
            lambda: None if (r := self._read_int(DREF_AUTOTHROTTLE)) is None else r >= on_from,
            want_on,
            fail=f"Autothrottle did not respond. {hint}." if hint else None,
        )

    def _set_altimeter(self, slots, intent):
        if not slots.get("setting") and not slots.get("pressure"):
            self._say("Altimeter setting?")
            return "rejected"
        if slots.get("setting") == "standard":
            command = self.profile.command("baro_std")
            if not (command and self._fire_command(command)):
                self._say("Cannot set standard")
                return "failed"
            self._say("Altimeter standard")
            return "executed"
        value, ok, _ = validate_slot("pressure", slots.get("pressure"))
        if not ok:
            self._say("Altimeter setting out of range")
            return "rejected"
        number = float(value)
        if 940 <= number <= 1060:
            inhg, spoken = number * HPA_TO_INHG, f"Q N H {spell_digits(str(int(number)))}"
        else:
            inhg = number / 100 if number >= 2800 else number
            spoken = f"Altimeter {spell_digits(f'{inhg:.2f}')}"
        spoken = spoken.replace(" decimal ", " ")
        if self.profile.probed.get("dial_baro") == "no_change":
            # The probe measured that this aircraft puts a written setting
            # back (the 737's scripts own it): turn its knob instead.
            return self._start_baro_steps(inhg, spoken)
        if not all(self._write(name, inhg) for name in DREF_BARO):
            self._say("Cannot set altimeter")
            return "failed"
        self._hold_check("Altimeter", DREF_BARO[0], inhg, tolerance=0.005)
        self._say(spoken)
        return "executed"

    def _start_baro_steps(self, inhg, spoken):
        up, down = self.profile.command("baro_up"), self.profile.command("baro_down")
        if not (up and down and self._has_command(up) and self._has_command(down)):
            self._say("Cannot set the altimeter on this aircraft")
            return "rejected"
        self._baro_step = {
            "cmd": self._current_cmd,
            "target": inhg,
            "spoken": spoken,
            "up": up,
            "down": down,
            "presses": 0,
            "last": None,  # reading before the last press
            "last_dir": 0,
            "stuck": 0,
            "finishing": False,
            "deadline": time.perf_counter() + BARO_STEP_MAX_SEC,
        }
        self.tel.event("baro_step_start", cmd=self._current_cmd, target=round(inhg, 4))
        return "pending"

    def _service_baro_step(self):
        """One knob press per frame toward the target, reading back between
        presses. A step can be bigger than the tolerance (an altimeter in
        hPa moves 0.03 inHg a click): on crossing the target, keep whichever
        side is closer."""
        j = self._baro_step
        if j is None:
            return
        now = self._read_float(DREF_BARO[0])
        if now is None:
            return self._end_baro_steps(False, None, "cannot read the altimeter")
        err = j["target"] - now
        if abs(err) <= BARO_TOLERANCE_INHG or j["finishing"]:
            return self._end_baro_steps(True, now)
        direction = 1 if err > 0 else -1
        if j["last"] is not None:
            j["stuck"] = j["stuck"] + 1 if abs(now - j["last"]) < 1e-6 else 0
            if j["stuck"] >= 6:
                return self._end_baro_steps(False, now, "the knob did not move it")
            if j["last_dir"] and direction != j["last_dir"]:
                if abs(j["target"] - j["last"]) < abs(err):
                    # The previous click was closer: go back one.
                    self._fire_command(j["up"] if direction > 0 else j["down"])
                    j["finishing"] = True
                    return
                return self._end_baro_steps(True, now)
        if j["presses"] >= BARO_STEP_MAX_PRESSES or time.perf_counter() > j["deadline"]:
            return self._end_baro_steps(False, now, "it took too many steps")
        self._fire_command(j["up"] if direction > 0 else j["down"])
        j.update(presses=j["presses"] + 1, last=now, last_dir=direction)

    def _end_baro_steps(self, ok, reading, why=None):
        j, self._baro_step = self._baro_step, None
        assert j is not None  # only called while stepping
        self.tel.event(
            "baro_step_end",
            logging.INFO if ok else logging.WARNING,
            cmd=j["cmd"],
            ok=ok,
            target=round(j["target"], 4),
            reading=None if reading is None else round(reading, 4),
            presses=j["presses"],
            why=why,
        )
        # The reply belongs to the command that started the steps (the test
        # bridge and the Flights tab group by it).
        previous, self._current_cmd = self._current_cmd, j["cmd"]
        try:
            if ok:
                self._write(DREF_BARO[1], reading)  # copilot side, where the aircraft keeps it
                self._say(j["spoken"])
            else:
                shown = "" if reading is None else f" It reads {spell_digits(f'{reading:.2f}')}."
                self._say(f"Altimeter not set: {why}.{shown}".replace(" decimal ", " "))
        finally:
            self._current_cmd = previous

    def _frequency(self, slots, slot):
        """The validated frequency in a slot, or None after saying why."""
        frequency = slots.get(slot)
        if not frequency:
            self._say("Frequency missing")
            return None
        fval, ok, reason = validate_slot(slot, frequency)
        if not ok:
            self.tel.event(
                "slot_rejected", logging.WARNING, slot=slot, value=frequency, reason=reason
            )
            self._say("Frequency out of range")
            return None
        return fval

    def _radio(self, slots, slot, table, kind):
        """The radio a slot names ("1" when unsaid), or None after saying
        which ones there are."""
        port = str(slots.get(slot) or "1")
        if port not in table:
            self._say(f"{kind} {port} is not supported. Use {kind} 1 or 2.")
            return None
        return port

    def _tune(self, label, name, fval, per_mhz, spoken):
        units = int(round(float(fval) * per_mhz))
        if not self._write(name, units, is_float=False):
            self._say(f"Cannot tune {label}")
            return "failed"
        self._hold_check(label, name, units, is_float=False)
        self._say(f"{spoken} {spell_digits(fval)}")
        return "executed"

    def _set_com_frequency(self, slots, intent):
        fval = self._frequency(slots, "frequency")
        port = None if fval is None else self._radio(slots, "com_port", DREF_COM_KHZ, "COM")
        if port is None:
            return "rejected"
        return self._tune(f"COM {port}", DREF_COM_KHZ[port], fval, 1000, f"COM {port} set to")

    def _set_com_standby(self, slots, intent):
        fval = self._frequency(slots, "frequency")
        port = None if fval is None else self._radio(slots, "com_port", DREF_COM_STANDBY_KHZ, "COM")
        if port is None:
            return "rejected"
        label = f"COM {port} standby"
        return self._tune(label, DREF_COM_STANDBY_KHZ[port], fval, 1000, label)

    def _set_nav_frequency(self, slots, intent):
        fval = self._frequency(slots, "nav_frequency")
        port = None if fval is None else self._radio(slots, "nav_port", DREF_NAV_10KHZ, "NAV")
        if port is None:
            return "rejected"
        return self._tune(f"NAV {port}", DREF_NAV_10KHZ[port], fval, 100, f"NAV {port} set to")

    def _set_nav_standby(self, slots, intent):
        fval = self._frequency(slots, "nav_frequency")
        port = (
            None if fval is None else self._radio(slots, "nav_port", DREF_NAV_STANDBY_10KHZ, "NAV")
        )
        if port is None:
            return "rejected"
        label = f"NAV {port} standby"
        return self._tune(label, DREF_NAV_STANDBY_10KHZ[port], fval, 100, label)

    def _swap_com_frequency(self, slots, intent):
        port = self._radio(slots, "com_port", CMD_COM_FLIP, "COM")
        if port is None:
            return "rejected"
        return self._flip(
            f"COM {port}", CMD_COM_FLIP[port], DREF_COM_KHZ[port], DREF_COM_STANDBY_KHZ[port], 1000
        )

    def _swap_nav_frequency(self, slots, intent):
        port = self._radio(slots, "nav_port", CMD_NAV_FLIP, "NAV")
        if port is None:
            return "rejected"
        return self._flip(
            f"NAV {port}",
            CMD_NAV_FLIP[port],
            DREF_NAV_10KHZ[port],
            DREF_NAV_STANDBY_10KHZ[port],
            100,
        )

    def _flip(self, label, command, active, standby, per_mhz):
        """Press the transfer button; the active frequency should become what
        standby was (the read-back says the new active one)."""
        was_standby = self._read_int(standby)
        if not self._fire_command(command):
            self._say(f"Cannot flip {label}")
            return "failed"
        if was_standby is None:
            self._say(f"{label} flipped")
            return "executed"
        self._expect(f"{label} active", lambda: self._read_int(active), was_standby)
        self._say(f"{label} active {spell_digits(mhz_text(was_standby, per_mhz))}")
        return "executed"

    def _set_course(self, slots, intent):
        degrees = slots.get("degrees")
        if not degrees:
            self._say("Course value missing")
            return "rejected"
        value, ok, reason = validate_slot("degrees", degrees)
        if not ok:
            self.tel.event(
                "slot_rejected", logging.WARNING, slot="degrees", value=degrees, reason=reason
            )
            self._say("Invalid course value")
            return "rejected"
        port = self._radio(slots, "nav_port", DREF_OBS, "NAV")
        if port is None:
            return "rejected"
        v = float(value)
        if not self._write(DREF_OBS[port], v):
            self._say("Cannot set course")
            return "failed"
        self._hold_check(f"Course {port}", DREF_OBS[port], v, modulo=360)
        # Always name the radio, like the frequency read-backs: speech that
        # loses "nav two" still sets NAV 1, and the pilot must hear that.
        self._say(f"NAV {port} course {spell_digits(f'{int(v):03d}')}")
        return "executed"

    def _set_transponder_code(self, slots, intent):
        code = slots.get("squawk")
        if not code:
            self._say("Squawk code missing")
            return "rejected"
        value, ok, reason = validate_slot("squawk", code)
        if not ok:
            self.tel.event(
                "slot_rejected", logging.WARNING, slot="squawk", value=code, reason=reason
            )
            self._say("Invalid squawk. Four digits, zero to seven.")
            return "rejected"
        number = int(value)  # "0120" -> 120: X-Plane keeps the digits as a number
        if not self._write(DREF_XPDR_CODE, number, is_float=False):
            self._say("Cannot set the transponder")
            return "failed"
        self._hold_check("Squawk", DREF_XPDR_CODE, number, is_float=False)
        self._say(f"Squawk {spell_digits(value)}")
        return "executed"

    def _set_transponder_mode(self, slots, intent):
        value, ok, _ = validate_slot("xpdr_mode", slots.get("xpdr_mode") or "")
        if not ok:
            self._say("Transponder off, standby, on, or altitude?")
            return "rejected"
        if value == "ident":
            return self._transponder_ident()
        knobs = self.profile.settings.get("xpdr_knobs")
        if knobs:
            return self._transponder_knobs(value, knobs)
        if self._read_int(DREF_XPDR_MODE) in XPDR_ALREADY[value]:
            self._say(XPDR_ALREADY_SPOKEN[value])
            return "no_op"
        if not self._fire_command(CMD_XPDR[value]):
            self._say("Cannot set the transponder")
            return "failed"
        self._expect("Transponder", lambda: self._read_int(DREF_XPDR_MODE), XPDR_SETS[value])
        self._say(f"Transponder {XPDR_SPOKEN[value]}")
        return "executed"

    def _transponder_ident(self):
        if self._read_int(DREF_XPDR_MODE) in (0, 1):
            self._say("Transponder is in standby. Say transponder on first.")
            return "rejected"
        command = self.profile.command("xpdr_ident") or CMD_XPDR["ident"]
        if not self._fire_command(command):
            self._say("Cannot ident")
            return "failed"
        self._expect(
            "Ident", lambda: self._read_int(DREF_XPDR_IDENT), 1, fail="Ident did not respond"
        )
        self._say("Ident")
        return "executed"

    def _transponder_knobs(self, value, knobs):
        """Turn the aircraft's own selectors (737, A330), one click a frame."""
        steps = knobs.get(value)
        if not steps:
            self._say("This transponder has no off. Say standby.")
            return "rejected"
        if all(self._knob_at(k) for k in steps):
            self._say(XPDR_ALREADY_SPOKEN[value])
            return "no_op"
        if not all(self._has_command(n) for k in steps for n in (k.down, k.up)):
            self._say("Cannot set the transponder on this aircraft")
            return "rejected"
        self._knob_step = {
            "cmd": self._current_cmd,
            "label": "Transponder",
            "knobs": tuple(steps),
            "index": 0,
            "presses": 0,
            "waiting": None,  # position before the last click, until it moves
            "stuck": 0,
            "spoken": f"Transponder {XPDR_SPOKEN[value]}",
            "deadline": time.perf_counter() + KNOB_STEP_MAX_SEC,
        }
        self.tel.event("knob_step_start", cmd=self._current_cmd, mode=value, knobs=len(steps))
        return "pending"

    def _knob_position(self, knob):
        value = self._read_float(knob.position)
        return None if value is None else int(round(value))

    def _knob_at(self, knob):
        position = self._knob_position(knob)
        return position is not None and knob.low <= position <= knob.high

    def _service_knob_step(self):
        """One click, then wait for the selector to move before the next, so
        a click the aircraft handles a frame late can't overshoot."""
        j = self._knob_step
        if j is None:
            return
        knob = j["knobs"][j["index"]]
        position = self._knob_position(knob)
        if position is None:
            return self._end_knob_steps(False, "cannot read the selector")
        if j["waiting"] is not None and position == j["waiting"]:
            j["stuck"] += 1
            if j["stuck"] >= KNOB_STUCK_FRAMES:
                return self._end_knob_steps(False, "the selector did not move")
            return
        j.update(waiting=None, stuck=0)
        if knob.low <= position <= knob.high:
            j["index"] += 1
            if j["index"] == len(j["knobs"]):
                return self._end_knob_steps(True)
            return
        if j["presses"] >= KNOB_STEP_MAX_PRESSES or time.perf_counter() > j["deadline"]:
            return self._end_knob_steps(False, "it took too many clicks")
        self._fire_command(knob.up if position < knob.low else knob.down)
        j.update(presses=j["presses"] + 1, waiting=position)

    def _end_knob_steps(self, ok, why=None):
        j, self._knob_step = self._knob_step, None
        assert j is not None  # only called while stepping
        previous, self._current_cmd = self._current_cmd, j["cmd"]
        try:
            self.tel.event(
                "knob_step_end",
                logging.INFO if ok else logging.WARNING,
                cmd=j["cmd"],
                ok=ok,
                presses=j["presses"],
                positions=[self._knob_position(k) for k in j["knobs"]],
                sim_mode=self._read_int(DREF_XPDR_MODE),
                why=why,
            )
            if ok:
                knobs = j["knobs"]
                self._expect(j["label"], lambda: all(self._knob_at(k) for k in knobs), True)
                self._say(j["spoken"])
            else:
                self._say(f"{j['label']} not set: {why}.")
        finally:
            self._current_cmd = previous

    def _toggle_landing_gear(self, slots, intent):
        state = slots.get("state")
        if state not in ("up", "down"):
            self._say("Gear up or down?")
            return "rejected"
        target = 1 if state == "down" else 0
        if self._read_int(DREF_GEAR_HANDLE) == target:
            self._say(f"Gear already {state}")
            return "no_op"
        if not self._fire_command(f"sim/flight_controls/landing_gear_{state}"):
            self._say("Gear command unavailable")
            return "failed"
        self._expect("gear_handle", lambda: self._read_int(DREF_GEAR_HANDLE), target)
        self._say(f"Gear {state}")
        return "executed"

    def _set_flaps(self, slots, intent):
        state = slots.get("position")
        before = self._read_float(DREF_FLAP_HANDLE)
        if state == "full" or (slots.get("flap_setting") and state not in ("up", "down")):
            return self._set_flap_notch(slots, before)
        if state == "up":
            # "Flaps up" in a cockpit means fully retracted, not one notch.
            if before is not None and before <= 0.01:
                self._say("Flaps already up")
                return "no_op"
            command, spoken = "sim/flight_controls/flaps_up_full", "Flaps up"
        elif state == "down":
            if before is not None and before >= 0.99:
                self._say("Flaps already fully down")
                return "no_op"
            command, spoken = "sim/flight_controls/flaps_down", "Flaps down one notch"
        else:
            self._say("Flaps up or down?")
            return "rejected"
        if not self._fire_command(command):
            self._say("Flaps command unavailable")
            return "failed"
        self.tel.event("flaps", cmd=self._current_cmd, handle_before=before)
        self._say(spoken)
        return "executed"

    def _set_flap_notch(self, slots, before):
        """ "Flaps full" or "flaps fifteen": a handle position by its label."""
        detents = self.profile.info.flap_detents
        if not detents:
            self._say("Cannot read this aircraft's flap settings")
            return "failed"
        notch: int | None
        if slots.get("position") == "full":
            notch, spoken = detents, "Flaps full"
        else:
            value, ok, _ = validate_slot("flap_setting", slots.get("flap_setting"))
            notch = self.profile.flap_notch(int(float(value))) if ok else None
            if notch is None:
                self._say(f"Flaps {slots.get('flap_setting')} is not a setting on this aircraft")
                return "rejected"
            spoken = f"Flaps {int(float(value))}"
        ratio = notch / detents
        if before is not None and abs(before - ratio) < 0.01:
            self._say(f"{spoken} already set")
            return "no_op"
        if not self._write(DREF_FLAP_HANDLE, ratio):
            self._say("Flaps command unavailable")
            return "failed"
        self._hold_check("Flaps", DREF_FLAP_HANDLE, ratio, tolerance=0.01)
        self.tel.event(
            "flaps", cmd=self._current_cmd, handle_before=before, notch=notch, of=detents
        )
        self._say(spoken)
        return "executed"

    def _toggle_parking_brake(self, slots, intent):
        state = slots.get("state")
        if state not in ("on", "off"):
            self._say("Parking brake on or off?")
            return "rejected"
        ratio = self._read_float(DREF_PARKING_BRAKE)
        if ratio is not None and (ratio > 0.5) == (state == "on"):
            self._say(f"Parking brake already {state}")
            return "no_op"
        command = "sim/flight_controls/park_brake_" + ("set" if state == "on" else "release")
        if not self._fire_command(command):
            self._say("Parking brake command unavailable")
            return "failed"
        self._expect(
            "parking_brake",
            lambda: None if (r := self._read_float(DREF_PARKING_BRAKE)) is None else r > 0.5,
            state == "on",
        )
        self._say(f"Parking brake {state}")
        return "executed"

    def _toggle_binary(self, slots, intent):
        state = slots.get("state")
        if state not in ("on", "off"):
            self._say("On or off?")
            return "rejected"
        instance = slots.get("instance", "1")
        binding = TOGGLE_BINDINGS[intent].get(instance)
        if binding is None:
            self._say(f"There is no {BY_INTENT[intent].display.lower()} {instance}")
            return "rejected"
        return self._ensure_state(binding, state)

    def _ensure_state(self, binding, desired):
        """Move a binary system to ``desired`` ('on'/'off') idempotently."""
        label = binding["label"]
        target = 1 if desired == "on" else 0
        raw = self._read_int(binding["dataref"])
        current = None if raw is None else int(raw > 0)

        if current == target:
            self._say(f"{label} already {desired}")
            return "no_op"

        if binding.get("writable"):
            if self._write(binding["dataref"], target, is_float=False):
                self._say(f"{label} {desired}")
                return "executed"

        command = binding["on_cmd"] if target else binding["off_cmd"]
        engage = self.profile.command(binding.get("engage_role") or "")
        if target and engage and self._has_command(engage):
            # The aircraft's own engage button keeps its logic in step.
            command = engage
        elif binding.get("prefer_toggle") and current is not None:
            # The state was just read and differs, so the toggle moves it the
            # right way, on aircraft that ignore the dedicated command too.
            command = binding["toggle"]
        if command is None:
            if current is None:
                # Without the state, a toggle could turn it ON. Refuse.
                self._say(f"Cannot read {label} state")
                return "failed"
            command = binding["toggle"]
        if not self._fire_command(command):
            self._say(f"Cannot {desired} {label}")
            return "failed"
        self._expect(
            label,
            lambda: None if (r := self._read_int(binding["dataref"])) is None else int(r > 0),
            target,
        )
        self._say(f"{label} {desired}")
        return "executed"

    # ------------------------------------------------------------------
    # Engines: the starter must be HELD until the engine catches, so a start
    # is a multi-frame job driven by the flight loop.
    # ------------------------------------------------------------------
    def _toggle_engine(self, slots, intent):
        state = slots.get("state")
        if state not in ("on", "off"):
            self._say("Start or shut down?")
            return "rejected"
        instance = slots.get("instance", "1")
        if instance != "both":
            return self._engine(ENGINE_BINDINGS[instance], state)
        num_engines = self._read_int(DREF_NUM_ENGINES) or len(ENGINE_BINDINGS)
        outcomes = [
            self._engine(eng, state)
            for eng in ENGINE_BINDINGS.values()
            if eng["index"] < num_engines
        ]
        for outcome in ("pending", "executed", "failed", "no_op"):
            if outcome in outcomes:
                return outcome
        return "rejected"

    def _engine(self, eng, state):
        """Start (held starter, driven by the flight loop) or shut down one engine."""
        label = eng["label"]
        running = self._read_array(DREF_ENGINE_RUNNING, eng["index"])

        if state == "off":
            if running == 0:
                self._say(f"{label} already off")
                return "no_op"
            if not self._fire_command(eng["shutdown"]):
                self._say(f"Cannot shut down {label}")
                return "failed"
            self._expect(
                label, lambda: self._read_array(DREF_ENGINE_RUNNING, eng["index"]), 0, delay=5.0
            )
            self._say(f"{label} shutting down")
            return "executed"

        if running == 1:
            self._say(f"{label} already running")
            return "no_op"
        if label in self._cranks:
            self._say(f"{label} already starting")
            return "no_op"
        kind = self.profile.engine_start
        if kind == "electric":
            self._say(f"{label} is electric and has no starter")
            return "rejected"
        cmd_ref = xp.findCommand(eng["starter"])
        if not cmd_ref:
            self.tel.event(
                "sim_command",
                logging.ERROR,
                cmd=self._current_cmd,
                command=eng["starter"],
                ok=False,
                reason="command_not_found",
            )
            self._say(f"Cannot start {label}")
            return "failed"
        xp.commandBegin(cmd_ref)
        turbine = kind in CORE_SPEED
        self._cranks[label] = {
            "cmd": self._current_cmd,
            "ref": cmd_ref,
            "started": time.perf_counter(),
            "kind": kind,
            "max_sec": TURBINE_MAX_CRANK_SEC if turbine else STARTER_MAX_CRANK_SEC,
            "fuel_on": not turbine,  # pistons start with the mixture as set
            "peak_pct": 0.0,
            **eng,
        }
        self.tel.event(
            "crank_begin",
            cmd=self._current_cmd,
            engine=label,
            start_type=kind,
            mixture=self._read_array(DREF_ENGINE_MIXTURE, eng["index"], True),
            fuel_lever=self._read_array(DREF_FUEL_LEVER, eng["index"], True),
            core_pct=self._core_pct(self._cranks[label]),
            ignition_key=self._read_array(DREF_IGNITION_KEY, eng["index"]),
            battery=self._read_array(DREF_BATTERY, 0),
        )
        self._say(f"Starting {label.lower()}")
        return "pending"

    def _core_pct(self, crank):
        name = CORE_SPEED.get(crank["kind"])
        return None if name is None else self._read_array(name, crank["index"], True)

    def _service_cranks(self):
        for label, crank in list(self._cranks.items()):
            core = self._core_pct(crank)
            if core is not None:
                crank["peak_pct"] = max(crank["peak_pct"], core)
                if not crank["fuel_on"] and core >= FUEL_ON_AT_PCT[crank["kind"]]:
                    self._fuel_on(crank, core)
            running = self._read_array(DREF_ENGINE_RUNNING, crank["index"]) == 1
            spooled = core is None or core >= STARTER_CUTOUT_PCT
            if running and spooled:
                self._end_crank(label, "running")
                self._say(f"{crank['label']} running")
            elif time.perf_counter() - crank["started"] > crank["max_sec"]:
                self._end_crank(label, "timeout")
                self._say(self._why_no_start(crank))

    def _fuel_on(self, crank, core):
        """Open the fuel once the core turns: the aircraft's own lever command
        if it has one (layer 3), else the sim's fuel lever (mixture ratio)."""
        crank["fuel_on"] = True
        index = crank["index"]
        before = self._read_array(DREF_FUEL_LEVER, index, True)
        role_cmd = self.profile.command(f"fuel_on_{index + 1}")
        if before is not None and before >= 0.5:
            method = "already_open"
        elif role_cmd and self._fire_command(role_cmd):
            method = role_cmd
        else:
            method = "write" if self._write_array(DREF_FUEL_LEVER, index, 1.0) else "failed"
        self.tel.event(
            "fuel_on",
            cmd=crank["cmd"],
            engine=crank["label"],
            core_pct=round(core, 1),
            method=method,
            lever_before=before,
        )

    @staticmethod
    def _why_no_start(crank):
        label = crank["label"]
        if crank["kind"] not in CORE_SPEED:
            return f"{label} did not start. Check fuel, mixture and ignition."
        core = "N2" if crank["kind"] == "jet" else "N1"
        if crank["peak_pct"] < FUEL_ON_AT_PCT[crank["kind"]]:
            return (
                f"{label} did not spool up. {core} reached {crank['peak_pct']:.0f} percent. "
                "Check bleed air and the engine start switch."
            )
        return (
            f"{label} did not light. {core} reached {crank['peak_pct']:.0f} percent. "
            "Check fuel and ignition."
        )

    def _write_array(self, name, index, value):
        """Write one element of a float[] dataref and read it back."""
        dref = self._find(name)
        if not dref:
            return False
        try:
            xp.setDatavf(dref, [float(value)], index, 1)
            readback = self._read_array(name, index, True)
        except Exception as exc:
            self.tel.event("dataref_write_error", logging.WARNING, dataref=name, error=repr(exc))
            return False
        ok = readback is not None and abs(readback - float(value)) < 0.01
        self.tel.event(
            "sim_write",
            logging.INFO if ok else logging.WARNING,
            cmd=self._current_cmd,
            dataref=f"{name}[{index}]",
            value=value,
            readback=readback,
            ok=ok,
        )
        return ok

    def _end_crank(self, intent, result):
        crank = self._cranks.pop(intent)
        xp.commandEnd(crank["ref"])
        self.tel.event(
            "crank_end",
            logging.INFO if result == "running" else logging.WARNING,
            cmd=crank["cmd"],
            engine=crank["label"],
            result=result,
            crank_sec=round(time.perf_counter() - crank["started"], 1),
            start_type=crank["kind"],
            peak_core_pct=round(crank["peak_pct"], 1),
            fuel_on=crank["fuel_on"],
            mixture=self._read_array(DREF_ENGINE_MIXTURE, crank["index"], True),
            ignition_key=self._read_array(DREF_IGNITION_KEY, crank["index"]),
        )
