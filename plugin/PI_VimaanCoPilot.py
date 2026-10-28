"""Vimaan AI Co-Pilot — X-Plane plugin entry point.

Hold-to-talk speech command for X-Plane via a joint intent/slot DistilBERT
model. Hold the ``Z`` key (or a yoke/joystick button bound to the
``vimaan/push_to_talk`` command), speak, release.

Threading model
---------------
All XPPython3 (`xp.*`) calls must run on the simulator's main thread.
PortAudio buffers audio in C between button down and up, and the flight loop
drains it every frame (Python threads are starved of the GIL inside X-Plane,
so no Python audio callback is used). STT is blocking network/CPU work, so it
runs on a worker thread — given GIL time by the flight loop — that posts
results back through a thread-safe queue. A flight-loop callback drains
the queue on the main thread and dispatches xp.* calls from there. The same
flight loop drives deferred work (engine cranking, post-write verification).

State management
----------------
Toggle-style intents consult the relevant dataref before acting so that
"off" never accidentally turns the system on when it was already off, and
vice-versa. Every dataref/command name here is verified against X-Plane
12.4's DataRefs.txt / Commands.txt (``scripts/check_xplane_refs.py``).

Diagnostics
-----------
Every push-to-talk gets a command id that tags each stage (capture, STT,
NLU, decision, sim writes, verification) in ``~/Vimaan_Logs/``; see
``ML/vimaan_nlu/telemetry.py`` and ``scripts/vimaan_log_report.py``.
"""

import time

_IMPORT_T0 = time.perf_counter()

import logging  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
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

# Telemetry first: it is stdlib-only, so even a failed torch import below is
# recorded by the plugin once it starts.
from vimaan_nlu.telemetry import Telemetry  # noqa: E402

_IMPORT_ERRORS: dict[str, str] = {}
_IMPORT_MS: dict[str, float] = {}


def _timed_import(name):
    t = time.perf_counter()
    try:
        module = __import__(name)
    except Exception as exc:  # recorded and surfaced at plugin start
        _IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        module = None
    _IMPORT_MS[name] = round((time.perf_counter() - t) * 1000, 1)
    return module


torch = _timed_import("torch")
sr = _timed_import("speech_recognition")

from vimaan_nlu.audio_capture import PushToTalkRecorder  # noqa: E402
from vimaan_nlu.audio_health import is_silent, rms_from_pcm16  # noqa: E402
from vimaan_nlu.readback import spell_digits  # noqa: E402
from vimaan_nlu.safety import (  # noqa: E402
    ConfirmationGate,
    SafetyContext,
    command_token,
    evaluate_safety,
)
from vimaan_nlu.stt import transcribe  # noqa: E402
from vimaan_nlu.validators import validate_slot  # noqa: E402

_IMPORT_MS["total"] = round((time.perf_counter() - _IMPORT_T0) * 1000, 1)

# ---------------------------------------------------------------------------
# Listening parameters
# ---------------------------------------------------------------------------
MAX_RECORD_SEC = 12.0  # recording is cut off after this even if still held
MIN_RECORD_SEC = 0.3  # shorter than this is an accidental tap, not a command
# Inside X-Plane, background Python threads only run when the main thread
# lets go of the GIL (measured in-sim: an audio-callback thread got ~0.3% of
# its time). While an STT worker is running, each flight loop sleeps this long
# so the worker can make progress; the cost is paid only while transcribing.
STT_GIL_YIELD_SEC = 0.004
QUEUE_POLL_INTERVAL_SEC = 0.2

# Engine start: hold the starter until the engine reports running, up to this.
STARTER_MAX_CRANK_SEC = 20.0
# Delay before reading back a dataref a *command* should have changed.
VERIFY_DELAY_SEC = 1.0

# ---------------------------------------------------------------------------
# Binary systems: (state dataref, dedicated on cmd, dedicated off cmd, toggle).
# X-Plane has no "servos off" / "fdir off" commands, so "off" uses the toggle
# — but only after reading the state, so it can never turn a system on. The
# flight-director mode datarefs are writable, so FD is set directly.
# ---------------------------------------------------------------------------
TOGGLE_BINDINGS: dict[str, dict[str, Any]] = {
    "toggle_autopilot_1": {
        "dataref": "sim/cockpit2/autopilot/servos_on",
        "on_cmd": "sim/autopilot/servos_on",
        "off_cmd": None,
        "toggle": "sim/autopilot/servos_toggle",
        "label": "Autopilot 1",
    },
    "toggle_autopilot_2": {
        "dataref": "sim/cockpit2/autopilot/servos2_on",
        "on_cmd": "sim/autopilot/servos2_on",
        "off_cmd": None,
        "toggle": "sim/autopilot/servos2_toggle",
        "label": "Autopilot 2",
    },
    "toggle_flight_director_1": {
        "dataref": "sim/cockpit2/autopilot/flight_director_mode",
        "writable": True,
        "on_cmd": "sim/autopilot/fdir_on",
        "off_cmd": None,
        "toggle": "sim/autopilot/fdir_toggle",
        "label": "Flight Director 1",
    },
    "toggle_flight_director_2": {
        "dataref": "sim/cockpit2/autopilot/flight_director2_mode",
        "writable": True,
        "on_cmd": "sim/autopilot/fdir2_on",
        "off_cmd": None,
        "toggle": "sim/autopilot/fdir2_toggle",
        "label": "Flight Director 2",
    },
}

ENGINE_BINDINGS: dict[str, dict[str, Any]] = {
    "toggle_engine_1": {
        "index": 0,
        "starter": "sim/starters/engage_starter_1",
        "shutdown": "sim/starters/shut_down_1",
        "label": "Engine 1",
    },
    "toggle_engine_2": {
        "index": 1,
        "starter": "sim/starters/engage_starter_2",
        "shutdown": "sim/starters/shut_down_2",
        "label": "Engine 2",
    },
}

DREF_HEADING = "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot"
DREF_ALTITUDE = "sim/cockpit2/autopilot/altitude_dial_ft"
# kHz, supports 8.33 kHz spacing: 118.750 MHz -> 118750.
DREF_COM_KHZ = {
    "1": "sim/cockpit2/radios/actuators/com1_frequency_hz_833",
    "2": "sim/cockpit2/radios/actuators/com2_frequency_hz_833",
}
DREF_GEAR_HANDLE = "sim/cockpit2/controls/gear_handle_down"
DREF_GEAR_RETRACTABLE = "sim/aircraft/gear/acf_gear_retract"
DREF_FLAP_HANDLE = "sim/cockpit2/controls/flap_handle_request_ratio"
DREF_PARKING_BRAKE = "sim/cockpit2/controls/parking_brake_ratio"
DREF_ENGINE_RUNNING = "sim/flightmodel/engine/ENGN_running"
DREF_ENGINE_MIXTURE = "sim/flightmodel/engine/ENGN_mixt"
DREF_IGNITION_KEY = "sim/cockpit2/engine/actuators/ignition_key"
DREF_NUM_ENGINES = "sim/aircraft/engine/acf_num_engines"
DREF_BATTERY = "sim/cockpit2/electrical/battery_on"
DREF_PAUSED = "sim/time/paused"

# Slots each actionable intent needs before it can be dispatched (self-test).
REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "set_autopilot_heading": ("degrees",),
    "set_autopilot_altitude": ("altitude",),
    "set_flight_level": ("flight_level",),
    "set_com_frequency": ("frequency",),
    "toggle_landing_gear": ("state",),
    "toggle_flaps": ("state",),
    "toggle_autopilot_1": ("state",),
    "toggle_autopilot_2": ("state",),
    "toggle_flight_director_1": ("state",),
    "toggle_flight_director_2": ("state",),
    "toggle_parking_brake": ("state",),
    "toggle_engine_1": ("state",),
    "toggle_engine_2": ("state",),
}


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
    # <model>/onnx/model.onnx; it duck-types the torch model so handlers are
    # unchanged.
    BACKEND = os.environ.get("VIMAAN_BACKEND", "torch").lower()

    # Speech-to-text backend (Phase 7 / R-09): "google" (online, default),
    # "vosk" or "whisper" (offline, no network). Override with
    # VIMAAN_STT_BACKEND. Offline backends need their package + a model
    # installed; see docs/OFFLINE_STT.md.
    STT_BACKEND = os.environ.get("VIMAAN_STT_BACKEND", "google").lower()
    WHISPER_MODEL = os.environ.get("VIMAAN_WHISPER_MODEL", "base.en")

    # Intents the model can emit that map to no sim command. "None" is a
    # deliberate out-of-scope reject sink (it improves precision on real
    # commands); the conversational intents get a short spoken acknowledgement
    # instead of the misleading "Command not found". A None value => stay
    # silent.
    NON_ACTIONABLE_INTENTS = {
        "None": None,
        "ask_status_generic": "All systems nominal",
        "ask_time": "I cannot read the clock yet",
        "chit_chat_greeting": "Hello, standing by",
    }

    # Seconds a risky command stays "armed" waiting for the pilot to repeat it
    # as confirmation (Phase 6B safety interlocks).
    CONFIRM_WINDOW_SEC = 8.0

    # Canned (utterance, expected intent) pairs for the Ctrl-Z self-test
    # (Phase 6C): one per actionable intent family, run through text -> NLU
    # (STT bypassed, nothing actuated). A probe passes only if the intent is
    # the expected one AND every slot its handler needs was extracted.
    SELF_TEST_PROBES = (
        ("set heading two seven zero", "set_autopilot_heading"),
        ("climb to flight level three five zero", "set_flight_level"),
        ("set altitude one five thousand", "set_autopilot_altitude"),
        ("gear down", "toggle_landing_gear"),
        ("flaps up", "toggle_flaps"),
        ("engage autopilot one", "toggle_autopilot_1"),
        ("flight director two on", "toggle_flight_director_2"),
        ("tune com one to one one eight decimal seven five", "set_com_frequency"),
        ("start engine one", "toggle_engine_1"),
        ("parking brake on", "toggle_parking_brake"),
    )

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
        self.loader: Any = None
        self.recorder: Any = None
        self.recognizer: Any = None
        self.device = None
        self._recording: dict[str, Any] | None = None  # the press being recorded
        self._transcribing = False  # an STT worker is running
        self._worker_lock = threading.Lock()
        self._result_queue: queue.Queue = queue.Queue()
        self._flight_loop_id = None
        self.hotkeyPress = None
        self.hotkeyRelease = None
        self.hotkeySelfTest = None
        self._commands: list[tuple[Any, Any]] = []
        self._current_cmd: str | None = None
        # Deferred main-thread work: dataref read-backs and starter cranks.
        self._verifications: list[dict[str, Any]] = []
        self._cranks: dict[str, dict[str, Any]] = {}

        # Safety interlocks (Phase 6B): risky commands must be repeated to confirm.
        self._confirm_gate = ConfirmationGate(window_sec=self.CONFIRM_WINDOW_SEC)
        self.intent_to_command = self._setup_intent_handlers()

        if _IMPORT_ERRORS:
            # Missing torch / SpeechRecognition in XPPython3's Python is the
            # most likely first-run failure; say exactly which and stay loaded
            # so the log reaches the user instead of a silent plugin failure.
            self.tel.event("dependency_missing", logging.ERROR, missing=_IMPORT_ERRORS)
            return

        self.recognizer = sr.Recognizer()
        self.recorder = self._setup_microphone()
        self.device = torch.device("cpu")
        self._init_model()

    def _setup_microphone(self):
        try:
            recorder = PushToTalkRecorder(max_sec=MAX_RECORD_SEC)
            self.tel.event("microphone_ready", **recorder.describe())
            return recorder
        except Exception as exc:
            # ImportError: sounddevice missing; PortAudioError: no input device.
            self.tel.error("microphone_unavailable", exc, hint="pip install sounddevice")
            return None

    def _init_model(self):
        from vimaan_nlu.model_loader import ModelLoader

        try:
            with self.tel.stage("model_load", backend=self.BACKEND) as res:
                self.loader = ModelLoader(self.device)
                use_onnx = self.BACKEND == "onnx"
                results = self.loader.load_all(
                    quantize=self.USE_QUANTIZED_MODEL and not use_onnx,
                    backend=self.BACKEND,
                )
                self.device = self.loader.device
                res.update(
                    model_path=results["model"]["model_path"],
                    device=str(self.device),
                    quantized=results.get("quantized"),
                    intents=results["maps"]["intents"],
                    slots=results["maps"]["slots"],
                    # transformers is imported lazily by the loader, so its
                    # version is only known now.
                    versions=self.tel.environment(),
                )
            self.ready = True
            # Warm-up: the first forward pass is several times slower than the
            # rest; pay it now rather than on the pilot's first command.
            with self.tel.stage("nlu", cmd="warmup"):
                self._predict("set heading two seven zero")
        except Exception:
            # stage() already logged the traceback; keep the plugin loaded so
            # the pilot hears the failure and the log has the cause.
            self.ready = False

    def _setup_intent_handlers(self):
        return {
            "set_autopilot_heading": self._set_heading,
            "set_autopilot_altitude": self._set_altitude,
            "set_flight_level": self._set_flight_level,
            "set_com_frequency": self._set_com_frequency,
            "toggle_landing_gear": self._toggle_landing_gear,
            "toggle_flaps": self._toggle_flaps,
            "toggle_autopilot_1": self._toggle_binary,
            "toggle_autopilot_2": self._toggle_binary,
            "toggle_flight_director_1": self._toggle_binary,
            "toggle_flight_director_2": self._toggle_binary,
            "toggle_parking_brake": self._toggle_parking_brake,
            "toggle_engine_1": self._toggle_engine,
            "toggle_engine_2": self._toggle_engine,
        }

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

        # Flight loop drains worker-thread results onto the main thread.
        self._flight_loop_id = xp.createFlightLoop(self._flight_loop, phase=1)
        xp.scheduleFlightLoop(self._flight_loop_id, QUEUE_POLL_INTERVAL_SEC, 1)

        self.tel.event("plugin_start", ready=self.ready, sim=self._sim_info())
        if self.ready:
            self._say("Vimaan AI CoPilot ready")
        elif _IMPORT_ERRORS:
            self._say("Vimaan could not load its Python packages. Check the log.")
        else:
            self._say("Vimaan could not load its model. Check the log.")
        return self.Name, self.Sig, self.Desc

    def XPluginStop(self):
        # Never leave a starter engaged.
        for engine in list(self._cranks):
            self._end_crank(engine, "plugin_stop")
        for cmd_ref, handler in self._commands:
            xp.unregisterCommandHandler(cmd_ref, handler, 1, None)
        self._commands = []
        if self._recording is not None and self.recorder is not None:
            self.recorder.stop()
            self._recording = None
        for hotkey in (self.hotkeyPress, self.hotkeyRelease):
            if hotkey:
                xp.unregisterHotKey(hotkey)
        if self.hotkeySelfTest:
            xp.unregisterHotKey(self.hotkeySelfTest)
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
        pass

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
    # Push-to-talk
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
        if self._recording is not None:
            return  # key auto-repeat while held
        if not self.ready:
            self.tel.event("ptt_ignored", logging.WARNING, reason="not_ready", source=source)
            self._say("Vimaan is not ready. Check the log.")
            return
        if self.recorder is None:
            self.tel.event("ptt_ignored", logging.WARNING, reason="no_microphone", source=source)
            self._say("Microphone not available")
            return
        with self._worker_lock:
            busy = self._transcribing
        if busy:
            self.tel.event("ptt_ignored", reason="still_transcribing", source=source)
            return

        cmd = self.tel.new_command_id()
        t_press = time.perf_counter()
        try:
            self.recorder.start()
        except Exception as exc:
            self.tel.error("microphone_start_failed", exc, cmd=cmd)
            self.tel.command_done(cmd, "mic_error", t_press)
            self._say("Microphone error")
            return
        self._recording = {"cmd": cmd, "t_press": t_press, "source": source}
        self.tel.event("ptt_press", cmd=cmd, source=source)

    def OnReleaseCallback(self, inRefcon, source="hotkey"):
        rec, self._recording = self._recording, None
        if rec is None:
            return
        cmd, t_press = rec["cmd"], rec["t_press"]
        raw = self.recorder.stop()
        audio_sec = self.recorder.seconds(raw)
        self.tel.event(
            "stage",
            stage="capture",
            cmd=cmd,
            ms=round((time.perf_counter() - t_press) * 1000, 1),
            audio_sec=round(audio_sec, 2),
            truncated=self.recorder.truncated,
            overflows=self.recorder.overflows,
            released_by=source,
        )
        if audio_sec < MIN_RECORD_SEC:
            self.tel.command_done(cmd, "too_short", t_press)
            return
        with self._worker_lock:
            self._transcribing = True
        threading.Thread(
            target=self._stt_worker, args=(cmd, t_press, raw), name=f"stt-{cmd}", daemon=True
        ).start()

    # ------------------------------------------------------------------
    # Worker / queue plumbing
    # ------------------------------------------------------------------
    def _enqueue(self, kind, payload=None):
        self._result_queue.put((kind, payload, time.perf_counter()))

    def _stt_worker(self, cmd, t_press, raw):
        outcome = None
        try:
            # Mic-health (Phase 6F): a dead/muted mic (or no macOS permission)
            # yields near-silent audio. Say so instead of a confusing
            # "didn't catch that".
            rms = rms_from_pcm16(raw)
            self.tel.event("audio_level", cmd=cmd, rms=round(rms, 1))
            if is_silent(rms):
                outcome = "silent_mic"
                self._enqueue("speak", "Microphone appears silent")
                return

            t_speech_end = time.perf_counter()
            audio = sr.AudioData(raw, self.recorder.sample_rate, 2)
            with self.tel.stage(
                "stt", cmd=cmd, expected=(sr.UnknownValueError,), backend=self.STT_BACKEND
            ) as res:
                text = transcribe(
                    self.recognizer, audio, self.STT_BACKEND, whisper_model=self.WHISPER_MODEL
                )
                res["text"] = text
            self._enqueue(
                "execute",
                {"cmd": cmd, "text": text, "t_press": t_press, "t_speech_end": t_speech_end},
            )
        except sr.UnknownValueError:
            outcome = "not_understood"
            self._enqueue("speak", "I could not understand you")
        except sr.RequestError:
            outcome = "stt_service_error"  # network down / quota for google
            self._enqueue("speak", "Recognition service failed")
        except Exception as exc:
            outcome = "worker_error"
            self.tel.error("stt_worker_error", exc, cmd=cmd)
            self._enqueue("speak", "An error occurred")
        finally:
            with self._worker_lock:
                self._transcribing = False
            if outcome is not None:
                self.tel.command_done(cmd, outcome, t_press)

    def _flight_loop(self, *_):
        t0 = time.perf_counter()
        try:
            while True:
                kind, payload, t_enq = self._result_queue.get_nowait()
                if kind == "speak":
                    self._say(payload)
                elif kind == "execute":
                    payload["queue_wait_ms"] = round((time.perf_counter() - t_enq) * 1000, 1)
                    self._execute_command(payload)
        except queue.Empty:
            pass
        except Exception as exc:  # never let the flight loop die
            self.tel.error("flight_loop_error", exc)
        try:
            if self._recording is not None:
                self.recorder.poll()
                # A missed key-up must not leave the mic open forever.
                if self.recorder.elapsed() > MAX_RECORD_SEC + 0.5:
                    self.OnReleaseCallback(None, source="auto_stop")
            self._service_cranks()
            self._service_verifications()
        except Exception as exc:
            self.tel.error("flight_loop_error", exc)
        busy_ms = (time.perf_counter() - t0) * 1000
        with self._worker_lock:
            transcribing = self._transcribing
        if transcribing:
            time.sleep(STT_GIL_YIELD_SEC)  # deliberate; not counted as our busy time
        self.tel.tick(
            busy_ms=busy_ms,
            queue_depth=self._result_queue.qsize(),
            pending_verifications=len(self._verifications),
            paused=self._read_int(DREF_PAUSED),
        )
        if self._recording is not None or transcribing:
            return -1.0  # every frame: drain the mic / give the STT worker GIL time
        return QUEUE_POLL_INTERVAL_SEC

    # ------------------------------------------------------------------
    # NLU + dispatch (main thread only)
    # ------------------------------------------------------------------
    def _predict(self, text):
        from vimaan_nlu.inference import predict

        return predict(
            text,
            self.loader.model,
            self.loader.tokenizer,
            self.device,
            self.loader.intent_map_rev,
            self.loader.slot_map_rev,
        )

    def _execute_command(self, job):
        cmd, text = job["cmd"], job["text"]
        self._current_cmd = cmd
        outcome = "failed"
        try:
            with self.tel.stage("nlu", cmd=cmd, queue_wait_ms=job.get("queue_wait_ms")) as res:
                result = self._predict(text)
                res.update(
                    normalized=result["normalized_text"],
                    intent=result["intent"],
                    confidence=round(result["confidence"], 3),
                    slots=result["slots"],
                )
            intent_pred, slots = result["intent"], result["slots"]

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

            # Commands this aircraft physically can't do are rejected before
            # the safety gate, which would otherwise ask the pilot to confirm
            # "gear up" on a fixed-gear Cessna.
            not_applicable = self._not_applicable(intent_pred)
            if not_applicable:
                outcome = "rejected"
                self.tel.event("not_applicable", cmd=cmd, intent=intent_pred, reason=not_applicable)
                self._say(not_applicable)
                return

            # Safety interlock: gate irreversible/dangerous commands on a
            # confirmation (repeat the command) when risky in the current flight
            # context. Fails safe if the relevant datarefs are unavailable.
            ctx = self._read_safety_context()
            verdict = evaluate_safety(intent_pred, slots, ctx)
            token = command_token(intent_pred, slots)
            if verdict.requires_confirmation and not self._confirm_gate.confirm(token):
                self._confirm_gate.arm(token)
                outcome = "safety_hold"
                self.tel.event(
                    "safety_hold",
                    logging.WARNING,
                    cmd=cmd,
                    rule=verdict.rule,
                    fail_safe=verdict.fail_safe,
                    context=ctx.__dict__,
                )
                self._say(f"{verdict.message}. Repeat to confirm.")
                return
            # Any other command cancels a pending confirmation.
            self._confirm_gate.clear()

            with self.tel.stage("dispatch", cmd=cmd, intent=intent_pred) as res:
                outcome = handler(slots, intent_pred) or "executed"
                res["outcome"] = outcome
        except Exception as exc:
            self.tel.error("execute_error", exc, cmd=cmd, text=text)
            self._say("Command execution failed")
        finally:
            self._current_cmd = None
            self.tel.command_done(
                cmd,
                outcome,
                job["t_press"],
                text=text,
                speech_to_action_ms=round((time.perf_counter() - job["t_speech_end"]) * 1000, 1),
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _run_self_test(self, *_):
        """Ctrl-Z: run canned utterances through text -> NLU (STT bypassed,
        nothing actuated) and check each resolves to the expected intent with
        every slot its handler needs. Logs each probe and speaks a summary."""
        if not self.ready:
            self._say("Vimaan is not ready. Check the log.")
            return
        self._say("Running self test")
        passed = 0
        with self.tel.stage("self_test") as summary:
            for text, expected in self.SELF_TEST_PROBES:
                try:
                    result = self._predict(text)
                except Exception as exc:
                    self.tel.error("self_test_error", exc, text=text)
                    continue
                intent, slots = result["intent"], result["slots"]
                missing = [s for s in REQUIRED_SLOTS.get(intent, ()) if not slots.get(s)]
                ok = intent == expected and intent in self.intent_to_command and not missing
                passed += ok
                self.tel.event(
                    "self_test_probe",
                    logging.INFO if ok else logging.WARNING,
                    text=text,
                    expected=expected,
                    intent=intent,
                    confidence=round(result["confidence"], 3),
                    slots=slots,
                    missing_slots=missing,
                    ok=ok,
                )
            summary.update(passed=passed, total=len(self.SELF_TEST_PROBES))
        self._say(f"Self test complete. {passed} of {len(self.SELF_TEST_PROBES)} passed.")

    # ------------------------------------------------------------------
    # Sim I/O helpers (every read/write/command is logged with the command id)
    # ------------------------------------------------------------------
    def _say(self, text):
        self.tel.event("speak", logging.DEBUG, cmd=self._current_cmd, text=text)
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

    def _write(self, name, value, is_float=True):
        """Write a dataref and read it straight back. Returns True on success."""
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
        xp.commandOnce(cmd_ref)
        self.tel.event("sim_command", cmd=self._current_cmd, command=command_path, ok=True)
        return True

    def _expect(self, label, read, expected, delay=None):
        """After a command, check the sim actually changed (logged, not spoken)."""
        self._verifications.append(
            {
                "cmd": self._current_cmd,
                "label": label,
                "read": read,
                "expected": expected,
                "due": time.perf_counter() + (VERIFY_DELAY_SEC if delay is None else delay),
            }
        )

    def _service_verifications(self):
        now = time.perf_counter()
        due = [v for v in self._verifications if v["due"] <= now]
        self._verifications = [v for v in self._verifications if v["due"] > now]
        for v in due:
            actual = v["read"]()
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

    def _not_applicable(self, intent):
        """Why this aircraft can't perform ``intent``, or None if it can."""
        if intent == "toggle_landing_gear" and self._read_int(DREF_GEAR_RETRACTABLE) == 0:
            return "This aircraft has fixed gear"
        if intent in ENGINE_BINDINGS:
            num_engines = self._read_int(DREF_NUM_ENGINES)
            eng = ENGINE_BINDINGS[intent]
            if num_engines is not None and eng["index"] >= num_engines:
                return f"This aircraft has no {eng['label'].lower()}"
        return None

    def _read_safety_context(self):
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
        # Validators enforce the documented ranges before we touch a dataref
        # (heading wraps into 0-360; the others reject out-of-range values so a
        # mis-decoded number can't be written to the sim).
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
        self._say(f"Altitude {int(v)} feet")
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
        self._say(f"Flight level {spell_digits(int(float(value)))}")
        return "executed"

    def _set_com_frequency(self, slots, intent):
        frequency = slots.get("frequency")
        if not frequency:
            self._say("Frequency missing")
            return "rejected"
        fval, ok, reason = validate_slot("frequency", frequency)
        if not ok:
            self.tel.event(
                "slot_rejected", logging.WARNING, slot="frequency", value=frequency, reason=reason
            )
            self._say("Frequency out of range")
            return "rejected"
        port = str(slots.get("com_port", "1"))
        if port not in DREF_COM_KHZ:
            self._say(f"COM {port} is not supported. Use COM 1 or 2.")
            return "rejected"
        khz = int(round(float(fval) * 1000))
        if not self._write(DREF_COM_KHZ[port], khz, is_float=False):
            self._say(f"Cannot tune COM {port}")
            return "failed"
        self._say(f"COM {port} set to {spell_digits(fval)}")
        return "executed"

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

    def _toggle_flaps(self, slots, intent):
        state = slots.get("state")
        before = self._read_float(DREF_FLAP_HANDLE)
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
        return self._ensure_state(intent, state)

    def _ensure_state(self, intent, desired):
        """Move a binary system to ``desired`` ('on'/'off') idempotently."""
        binding = TOGGLE_BINDINGS[intent]
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
        eng = ENGINE_BINDINGS[intent]
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
        if intent in self._cranks:
            self._say(f"{label} already starting")
            return "no_op"
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
        self._cranks[intent] = {
            "cmd": self._current_cmd,
            "ref": cmd_ref,
            "started": time.perf_counter(),
            **eng,
        }
        self.tel.event(
            "crank_begin",
            cmd=self._current_cmd,
            engine=label,
            mixture=self._read_array(DREF_ENGINE_MIXTURE, eng["index"], True),
            ignition_key=self._read_array(DREF_IGNITION_KEY, eng["index"]),
            battery=self._read_array(DREF_BATTERY, 0),
        )
        self._say(f"Starting {label.lower()}")
        return "pending"

    def _service_cranks(self):
        for intent, crank in list(self._cranks.items()):
            if self._read_array(DREF_ENGINE_RUNNING, crank["index"]) == 1:
                self._end_crank(intent, "running")
                self._say(f"{crank['label']} running")
            elif time.perf_counter() - crank["started"] > STARTER_MAX_CRANK_SEC:
                self._end_crank(intent, "timeout")
                self._say(f"{crank['label']} did not start. Check fuel, mixture and ignition.")

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
            mixture=self._read_array(DREF_ENGINE_MIXTURE, crank["index"], True),
            ignition_key=self._read_array(DREF_IGNITION_KEY, crank["index"]),
        )
