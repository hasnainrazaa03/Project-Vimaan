"""Vimaan AI Co-Pilot — X-Plane plugin entry point.

Push-to-talk (Z key) speech command for X-Plane via a joint intent/slot
DistilBERT model.

Threading model
---------------
All XPPython3 (`xp.*`) calls must run on the simulator's main thread.
Speech capture and Google STT are blocking I/O and would freeze the sim if
invoked synchronously from the key callback, so we run them on a worker
thread and post results back through a thread-safe queue. A flight-loop
callback drains the queue on the main thread and dispatches xp.* calls
from there.

State management
----------------
Toggle-style intents (autopilot, FD, parking brake, engine) consult the
relevant dataref before acting so that "off" never accidentally turns the
system on when it was already off, and vice-versa.
"""

import logging
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

import speech_recognition as sr
import torch
from XPPython3 import xp

# Plugin lives in <repo>/plugin/, ML package is at <repo>/ML/
_ML_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ML"))
if _ML_PATH not in sys.path:
    sys.path.insert(0, _ML_PATH)

from vimaan_nlu.inference import predict  # noqa: E402
from vimaan_nlu.model_loader import ModelLoader  # noqa: E402

# ---------------------------------------------------------------------------
# Listening parameters
# ---------------------------------------------------------------------------
LISTEN_TIMEOUT_SEC = 8  # max seconds to wait for speech to start
PHRASE_TIME_LIMIT_SEC = 12  # max seconds of recorded speech per press
QUEUE_POLL_INTERVAL_SEC = 0.2

# ---------------------------------------------------------------------------
# Intent -> (dataref to read state, dedicated on cmd, dedicated off cmd,
#            fallback toggle cmd)
# Datarefs are 0/1 integers indicating the system is off/on.
# ---------------------------------------------------------------------------
TOGGLE_BINDINGS = {
    "toggle_autopilot_1": {
        "dataref": "sim/cockpit2/autopilot/servos_on",
        "on_cmd": "sim/autopilot/servos_on",
        "off_cmd": "sim/autopilot/servos_off",
        "toggle": "sim/autopilot/servos_toggle",
        "label": "Autopilot 1",
    },
    "toggle_autopilot_2": {
        "dataref": "sim/cockpit2/autopilot/servos2_on",
        "on_cmd": "sim/autopilot/servos2_on",
        "off_cmd": "sim/autopilot/servos2_off",
        "toggle": "sim/autopilot/servos2_toggle",
        "label": "Autopilot 2",
    },
    "toggle_flight_director_1": {
        "dataref": "sim/cockpit2/autopilot/flight_director_mode",
        "on_cmd": "sim/autopilot/fdir_on",
        "off_cmd": "sim/autopilot/fdir_off",
        "toggle": "sim/autopilot/fdir_toggle",
        "label": "Flight Director 1",
    },
    "toggle_flight_director_2": {
        "dataref": "sim/cockpit2/autopilot/flight_director2_mode",
        "on_cmd": "sim/autopilot/fdir2_on",
        "off_cmd": "sim/autopilot/fdir2_off",
        "toggle": "sim/autopilot/fdir2_toggle",
        "label": "Flight Director 2",
    },
    "toggle_parking_brake": {
        "dataref": "sim/cockpit2/controls/parking_brake_ratio",
        "on_cmd": "sim/flight_controls/brakes_toggle_max",  # not ideal; see set below
        "off_cmd": "sim/flight_controls/brakes_toggle_max",
        "toggle": "sim/flight_controls/brakes_toggle_max",
        "label": "Parking brake",
        "is_float_dataref": True,
    },
    "toggle_engine_1": {
        "dataref": "sim/cockpit2/engine/actuators/starter_hit[0]",
        "on_cmd": "sim/starters/engage_starter_1",
        "off_cmd": "sim/starters/shut_down_1",
        "toggle": None,
        "label": "Engine 1",
        "stateless": True,  # momentary; always fire on/off cmd directly
    },
    "toggle_engine_2": {
        "dataref": "sim/cockpit2/engine/actuators/starter_hit[1]",
        "on_cmd": "sim/starters/engage_starter_2",
        "off_cmd": "sim/starters/shut_down_2",
        "toggle": None,
        "label": "Engine 2",
        "stateless": True,
    },
}


class PythonInterface:
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self):
        self.Name = "Vimaan AI CoPilot"
        self.Sig = "plugin.vimaan.aicopilot.bymhr"
        self.Desc = "Voice command interface with joint intent + slot NLU for X-Plane"

        self.logger = None
        self._setup_logging()

        self.recognizer = sr.Recognizer()
        self.microphone = self._setup_microphone()
        self._is_listening = False
        self._listen_lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"[Vimaan] Using device: {self.device}")

        self.loader = ModelLoader(self.device)
        self._init_model()

        self.intent_to_command = self._setup_intent_handlers()

        self._result_queue = queue.Queue()
        self._flight_loop_id = None
        self.hotkeyPress = None
        self.hotkeyRelease = None

    def _setup_logging(self):
        try:
            log_folder = Path.home() / "Vimaan_Logs"
            log_folder.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_folder / f"vimaan_plugin_{timestamp}.log"

            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(log_file, mode="w", encoding="utf-8"),
                    logging.StreamHandler(),
                ],
            )
            self.logger = logging.getLogger("VimaanCoPilot")
            self.logger.info("=== VIMAAN AI COPILOT LOG START ===")
            self.logger.info(f"Log file: {log_file}")
        except Exception as exc:  # pragma: no cover — best effort
            self.logger = None
            self.log(f"[Vimaan] Failed to setup logging: {exc}")

    def log(self, message):
        try:
            xp.log(message)
        except Exception:
            pass
        if self.logger:
            self.logger.info(message)

    def _setup_microphone(self):
        try:
            return sr.Microphone()
        except (AttributeError, OSError) as exc:
            self.log(f"[Vimaan] WARNING: microphone unavailable ({exc})")
            return None

    def _init_model(self):
        try:
            results = self.loader.load_all()
            self.log(f"[Vimaan] Model loaded from: {results['model']['model_path']}")
            self.log(f"[Vimaan] Device: {results['model']['device']}")
            self.log(
                f"[Vimaan] Intents: {results['maps']['intents']}, Slots: {results['maps']['slots']}"
            )
        except Exception as exc:
            self.log(f"[Vimaan] ERROR loading model: {exc}")
            raise

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
            "toggle_engine_1": self._toggle_binary,
            "toggle_engine_2": self._toggle_binary,
        }

    # ------------------------------------------------------------------
    # XPPython3 plugin hooks
    # ------------------------------------------------------------------
    def XPluginStart(self):
        self.hotkeyPress = xp.registerHotKey(
            xp.VK_Z,
            xp.DownFlag,
            "Vimaan Push-to-Talk -> Press",
            self.OnPressCallback,
        )
        self.hotkeyRelease = xp.registerHotKey(
            xp.VK_Z,
            xp.UpFlag,
            "Vimaan Push-to-Talk -> Release",
            self.OnReleaseCallback,
        )

        # Flight loop drains worker-thread results onto the main thread.
        self._flight_loop_id = xp.createFlightLoop(self._drain_results, phase=1)
        xp.scheduleFlightLoop(self._flight_loop_id, QUEUE_POLL_INTERVAL_SEC, 1)

        xp.speakString("Vimaan AI CoPilot ready")
        return self.Name, self.Sig, self.Desc

    def XPluginStop(self):
        if self.hotkeyPress:
            xp.unregisterHotKey(self.hotkeyPress)
        if self.hotkeyRelease:
            xp.unregisterHotKey(self.hotkeyRelease)
        if self._flight_loop_id:
            xp.destroyFlightLoop(self._flight_loop_id)
            self._flight_loop_id = None

    def XPluginEnable(self):
        return 1

    def XPluginDisable(self):
        pass

    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        pass

    # ------------------------------------------------------------------
    # Push-to-talk hooks
    # ------------------------------------------------------------------
    def OnPressCallback(self, inRefcon):
        if self.microphone is None:
            xp.speakString("Microphone not available")
            return

        # Spin up a worker if nothing is already listening. The worker
        # captures audio + runs Google STT off-thread so the sim main loop
        # is never blocked.
        with self._listen_lock:
            if self._is_listening:
                return
            self._is_listening = True

        threading.Thread(target=self._listen_worker, daemon=True).start()

    def OnReleaseCallback(self, inRefcon):
        # Advisory only — SpeechRecognition's listen() cannot be cut short
        # from outside; the worker will exit on natural phrase end or on
        # PHRASE_TIME_LIMIT_SEC.
        pass

    # ------------------------------------------------------------------
    # Worker / queue plumbing
    # ------------------------------------------------------------------
    def _enqueue(self, kind, payload=None):
        self._result_queue.put((kind, payload))

    def _listen_worker(self):
        try:
            self._enqueue("speak", "Listening")
            self.log("[Vimaan] Recording started…")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT_SEC,
                    phrase_time_limit=PHRASE_TIME_LIMIT_SEC,
                )
            self._enqueue("speak", "Processing")
            text = self.recognizer.recognize_google(audio)
            self.log(f"[Vimaan] Recognized text: {text}")
            self._enqueue("execute", text)
        except sr.WaitTimeoutError:
            self.log("[Vimaan] No speech detected (timeout)")
            self._enqueue("speak", "No speech detected")
        except sr.UnknownValueError:
            self.log("[Vimaan] Speech not understood")
            self._enqueue("speak", "I could not understand you")
        except sr.RequestError as exc:
            self.log(f"[Vimaan] Google STT error: {exc}")
            self._enqueue("speak", "Recognition service failed")
        except Exception as exc:
            self.log(f"[Vimaan] Listen worker error: {exc}")
            self._enqueue("speak", "An error occurred")
        finally:
            with self._listen_lock:
                self._is_listening = False

    def _drain_results(self, *_):
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()
                if kind == "speak":
                    xp.speakString(payload)
                elif kind == "execute":
                    self._execute_command(payload)
        except queue.Empty:
            pass
        return QUEUE_POLL_INTERVAL_SEC

    # ------------------------------------------------------------------
    # NLU + dispatch (main thread only)
    # ------------------------------------------------------------------
    def _execute_command(self, text):
        try:
            result = predict(
                text,
                self.loader.model,
                self.loader.tokenizer,
                self.device,
                self.loader.intent_map_rev,
                self.loader.slot_map_rev,
            )
            intent_pred = result["intent"]
            slots = result["slots"]

            self.log(
                f"[Vimaan] Intent: {intent_pred}  Slots: {slots}  Conf: {result['confidence']:.2f}"
            )

            handler = self.intent_to_command.get(intent_pred)
            if handler is None or intent_pred == "None":
                self.log(f"[Vimaan] Intent '{intent_pred}' has no handler")
                xp.speakString("Command not found")
                return

            # Dispatch with intent context so generic handlers can branch.
            handler(slots, intent_pred)
        except Exception as exc:
            import traceback

            self.log(f"[Vimaan] Error executing command: {exc}\n{traceback.format_exc()}")
            xp.speakString("Command execution failed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_state_int(dataref_name, is_float=False):
        dref = xp.findDataRef(dataref_name)
        if not dref:
            return None
        try:
            val = xp.getDataf(dref) if is_float else xp.getDatai(dref)
            return 1 if val > 0.5 else 0
        except Exception:
            return None

    def _fire_command(self, command_path):
        if not command_path:
            return False
        cmd = xp.findCommand(command_path)
        if not cmd:
            self.log(f"[Vimaan] Command not found: {command_path}")
            return False
        xp.commandOnce(cmd)
        return True

    def _ensure_state(self, intent, desired):
        """Move a binary system to `desired` ('on'/'off') idempotently."""
        binding = TOGGLE_BINDINGS.get(intent)
        if not binding:
            self.log(f"[Vimaan] No toggle binding for {intent}")
            return

        target = 1 if desired == "on" else 0

        # Stateless systems (engines): always fire the dedicated cmd.
        if binding.get("stateless"):
            ok = self._fire_command(binding["on_cmd"] if target else binding["off_cmd"])
            if ok:
                xp.speakString(f"{binding['label']} {desired}")
            return

        current = self._read_state_int(
            binding["dataref"], is_float=binding.get("is_float_dataref", False)
        )
        if current is None:
            # Best effort: fire the dedicated cmd; otherwise toggle.
            ok = self._fire_command(
                binding["on_cmd"] if target else binding["off_cmd"]
            ) or self._fire_command(binding["toggle"])
        elif current == target:
            self.log(f"[Vimaan] {binding['label']} already {desired}; no action")
            xp.speakString(f"{binding['label']} already {desired}")
            return
        else:
            ok = self._fire_command(
                binding["on_cmd"] if target else binding["off_cmd"]
            ) or self._fire_command(binding["toggle"])

        if ok:
            xp.speakString(f"{binding['label']} {desired}")
        else:
            xp.speakString(f"Cannot {desired} {binding['label']}")

    # ------------------------------------------------------------------
    # Intent handlers (all take (slots, intent))
    # ------------------------------------------------------------------
    def _set_heading(self, slots, intent):
        degrees = slots.get("degrees")
        if not degrees:
            xp.speakString("Heading value missing")
            return
        try:
            value = float(degrees) % 360
            dref = xp.findDataRef("sim/cockpit/autopilot/heading_mag")
            if dref:
                xp.setDataf(dref, value)
                xp.speakString(f"Heading {int(value):03d}")
        except (TypeError, ValueError):
            xp.speakString("Invalid heading value")

    def _set_altitude(self, slots, intent):
        altitude = slots.get("altitude")
        if not altitude:
            xp.speakString("Altitude value missing")
            return
        try:
            value = float(altitude)
            dref = xp.findDataRef("sim/cockpit/autopilot/altitude")
            if dref:
                xp.setDataf(dref, value)
                xp.speakString(f"Altitude {int(value)} feet")
        except (TypeError, ValueError):
            xp.speakString("Invalid altitude value")

    def _set_flight_level(self, slots, intent):
        flight_level = slots.get("flight_level")
        if not flight_level:
            xp.speakString("Flight level missing")
            return
        try:
            value = float(flight_level) * 100
            dref = xp.findDataRef("sim/cockpit/autopilot/altitude")
            if dref:
                xp.setDataf(dref, value)
                xp.speakString(f"Flight level {int(float(flight_level))}")
        except (TypeError, ValueError):
            xp.speakString("Invalid flight level")

    def _set_com_frequency(self, slots, intent):
        com_port = str(slots.get("com_port", "1"))
        frequency = slots.get("frequency")
        if not frequency:
            xp.speakString("Frequency missing")
            return
        try:
            freq_hz = int(round(float(frequency) * 1_000_000))
            dref_name = (
                "sim/cockpit/radios/com1_freq_hz"
                if com_port == "1"
                else "sim/cockpit/radios/com2_freq_hz"
            )
            dref = xp.findDataRef(dref_name)
            if dref:
                xp.setDatai(dref, freq_hz)
                xp.speakString(f"COM {com_port} set to {frequency}")
        except (TypeError, ValueError):
            xp.speakString("Invalid frequency")

    def _toggle_landing_gear(self, slots, intent):
        state = slots.get("state")
        if state == "up":
            self._fire_command("sim/flight_controls/landing_gear_up")
            xp.speakString("Gear up")
        elif state == "down":
            self._fire_command("sim/flight_controls/landing_gear_down")
            xp.speakString("Gear down")
        else:
            self._fire_command("sim/flight_controls/landing_gear_toggle")
            xp.speakString("Gear toggled")

    def _toggle_flaps(self, slots, intent):
        state = slots.get("state")
        if state == "up":
            self._fire_command("sim/flight_controls/flaps_up")
            xp.speakString("Flaps up")
        elif state == "down":
            self._fire_command("sim/flight_controls/flaps_down")
            xp.speakString("Flaps down")
        else:
            xp.speakString("Flaps direction missing")

    def _toggle_parking_brake(self, slots, intent):
        state = slots.get("state")
        if state not in ("on", "off"):
            xp.speakString("Parking brake state missing")
            return
        target = 1.0 if state == "on" else 0.0
        dref = xp.findDataRef("sim/cockpit2/controls/parking_brake_ratio")
        if dref:
            xp.setDataf(dref, target)
            xp.speakString(f"Parking brake {state}")
        else:
            xp.speakString("Parking brake dataref not found")

    def _toggle_binary(self, slots, intent):
        state = slots.get("state")
        if state not in ("on", "off"):
            xp.speakString("Command state missing")
            return
        self._ensure_state(intent, state)
