from XPPython3 import xp
import sounddevice as sd
import numpy as np
import whisper
import time
from scipy.signal import resample

# test terminal commands


class PythonInterface:

    def __init__(self):
        # ---------------- Plugin Metadata ----------------
        self.Name = "AP1 Voice Control"
        self.Sig = "vyomshukla.plugin.ap1.voice"
        self.Desc = "Whisper Tiny GPU voice control for AP1"

        # ---------------- State ----------------
        self.enabled = False
        self.isRecording = False
        self.start_time = 0.0

        # ---------------- Audio ----------------
        self.sample_rate = 16000          # Whisper required
        self.input_rate = None            # Device native rate
        self.device = None
        self.channels = 1

        # ---------------- Menu / Hotkeys ----------------
        self.menu_id = None
        self.menu_item = None
        self.hk_press = None
        self.hk_release = None

        # ---------------- Whisper ----------------
        xp.log("[AP1 Voice] Loading Whisper Tiny (GPU)...")
        self.model = whisper.load_model("tiny", device="cuda")
        xp.log("[AP1 Voice] Whisper ready")

        # ---------------- Commands ----------------
        self.ap1_engage_phrases = {
            "ap1 on",
            "engage ap1",
            "activate ap1",
            "enable ap1",
            "autopilot one on",
            "autopilot on"
        }

        self.ap1_disengage_phrases = {
            "ap1 off",
            "disengage ap1",
            "deactivate ap1",
            "disable ap1",
            "autopilot one off",
            "autopilot off"
        }

    # =====================================================
    # MICROPHONE DETECTION (HARDWARE ONLY)
    # =====================================================

    def detect_microphone(self):
        try:
            devices = sd.query_devices()
            xp.log("[AP1 Voice] Scanning audio input devices...")

            for idx, dev in enumerate(devices):
                name = dev["name"].lower()

                if dev["max_input_channels"] < 1:
                    continue

                # Skip Windows virtual devices
                if "sound mapper" in name:
                    continue
                if "stereo mix" in name:
                    continue
                if "loopback" in name:
                    continue

                self.device = idx
                self.channels = dev["max_input_channels"]
                self.input_rate = int(dev["default_samplerate"])

                xp.log(
                    f"[AP1 Voice] Selected mic #{idx}: {dev['name']} "
                    f"(channels={self.channels}, rate={self.input_rate})"
                )
                return

            xp.log("[AP1 Voice] ERROR: No suitable microphone found")
            self.device = None

        except Exception as e:
            xp.log(f"[AP1 Voice] Mic detection failed: {e}")
            self.device = None

    # =====================================================
    # X-PLANE LIFECYCLE
    # =====================================================

    def XPluginStart(self):
        parent = xp.findPluginsMenu()
        item = xp.appendMenuItem(parent, "AP1 Voice Control", 0)

        self.menu_id = xp.createMenu(
            "AP1 Voice Control",
            parent,
            item,
            self.menuHandler,
            0
        )

        self.menu_item = xp.appendMenuItem(self.menu_id, "Toggle: ON", 0)

        self.hk_press = xp.registerHotKey(
            xp.VK_Z, xp.DownFlag,
            "AP1 Voice Press",
            self.OnPress
        )

        self.hk_release = xp.registerHotKey(
            xp.VK_Z, xp.UpFlag,
            "AP1 Voice Release",
            self.OnRelease
        )

        self.detect_microphone()

        xp.log("[AP1 Voice] Loaded (OFF by default)")
        return self.Name, self.Sig, self.Desc

    def XPluginStop(self):
        if self.hk_press:
            xp.unregisterHotKey(self.hk_press)
        if self.hk_release:
            xp.unregisterHotKey(self.hk_release)

    def XPluginEnable(self):
        return 1

    def XPluginDisable(self):
        return 1

    # =====================================================
    # MENU HANDLER
    # =====================================================

    def menuHandler(self, menuRef, itemRef):
        self.enabled = not self.enabled

        if self.enabled:
            xp.setMenuItemName(self.menu_id, self.menu_item, "Toggle: OFF")
            xp.speakString("AP1 voice control enabled")
        else:
            xp.setMenuItemName(self.menu_id, self.menu_item, "Toggle: ON")
            xp.speakString("AP1 voice control disabled")

    # =====================================================
    # PUSH TO TALK
    # =====================================================

    def OnPress(self, refcon):
        if not self.enabled or self.isRecording:
            return

        self.isRecording = True
        self.start_time = time.time()
        xp.speakString("Listening")

        # Bluetooth wake-up
        time.sleep(0.3)

    def OnRelease(self, refcon):
        if not self.enabled or not self.isRecording:
            return

        self.isRecording = False
        xp.speakString("Processing")

        if self.device is None:
            xp.speakString("Microphone not available")
            return

        duration = time.time() - self.start_time
        duration = max(1.5, min(duration, 5.0))  # 🔴 enforce minimum

        try:
            audio = sd.rec(
                int(duration * self.input_rate),
                samplerate=self.input_rate,
                channels=self.channels,
                device=self.device,
                dtype="float32"
            )
            sd.wait()
        except Exception as e:
            xp.log(f"[AP1 Voice] Recording failed: {e}")
            xp.speakString("Microphone error")
            return

        # Stereo → mono
        if self.channels > 1:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.flatten()

        # Resample to 16 kHz
        if self.input_rate != self.sample_rate:
            target_len = int(len(audio) * self.sample_rate / self.input_rate)
            audio = resample(audio, target_len)

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio /= max_val

        rms = np.sqrt(np.mean(audio ** 2))
        xp.log(f"[AP1 Voice] Audio RMS: {rms:.5f}")

        # 🔴 Silence gate
        if rms < 0.008:
            xp.speakString("No speech detected")
            xp.log("[AP1 Voice] Below speech threshold")
            return

        result = self.model.transcribe(
            audio,
            language="en",
            fp16=True,
            temperature=0.2,          # 🔴 critical fix
            no_speech_threshold=0.4   # tolerate short commands
        )

        text = result["text"].strip().lower()
        xp.log(f"[AP1 Voice] Recognized: {text}")

        self.execute_command(text)

    # =====================================================
    # COMMAND EXECUTION
    # =====================================================

    def execute_command(self, text):
        if not text:
            xp.speakString("Command not recognized")
            return

        for phrase in self.ap1_engage_phrases:
            if phrase in text:
                xp.commandOnce(xp.findCommand("sim/autopilot/servos_on"))
                xp.speakString("Autopilot one engaged")
                return

        for phrase in self.ap1_disengage_phrases:
            if phrase in text:
                xp.commandOnce(xp.findCommand("sim/autopilot/servos_off"))
                xp.speakString("Autopilot one disengaged")
                return

        xp.speakString("Command not recognized")
