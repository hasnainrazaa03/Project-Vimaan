import os
from XPPython3 import xp  # type: ignore
import speech_recognition as sr

class PythonInterface:
    def __init__(self):
        self.Name = "AI CoPilot"
        self.Sig = "plugin004.aicopilot.byaryanshukla"
        self.Desc = "Voice command interface for X-Plane"

        self.hotkeyPress = None
        self.hotkeyRelease = None

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.isRecording = False
        self.audioData = None

        self.command_map = {
            "gear up": ("command", "sim/flight_controls/landing_gear_up"),
            "gear down": ("command", "sim/flight_controls/landing_gear_down"),
            "flaps down": ("command", "sim/flight_controls/flaps_down"),
            "flaps up": ("command", "sim/flight_controls/flaps_up")
        }

    def XPluginStart(self):

        self.hotkeyPress = xp.registerHotKey(
            xp.VK_Z,
            xp.DownFlag,
            "Push-to-Talk -> Press",
            self.OnPressCallback
        )
        self.hotkeyRelease = xp.registerHotKey(
            xp.VK_Z,
            xp.UpFlag,
            "Push-to-Talk -> Release",
            self.OnReleaseCallback
        )

        return self.Name, self.Sig, self.Desc

    def XPluginEnable(self): 
        return 1

    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        pass

    def XPluginStop(self):
        xp.unregisterHotKey(self.hotkeyPress)
        xp.unregisterHotKey(self.hotkeyRelease)

    def XPluginDisable(self):
        pass

    def OnPressCallback(self, inRefcon):
        if not self.isRecording:
            xp.speakString("Listening")
            self.isRecording = True
            self.source = self.microphone.__enter__()
            self.recognizer.adjust_for_ambient_noise(self.source, duration=0.5)
            self.audioData = self.recognizer.listen(self.source, timeout=None, phrase_time_limit=None)

    def OnReleaseCallback(self, inRefcon):
        if self.isRecording:
            xp.speakString("Processing")
            self.isRecording = False
            self.microphone.__exit__(None, None, None)

            try:
                text = self.recognizer.recognize_google(self.audioData).lower()
                xp.log(f"[AI CoPilot] Recognized: {text}")
                self.ExecuteCommand(text)
            except sr.UnknownValueError:
                xp.speakString("I could not understand you")
            except sr.RequestError:
                xp.speakString("Recognition service failed")

    def ExecuteCommand(self, text: str):
        for phrase, action in self.command_map.items():
            if phrase in text:
                cmd_ref = xp.findCommand(action[1])
                xp.commandOnce(cmd_ref)
                xp.speakString(f"Executing {phrase}")
                return

        xp.speakString("Command not recognized")
