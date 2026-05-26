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
            self.audioData = self.recognizer.listen(
                self.source,
                timeout=None,
                phrase_time_limit=None
            )

    def OnReleaseCallback(self, inRefcon):
        if self.isRecording:
            xp.speakString("Processing")
            self.isRecording = False
            self.microphone.__exit__(None, None, None)

            try:
                text = self.recognizer.recognize_google(self.audioData).upper()
                self.ExecuteCommand(text)
            except sr.UnknownValueError:
                xp.speakString("i could not understand you")
            except sr.RequestError:
                xp.speakString("recognition service failed")

    def ExecuteCommand(self, text: str):
        ap1_engage_phrases = {
            "ap1 on",
            "engage ap1",
            "activate ap1",
            "enable ap1",
            "autopilot one on",
            "autopilot on"
        }

        ap1_disengage_phrases = {
            "ap1 off",
            "disengage ap1",
            "deactivate ap1",
            "disable ap1",
            "autopilot one off",
            "autopilot off"
        }

        for phrase in ap1_engage_phrases:
            if phrase in text:
                xp.commandOnce(xp.findCommand("sim/autopilot/servos_on"))
                xp.speakString("autopilot one engaged")
                return

        for phrase in ap1_disengage_phrases:
            if phrase in text:
                xp.commandOnce(xp.findCommand("sim/autopilot/servos_toggle"))
                xp.speakString("autopilot one disengaged")
                return

        xp.speakString("command not recognized")
