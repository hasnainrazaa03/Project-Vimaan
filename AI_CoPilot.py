import os
import joblib
from XPPython3 import xp  # type: ignore
import speech_recognition as sr
from sentence_transformers import SentenceTransformer

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

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML", "ai_copilot.pkl")
        xp.log(f"path - {model_path}")

        self.classifier = joblib.load(model_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.intent_to_command = {
            "GEAR_UP": "sim/flight_controls/landing_gear_up",               # verified
            "GEAR_DOWN": "sim/flight_controls/landing_gear_down",           # verified
            "FLAPS_UP": "sim/flight_controls/flaps_up",                     # verified
            "FLAPS_DOWN": "sim/flight_controls/flaps_down",                 # verified
            "AUTOPILOT_1_ON": "sim/autopilot/servos_on",                    # verified
            "AUTOPILOT_1_OFF": "sim/autopilot/servos_toggle",               # verified
            "AUTOPILOT_2_ON": "sim/autopilot/servos2_on",                   # verified
            "AUTOPILOT_2_OFF": "sim/autopilot/servos2_toggle",              # verified
            "FLIGHT_DIRECTOR_1_ON": "sim/autopilot/fdir_on",                # verified
            "FLIGHT_DIRECTOR_1_OFF": "sim/autopilot/fdir_toggle",           # verified
            "FLIGHT_DIRECTOR_2_ON": "sim/autopilot/fdir2_on",               # verified
            "FLIGHT_DIRECTOR_2_OFF": "sim/autopilot/fdir2_toggle",          # verified
            "PARKING_BRAKE_ON": "sim/flight_controls/park_brake_set",       # verified
            "PARKING_BRAKE_OFF": "sim/flight_controls/park_brake_release",  # verified
            "ENGINE_1_ON": "sim/starters/engage_starter_1",                 # verified
            "ENGINE_1_OFF": "sim/starters/shut_down_1",                     # verified
            "ENGINE_2_ON": "sim/starters/engage_starter_2",                 # verified
            "ENGINE_2_OFF": "sim/starters/shut_down_2"                      # verified
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
                text = self.recognizer.recognize_google(self.audioData).upper()
                self.ExecuteCommand(text)
            except sr.UnknownValueError:
                xp.speakString("I could not understand you")
            except sr.RequestError:
                xp.speakString("Recognition service failed")

    def ExecuteCommand(self, text: str):
        embedding = self.embedding_model.encode([text])
        intent_idx = self.classifier.predict(embedding)[0]
        
        command_ref_name = self.intent_to_command.get(intent_idx)
        if command_ref_name:
            cmd_ref = xp.findCommand(command_ref_name)
            xp.commandOnce(cmd_ref)
            xp.log(f"[AI CoPilot] Recognized: {text} | Executing: {intent_idx}")
            xp.speakString(f"Executing {intent_idx}")
        else:
            xp.speakString("Command not recognized")
