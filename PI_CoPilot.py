"""
Author:         Aryan Shukla
Plugin Name:    AI Co-Pilot
Tools Used:     Python 3.13.3, XPPython3 4.5.0
"""

from XPPython3 import xp  # type: ignore
import speech_recognition as sr
import threading

class PythonInterface:
    def __init__(self):
        self.Name = "AI CoPilot"
        self.Sig = "plugin004.aicopilot.byaryanshukla"
        self.Desc = "Voice command interface for X-Plane"

        self.parameters = {
            "GSD": "sim/cockpit/switches/gear_handle_status"
        }
        self.datarefs_pointer = {}

        self.hotkeyPress = None
        self.hotkeyRelease = None

        self.recognizer = sr.Recognizer()
        # self.microphone = sr.Microphone()
        self.isRecording = False
        self.audioData = None

    def XPluginStart(self):

        self.datarefs_pointer = {
            param: xp.findDataRef(dataref) for param, dataref in self.parameters.items()
        }

        self.hotkeyPress = xp.registerHotKey(
            xp.VK_Z,
            xp.DownFlag,
            "Push-to-Talk -> Press",
            self.OnPressCallback
        )
        # self.hotkeyRelease = xp.registerHotKey(
        #     xp.VK_Z,
        #     xp.UpFlag,
        #     "Push-to-Talk -> Release",
        #     self.OnReleaseCallback
        # )
        return self.Name, self.Sig, self.Desc
    
    def XPluginEnable(self): 
        return 1
    
    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        pass

    def XPluginStop(self):
        xp.unregisterHotKey(self.hotkeyPress)
        # xp.unregisterHotKey(self.hotkeyRelease)

    def XPluginDisable(self):
        pass

    def OnPressCallback(self, inRefcon):
        self.StartRecording()

    def OnReleaseCallback(self, inRefcon):
        xp.log(f"Stop recording...")
        # self.StopRecordingAndProcess()

    def StartRecording(self):
        def record():
            try:
                with sr.Microphone() as source:
                    xp.log("2Calibrating for ambient noise...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    xp.log("2Mic ready, listening now...")
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                    xp.log(f"2Finished listening. Captured {len(audio.get_raw_data())} bytes")

                    self.audio_data = audio

            except Exception as e:
                xp.log(f"Recording error: {e}")

        threading.Thread(target=record, daemon=True).start()

    # def StartRecording(self):
    #     xp.log(f"Start recording...")
    #     self.isRecording = True
    #     with self.microphone as source:
    #         xp.log("Calibrating for ambient noise...")
    #         self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
    #         self._audio_source = source

    #     def record():
    #         xp.log(f"Opening microphone...")
    #         xp.log(f"Mic ready, listening now...")
    #         self.audioData = self.recognizer.listen(self._audio_source, phrase_time_limit=5)
    #         xp.log(f"Finished listening. Captured {len(self.audioData.get_raw_data())} bytes")
    #     threading.Thread(target=record, daemon=True).start()

    # def StopRecordingAndProcess(self):
    #     xp.log(f"Stop recording...")
    #     self.isRecording = False
    #     if not self.audioData:
    #         xp.log(f"No audio captured.")
    #         return

    #     def process():
    #         try:
    #             command = self.recognizer.recognize_google(self.audioData).lower()
    #             xp.log(f"Recognized command: {command}")

    #             if "gear down" in command:
    #                 xp.setDatai(self.datarefs_pointer['GSD'], 1)
    #                 xp.speakString("Gear down")
    #             elif "gear up" in command:
    #                 xp.setDatai(self.datarefs_pointer['GSD'], 0)
    #                 xp.speakString("Gear up")
    #             else:
    #                 xp.speakString("Command not recognized")

    #         except Exception as e:
    #             xp.log(f"Speech recognition error: {e}")
    #             xp.speakString(f"Sorry, I did not understand")

    #     threading.Thread(target=process, daemon=True).start()