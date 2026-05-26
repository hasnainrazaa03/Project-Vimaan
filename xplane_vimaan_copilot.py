import os
import sys
import json
import torch
import logging
import speech_recognition as sr
from datetime import datetime
from XPPython3 import xp

ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML")
sys.path.insert(0, ml_path)

from core.model_loader import ModelLoader
from core.inference import predict


class PythonInterface:
    
    def __init__(self):
        self.Name = "Vimaan AI CoPilot"
        self.Sig = "plugin.vimaan.aicopilot.bymhr"
        self.Desc = "Advanced Voice Command Interface with Intent & Slot Recognition for X-Plane"
        
        self._setup_logging()
        
        self.recognizer = sr.Recognizer()
        self.microphone = self._setup_microphone()
        self.isRecording = False
        self.audioData = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"[Vimaan] Using device: {self.device}")
        
        self.loader = ModelLoader(self.device)
        self._init_model()
        
        self.hotkeyPress = None
        self.hotkeyRelease = None
        
        self.intent_to_command = self._setup_intent_handlers()
    
    def _setup_logging(self):
        try:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            log_folder = os.path.join(desktop_path, "Vimaan_Logs")
            
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_folder, f"vimaan_plugin_{timestamp}.log")
            
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger('VimaaNCoPilot')
            self.logger.info("=== VIMAAN AI COPILOT LOG START ===")
            self.logger.info(f"Log file: {log_file}")
            
        except Exception as e:
            self.logger = None
            self.log(f"[Vimaan] Failed to setup logging: {str(e)}")
    
    def log(self, message):
        try:
            xp.log(message)
        except:
            pass
        
        if self.logger:
            self.logger.info(message)
    
    def _setup_microphone(self):
        try:
            return sr.Microphone()
        except AttributeError:
            self.log("[Vimaan] WARNING: PyAudio not available - microphone may not work")
            return None
    
    def _init_model(self):
        try:
            results = self.loader.load_all()
            self.log(f"[Vimaan] Model loaded from: {results['model']['model_path']}")
            self.log(f"[Vimaan] Device: {results['model']['device']}")
            self.log(f"[Vimaan] Intents: {results['maps']['intents']}, Slots: {results['maps']['slots']}")
        except Exception as e:
            self.log(f"[Vimaan] ERROR loading model: {str(e)}")
            raise
    
    def _setup_intent_handlers(self):
        return {
            "set_autopilot_heading": self._set_heading,
            "set_autopilot_altitude": self._set_altitude,
            "set_flight_level": self._set_flight_level,
            "toggle_landing_gear": self._toggle_landing_gear,
            "toggle_flaps": self._toggle_flaps,
            "toggle_autopilot_1": self._toggle_autopilot_1,
            "toggle_autopilot_2": self._toggle_autopilot_2,
            "toggle_flight_director_1": self._toggle_flight_director_1,
            "toggle_flight_director_2": self._toggle_flight_director_2,
            "toggle_parking_brake": self._toggle_parking_brake,
            "toggle_engine_1": self._toggle_engine_1,
            "toggle_engine_2": self._toggle_engine_2,
            "set_com_frequency": self._set_com_frequency,
        }
    
    def XPluginStart(self):
        self.hotkeyPress = xp.registerHotKey(
            xp.VK_Z, xp.DownFlag,
            "Vimaan Push-to-Talk -> Press",
            self.OnPressCallback
        )
        self.hotkeyRelease = xp.registerHotKey(
            xp.VK_Z, xp.UpFlag,
            "Vimaan Push-to-Talk -> Release",
            self.OnReleaseCallback
        )
        xp.speakString("Vimaan AI CoPilot Ready")
        return self.Name, self.Sig, self.Desc
    
    def XPluginStop(self):
        if self.hotkeyPress:
            xp.unregisterHotKey(self.hotkeyPress)
        if self.hotkeyRelease:
            xp.unregisterHotKey(self.hotkeyRelease)
    
    def XPluginEnable(self):
        return 1
    
    def XPluginDisable(self):
        pass
    
    def XPluginReceiveMessage(self, inFromWho, inMessage, inParam):
        pass
    
    def OnPressCallback(self, inRefcon):
        if not self.isRecording:
            if self.microphone is None:
                self.log("[Vimaan] Microphone not available")
                xp.speakString("Microphone not available")
                return
            
            self.log("[Vimaan] Recording started...")
            xp.speakString("Listening")
            self.isRecording = True
            
            try:
                self.source = self.microphone.__enter__()
                self.recognizer.adjust_for_ambient_noise(self.source, duration=0.5)
                self.audioData = self.recognizer.listen(self.source, timeout=None, phrase_time_limit=None)
            except Exception as e:
                self.log(f"[Vimaan] Error during recording: {str(e)}")
                xp.speakString("Microphone error")
    
    def OnReleaseCallback(self, inRefcon):
        if self.isRecording:
            self.log("[Vimaan] Recording stopped. Processing...")
            xp.speakString("Processing")
            self.isRecording = False
            self.microphone.__exit__(None, None, None)
            
            try:
                text = self.recognizer.recognize_google(self.audioData)
                self.log(f"[Vimaan] Recognized text: {text}")
                self.ExecuteCommand(text)
            except sr.UnknownValueError:
                self.log("[Vimaan] Speech not recognized")
                xp.speakString("I could not understand you")
            except sr.RequestError as e:
                self.log(f"[Vimaan] Google Speech API error: {str(e)}")
                xp.speakString("Recognition service failed")
            except Exception as e:
                self.log(f"[Vimaan] Unexpected error: {str(e)}")
                xp.speakString("An error occurred")
    
    def ExecuteCommand(self, text: str):
        try:
            result = predict(
                text,
                self.loader.model,
                self.loader.tokenizer,
                self.device,
                self.loader.intent_map_rev,
                self.loader.slot_map_rev
            )
            
            intent_pred = result['intent']
            slots = result['slots']
            
            self.log(f"[Vimaan] Predicted Intent: {intent_pred}")
            self.log(f"[Vimaan] Extracted Slots: {slots}")
            
            if intent_pred in self.intent_to_command and intent_pred != "None":
                handler = self.intent_to_command[intent_pred]
                handler(slots)
                self.log(f"[Vimaan] Command executed: {intent_pred}")
                xp.speakString("Command executed")
            else:
                self.log(f"[Vimaan] Intent '{intent_pred}' not recognized")
                xp.speakString("Command not found")
                
        except Exception as e:
            self.log(f"[Vimaan] Error executing command: {str(e)}")
            import traceback
            self.log(f"[Vimaan] Traceback: {traceback.format_exc()}")
            xp.speakString("Command execution failed")
    
    def _set_heading(self, slots):
        degrees = slots.get('degrees')
        if degrees:
            try:
                heading_value = float(degrees)
                xp.setDataf(xp.findDataRef("sim/cockpit/autopilot/heading_mag"), heading_value)
                xp.speakString(f"Setting heading to {degrees} degrees")
            except:
                xp.speakString("Invalid heading value")
    
    def _set_altitude(self, slots):
        altitude = slots.get('altitude')
        if altitude:
            try:
                altitude_value = float(altitude)
                xp.setDataf(xp.findDataRef("sim/cockpit/autopilot/altitude"), altitude_value)
                xp.speakString(f"Setting altitude to {altitude} feet")
            except:
                xp.speakString("Invalid altitude value")
    
    def _set_flight_level(self, slots):
        flight_level = slots.get('flight_level')
        if flight_level:
            try:
                fl_value = float(flight_level) * 100
                xp.setDataf(xp.findDataRef("sim/cockpit/autopilot/altitude"), fl_value)
                xp.speakString(f"Setting flight level {flight_level}")
            except:
                xp.speakString("Invalid flight level")
    
    def _toggle_landing_gear(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'up':
            xp.commandOnce(xp.findCommand("sim/flight_controls/landing_gear_up"))
            xp.speakString("Gear up")
        elif state == 'down':
            xp.commandOnce(xp.findCommand("sim/flight_controls/landing_gear_down"))
            xp.speakString("Gear down")
        else:
            xp.commandOnce(xp.findCommand("sim/flight_controls/landing_gear_toggle"))
    
    def _toggle_flaps(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'up':
            xp.commandOnce(xp.findCommand("sim/flight_controls/flaps_up"))
            xp.speakString("Flaps up")
        elif state == 'down':
            xp.commandOnce(xp.findCommand("sim/flight_controls/flaps_down"))
            xp.speakString("Flaps down")
    
    def _toggle_autopilot_1(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/autopilot/servos_on"))
            xp.speakString("Autopilot 1 on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/autopilot/servos_toggle"))
            xp.speakString("Autopilot 1 off")
    
    def _toggle_autopilot_2(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/autopilot/servos2_on"))
            xp.speakString("Autopilot 2 on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/autopilot/servos2_toggle"))
            xp.speakString("Autopilot 2 off")
    
    def _toggle_flight_director_1(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/autopilot/fdir_on"))
            xp.speakString("Flight Director 1 on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/autopilot/fdir_toggle"))
            xp.speakString("Flight Director 1 off")
    
    def _toggle_flight_director_2(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/autopilot/fdir2_on"))
            xp.speakString("Flight Director 2 on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/autopilot/fdir2_toggle"))
            xp.speakString("Flight Director 2 off")
    
    def _toggle_parking_brake(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/flight_controls/park_brake_set"))
            xp.speakString("Parking brake on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/flight_controls/park_brake_release"))
            xp.speakString("Parking brake off")
    
    def _toggle_engine_1(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/starters/engage_starter_1"))
            xp.speakString("Engine 1 on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/starters/shut_down_1"))
            xp.speakString("Engine 1 off")
    
    def _toggle_engine_2(self, slots):
        state = slots.get('state', 'toggle')
        if state == 'on':
            xp.commandOnce(xp.findCommand("sim/starters/engage_starter_2"))
            xp.speakString("Engine 2 on")
        elif state == 'off':
            xp.commandOnce(xp.findCommand("sim/starters/shut_down_2"))
            xp.speakString("Engine 2 off")
    
    def _set_com_frequency(self, slots):
        com_port = slots.get('com_port', '1')
        frequency = slots.get('frequency')
        
        if frequency:
            try:
                freq_value = float(frequency)
                if com_port == '1':
                    xp.setDataf(xp.findDataRef("sim/cockpit/radios/com1_freq_hz"), freq_value * 1000000)
                    xp.speakString(f"COM 1 set to {frequency}")
                elif com_port == '2':
                    xp.setDataf(xp.findDataRef("sim/cockpit/radios/com2_freq_hz"), freq_value * 1000000)
                    xp.speakString(f"COM 2 set to {frequency}")
            except:
                xp.speakString("Invalid frequency")