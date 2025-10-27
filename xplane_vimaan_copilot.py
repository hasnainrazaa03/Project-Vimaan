import os
import torch
import speech_recognition as sr
from XPPython3 import xp

import sys
ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML")
sys.path.insert(0, ml_path)

from core import normalize_aviation_input, JointIntentAndSlotModel, postprocess_slots
from utils import get_latest_model_path


class VimaaNCoPilot:
    
    def __init__(self):
        self.Name = "Vimaan AI CoPilot"
        self.Sig = "plugin.vimaan.aicopilot.bymhr"
        self.Desc = "Advanced Voice Command Interface with Intent & Slot Recognition for X-Plane"

        self.hotkeyPress = None
        self.hotkeyRelease = None

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.isRecording = False
        self.audioData = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xp.log(f"[Vimaan] Using device: {self.device}")
        
        try:
            self._load_model()
            xp.log("[Vimaan] Model loaded successfully!")
        except Exception as e:
            xp.log(f"[Vimaan] ERROR loading model: {str(e)}")
            raise

        self.intent_to_command = {
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

        self._load_label_maps()

    def _load_model(self):
        model_path = get_latest_model_path()
        xp.log(f"[Vimaan] Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = JointIntentAndSlotModel(
            num_intents=22,
            num_slots=10,
            device=self.device
        )
        self.model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), 
                                               map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        xp.log("[Vimaan] Model loaded and set to eval mode")

    def _load_label_maps(self):
        self.intent_map = {
            0: "set_autopilot_heading",
            1: "set_autopilot_altitude",
            2: "set_flight_level",
            3: "toggle_landing_gear",
            4: "toggle_flaps",
            5: "toggle_autopilot_1",
            6: "toggle_autopilot_2",
            7: "toggle_flight_director_1",
            8: "toggle_flight_director_2",
            9: "toggle_parking_brake",
            10: "toggle_engine_1",
            11: "toggle_engine_2",
            12: "set_com_frequency",
            13: "ask_time",
            14: "ask_status",
            15: "chit_chat_greeting",
            16: "none",
            17: "none",
            18: "none",
            19: "none",
            20: "none",
            21: "none",
        }
        
        self.slot_map = {
            0: "degrees",
            1: "altitude",
            2: "flight_level",
            3: "state",
            4: "com_port",
            5: "frequency",
            6: "O",
            7: "O",
            8: "O",
            9: "O",
        }

    def XPluginStart(self):
        self.hotkeyPress = xp.registerHotKey(
            xp.VK_Z,
            xp.DownFlag,
            "Vimaan Push-to-Talk -> Press",
            self.OnPressCallback
        )
        self.hotkeyRelease = xp.registerHotKey(
            xp.VK_Z,
            xp.UpFlag,
            "Vimaan Push-to-Talk -> Release",
            self.OnReleaseCallback
        )
        xp.speakString("Vimaan AI CoPilot Ready")
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
            xp.log("[Vimaan] Recording started...")
            xp.speakString("Listening")
            self.isRecording = True
            try:
                self.source = self.microphone.__enter__()
                self.recognizer.adjust_for_ambient_noise(self.source, duration=0.5)
                self.audioData = self.recognizer.listen(self.source, timeout=None, phrase_time_limit=None)
            except Exception as e:
                xp.log(f"[Vimaan] Error during recording: {str(e)}")
                xp.speakString("Microphone error")

    def OnReleaseCallback(self, inRefcon):
        if self.isRecording:
            xp.log("[Vimaan] Recording stopped. Processing...")
            xp.speakString("Processing")
            self.isRecording = False
            self.microphone.__exit__(None, None, None)

            try:
                text = self.recognizer.recognize_google(self.audioData)
                xp.log(f"[Vimaan] Recognized text: {text}")
                self.ExecuteCommand(text)
            except sr.UnknownValueError:
                xp.log("[Vimaan] Speech not recognized")
                xp.speakString("I could not understand you")
            except sr.RequestError as e:
                xp.log(f"[Vimaan] Google Speech API error: {str(e)}")
                xp.speakString("Recognition service failed")
            except Exception as e:
                xp.log(f"[Vimaan] Unexpected error: {str(e)}")
                xp.speakString("An error occurred")

    def ExecuteCommand(self, text: str):
        try:
            normalized_text = normalize_aviation_input(text.lower())
            xp.log(f"[Vimaan] Normalized text: {normalized_text}")
            
            with torch.no_grad():
                from transformers import DistilBertTokenizerFast
                tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
                
                inputs = tokenizer(normalized_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                
                _, intent_logits, slot_logits = self.model(input_ids, attention_mask)
                
                intent_idx = torch.argmax(intent_logits, dim=1).item()
                intent_name = self.intent_map.get(intent_idx, "none")
                
                xp.log(f"[Vimaan] Predicted Intent: {intent_name} (confidence: {intent_logits[0, intent_idx].item():.2f})")
                
                if slot_logits is not None:
                    slot_predictions = torch.argmax(slot_logits, dim=2)
                    slots = postprocess_slots(slot_predictions[0].cpu().numpy(), self.slot_map)
                    xp.log(f"[Vimaan] Extracted Slots: {slots}")
                else:
                    slots = {}
            
            if intent_name in self.intent_to_command and intent_name != "none":
                handler = self.intent_to_command[intent_name]
                handler(slots)
                xp.log(f"[Vimaan] Command executed: {intent_name}")
            else:
                xp.log(f"[Vimaan] Intent '{intent_name}' not recognized")
                xp.speakString("Command not found")
                
        except Exception as e:
            xp.log(f"[Vimaan] Error executing command: {str(e)}")
            xp.speakString("Command execution failed")
    
    def _set_heading(self, slots):
        degrees = slots.get('degrees', None)
        if degrees:
            try:
                heading_value = float(degrees)
                xp.setDataf(xp.findDataRef("sim/cockpit/autopilot/heading_mag"), heading_value)
                xp.speakString(f"Setting heading to {degrees} degrees")
            except:
                xp.speakString("Invalid heading value")

    def _set_altitude(self, slots):
        altitude = slots.get('altitude', None)
        if altitude:
            try:
                altitude_value = float(altitude)
                xp.setDataf(xp.findDataRef("sim/cockpit/autopilot/altitude"), altitude_value)
                xp.speakString(f"Setting altitude to {altitude} feet")
            except:
                xp.speakString("Invalid altitude value")

    def _set_flight_level(self, slots):
        flight_level = slots.get('flight_level', None)
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
        frequency = slots.get('frequency', None)
        
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
