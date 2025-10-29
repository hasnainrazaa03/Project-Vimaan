import os
import torch
import speech_recognition as sr
from XPPython3 import xp
import logging
from datetime import datetime
import json

import sys
ml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML")
sys.path.insert(0, ml_path)

from core import normalize_aviation_input, JointIntentAndSlotModel, postprocess_slots
from utils import get_latest_model_path

class PythonInterface:
    
    def __init__(self):
        self.Name = "Vimaan AI CoPilot"
        self.Sig = "plugin.vimaan.aicopilot.bymhr"
        self.Desc = "Advanced Voice Command Interface with Intent & Slot Recognition for X-Plane"

        self._setup_custom_logging()
        
        self.hotkeyPress = None
        self.hotkeyRelease = None

        self.recognizer = sr.Recognizer()
        
        try:
            self.microphone = sr.Microphone()
        except AttributeError:
            self.microphone = None
            self.log("WARNING: PyAudio not available - microphone may not work")

        self.isRecording = False
        self.audioData = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log(f"[Vimaan] Using device: {self.device}")
        
        try:
            self._load_model()
            self.log("[Vimaan] Model loaded successfully!")
        except Exception as e:
            self.log(f"[Vimaan] ERROR loading model: {str(e)}")
            raise

        try:
            from transformers import DistilBertTokenizerFast
            model_path = get_latest_model_path()
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.log(f"[Vimaan] Tokenizer loaded successfully!")
        except Exception as e:
            self.log(f"[Vimaan] ERROR loading tokenizer: {str(e)}")
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

    def _setup_custom_logging(self):
        try:
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            self.log_folder = os.path.join(desktop_path, "Vimaan_Logs")
            
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(self.log_folder, f"vimaan_plugin_{timestamp}.log")
            
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_file, mode='w', encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            
            self.logger = logging.getLogger('VimaaNCoPilot')
            self.logger.info(f"=== VIMAAN AI COPILOT LOG START ===")
            self.logger.info(f"Log file: {self.log_file}")
            self.logger.info(f"Plugin directory: {os.path.dirname(__file__)}")
            self.log(f"[Vimaan] Custom log created: {self.log_file}")
            
        except Exception as e:
            self.log_file = None
            self.logger = None
            self.log(f"[Vimaan] Failed to setup custom logging: {str(e)}")

    def log(self, message):
        formatted_msg = f"[Vimaan] {message}"
        try:
            xp.log(formatted_msg)
        except:
            pass
        
        if self.logger:
            self.logger.info(message)


    def _load_model(self):
        model_path = get_latest_model_path()
        self.log(f"[Vimaan] Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(f"{model_path}/intent_map.json", "r") as f:
            self.intent_map = json.load(f)
        
        with open(f"{model_path}/slot_map.json", "r") as f:
            self.slot_map = json.load(f)
        
        self.intent_map_rev = {v: k for k, v in self.intent_map.items()}
        self.slot_map_rev = {v: k for k, v in self.slot_map.items()}
        
        num_intents = len(self.intent_map)
        num_slots = len(self.slot_map)
        
        self.log(f"[Vimaan] Loaded intent map: {num_intents} intents")
        self.log(f"[Vimaan] Loaded slot map: {num_slots} slots")
        
        self.model = JointIntentAndSlotModel(num_intents=num_intents, num_slots=num_slots)

        from transformers import DistilBertForTokenClassification
        self.model.bert_for_slots = DistilBertForTokenClassification.from_pretrained(model_path)
        self.log(f"[Vimaan] Loaded bert_for_slots from pretrained")
        
        intent_classifier_path = os.path.join(model_path, "intent_classifier.bin")
        if os.path.exists(intent_classifier_path):
            self.log(f"[Vimaan] Loading intent classifier")
            intent_classifier_state = torch.load(intent_classifier_path, map_location=self.device)
            self.model.intent_classifier.load_state_dict(intent_classifier_state)
            self.log(f"[Vimaan] Intent classifier loaded successfully")
        else:
            self.log(f"[Vimaan] WARNING: Intent classifier not found")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.log("[Vimaan] Model moved to device and set to eval mode")

    def _load_label_maps(self):
        self.log(f"[Vimaan] Label maps verified: {len(self.intent_map)} intents, {len(self.slot_map)} slots")

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

    def reconstruct_slot_value(self, tokens):
        if not tokens:
            return ""
        
        reconstructed = ""
        
        for i, token in enumerate(tokens):
            clean_token = token.replace("##", "")
            
            if clean_token == ".":
                reconstructed += clean_token
            elif token.startswith("##"):
                reconstructed += clean_token
            else:
                if i == 0:
                    reconstructed = clean_token
                else:
                    reconstructed += " " + clean_token
        
        return reconstructed.strip()

    def ExecuteCommand(self, text: str):
        try:
            text_normalized = normalize_aviation_input(text)
            self.log(f"[Vimaan] Normalized text: {text_normalized}")
            
            encoding = self.tokenizer(
                text_normalized,
                padding='max_length',
                truncation=True,
                max_length=64,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                _, intent_logits, slot_logits = self.model(input_ids, attention_mask)
            
            intent_pred_idx = torch.argmax(intent_logits, dim=1).item()
            intent_pred = self.intent_map_rev[intent_pred_idx]
            
            slot_pred_indices = torch.argmax(slot_logits, dim=2)[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            
            extracted_slots = {}
            current_slot_name = None
            current_slot_tokens = []
            
            for token, slot_idx in zip(tokens, slot_pred_indices):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
                
                slot_name_bio = self.slot_map_rev.get(int(slot_idx), 'O')
                
                if slot_name_bio.startswith("B-"):
                    if current_slot_name and current_slot_tokens:
                        extracted_slots[current_slot_name] = self.reconstruct_slot_value(current_slot_tokens)
                    
                    current_slot_name = slot_name_bio[2:]
                    current_slot_tokens = [token]
                
                elif slot_name_bio.startswith("I-") and current_slot_name:
                    slot_name_from_tag = slot_name_bio[2:]
                    if slot_name_from_tag == current_slot_name:
                        current_slot_tokens.append(token)
                
                else:
                    if current_slot_name and current_slot_tokens:
                        extracted_slots[current_slot_name] = self.reconstruct_slot_value(current_slot_tokens)
                        current_slot_name = None
                        current_slot_tokens = []
            
            if current_slot_name and current_slot_tokens:
                extracted_slots[current_slot_name] = self.reconstruct_slot_value(current_slot_tokens)
            
            slots = postprocess_slots(extracted_slots, text_normalized, intent_pred)
            
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