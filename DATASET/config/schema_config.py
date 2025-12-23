# --- SHARED VOCABULARY BANKS ---

SYN_ON = ["engage", "activate", "enable", "start", "turn on", "switch on", "power up"]
SYN_OFF = ["disengage", "deactivate", "disable", "stop", "kill", "turn off", "switch off", "shut down", "cut"]

SYN_UP = ["retract", "raise", "stow", "pull up"]
SYN_DOWN = ["extend", "lower", "deploy", "drop", "push down"]

SYN_SET = ["change", "make", "input", "select", "enter", "tune"]
SYN_CHECK = ["check", "verify", "confirm"]

# --- GENERATORS ---
def range_str(start, end, step=1):
    return [str(i) for i in range(start, end, step)]

# --- MAIN SCHEMA ---
SCHEMA = {
    
    # ==========================================
    # GROUP 1: AUTOPILOT & NAVIGATION (PRIMARY)
    # ==========================================

    # 1. Heading

    "set_autopilot_heading": {
        "templates": [
            "set heading {degrees}", 
            "turn to {degrees}", 
            "fly heading {degrees}", 
            "heading {degrees}"
        ],
        "slots": {
            "degrees": {
                "type": "numerical", 
                "values": range_str(0, 361)
                }
        }
    },

    # 2. Altitude and Flight Level

    "set_autopilot_altitude": {
        "templates": [
            "set altitude {altitude}", 
            "climb to {altitude}", 
            "descend to {altitude}", 
            "maintain {altitude} feet"
        ],
        "slots": {
            "altitude": {
                "type": "numerical", 
                "values": range_str(100, 45100, 100)
                }
        }
    },

    "set_flight_level": {
        "templates": [
            "climb to flight level {flight_level}", 
            "descend to flight level {flight_level}", 
            "maintain fl {flight_level}"
        ],
        "slots": {
            "flight_level": {
                "type": "numerical", 
                "values": range_str(50, 450, 10)
                }
        }
    },

    # 3. Vertical Speed

    "set_vertical_speed": {
        "templates": [
            "set vertical speed {value}", 
            "climb at {value} feet per minute", 
            "descend at {value} feet per minute", 
            "v s {value}"
        ],
        "slots": {
            "value": {
                "type": "numerical", 
                "values": range_str(100, 6001, 100)
                }
        }
    },

    # 4. Airspeed

    "set_airspeed": {
        "templates": [
            "set speed {knots}", 
            "speed {knots} knots", 
            "maintain {knots} knots", 
            "set airspeed {knots}"
        ],
        "slots": {
            "knots": {
                "type": "numerical", 
                "values": range_str(80, 360, 5)
                }
        }
    },

    # ==========================================
    # GROUP 2: FLIGHT CONTROL SURFACES
    # ==========================================
    
    # 5. Landing Gear
    
    "toggle_landing_gear": {
        "templates": [
            "gear {state}", 
            "{state} landing gear", 
            "landing gear {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["up", "down"], 
                "synonyms": {
                    "up": SYN_UP, 
                    "down": SYN_DOWN
                    }
                }
            }
    },

    # 6. Flaps

    "toggle_flaps": {
        "templates": [
            "flaps {state}", 
            "{state} flaps", 
            "set flaps {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["up", "down", "approach", "full"], 
                "synonyms": {
                    "up": SYN_UP, 
                    "down": SYN_DOWN, 
                    "approach": ["1", "5", "15"], 
                    "full": ["30", "40", "landing"]
                    }
            }
        }
    },

    # 7. Speed Brakes

    "toggle_speed_brakes": {
        "templates": [
            "speed brakes {state}", 
            "{state} speed brakes", 
            "spoilers {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["retracted", "extended", "armed"],
                "synonyms": {
                    "retracted": ["stow", "in", "down", "off"],
                    "extended": ["deploy", "out", "up", "full"],
                    "armed": ["arm", "automatic"]
                }
            }
        }
    },

    # 8. Parking Brakes

    "toggle_parking_brake": {
        "templates": [
            "parking brake {state}", 
            "{state} parking brake", 
            "set parking brake"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": ["set", "engage"], 
                    "off": ["release"]
                }
            }
        }
    },

    # 9. Trim

    "set_pitch_trim": {
        "templates": [
            "trim {state}", 
            "pitch trim {state}", 
            "set trim {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["neutral", "nose up", "nose down"],
                "synonyms": {
                    "neutral": ["reset", "zero", "center"], 
                    "nose up": ["up", "climb"], 
                    "nose down": ["down", "descend"]
                }
            }
        }
    },

    # ==========================================
    # GROUP 3: RADIOS & AVIONICS
    # ==========================================

    # 10. Radio COM

    "set_com_frequency": {
        "templates": [
            "set com {com_port} to {frequency}", 
            "tune com {com_port} {frequency}", 
            "radio {com_port} {frequency}"
            ],
        "slots": {
            "com_port": {
                "type": "categorical", 
                "values": ["1", "2"]},
            "frequency": {
                "type": "numerical", 
                "values": ["<DYNAMIC>"]
            }
        }
    },

    # 11. NAV

    "set_nav_frequency": {
        "templates": [
            "set nav {nav_port} to {frequency}", 
            "tune nav {nav_port} {frequency}", 
            "navigation {nav_port} {frequency}"
            ],
        "slots": {
            "nav_port": {
                "type": "categorical", 
                "values": ["1", "2"]},
            "frequency": {
                "type": "numerical", 
                "values": ["<DYNAMIC_NAV>"]
            }
        }
    },

    # 12. OBS / Course Selector

    "set_obs_course": {
        "templates": [
            "set course {degrees}", 
            "set obs {degrees}", 
            "course {degrees}"
            ],
        "slots": {
            "degrees": {
                "type": "numerical", 
                "values": range_str(0, 361)
            }
        }
    },

    # 13. ADF frequency

    "set_adf_frequency": {
        "templates": [
            "set adf to {frequency}", 
            "tune adf {frequency}", 
            "adf {frequency}"
            ],
        "slots": {
            "frequency": {
                "type": "numerical", 
                "values": ["<DYNAMIC_ADF>"]
            }
        }
    },

    # 14. Transponder

    "set_transponder": {
        "templates": [
            "set transponder {code}", 
            "squawk {code}", 
            "set squawk {code}"
            ],
        "slots": {
            "code": {
                "type": "numerical", 
                "values": ["<DYNAMIC_SQUAWK>"]
            }
        }
    },

    # 15. Barometer

    "set_barometer": {
        "templates": [
            "set altimeter {pressure}", 
            "qnh {pressure}", 
            "barometer {pressure}"
            ],
        "slots": {
            "pressure": {
                "type": "numerical", 
                "values": ["<DYNAMIC_BARO>"]
            }
        }
    },

    # ==========================================
    # GROUP 4: ELECTRICAL & STARTUP
    # ==========================================

    # 16. Master Battery

    "toggle_master_battery": {
        "templates": [
            "battery {state}", 
            "master battery {state}", 
            "turn {state} the battery"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 17. Avionics Master

    "toggle_avionics_master": {
        "templates": [
            "avionics {state}", 
            "avionics master {state}", 
            "radios {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 18. Magnetos

    "toggle_magnetos": {
        "templates": [
            "magnetos {state}", 
            "mags {state}", 
            "set magnetos {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["off", "left", "right", "both", "start"],
                "synonyms": {"both": ["on", "all"], "start": ["engage"]}
            }
        }
    },

    # 19. Fuel Pumps

    "toggle_fuel_pump": {
        "templates": [
            "fuel pump {state}", 
            "auxiliary fuel pump {state}", 
            "boost pump {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 20. Lights

    "toggle_lights": {
        "templates": [
            "{state} {light_type} lights", 
            "{light_type} lights {state}", 
            "set {light_type} to {state}"],
        "slots": {
            "light_type": {
                "type": "categorical", 
                "values": ["landing", "taxi", "strobe", "nav", "beacon", "panel", "logo", "wing"],
                "synonyms": {
                    "nav": ["navigation", "position"], 
                    "panel": ["instrument"]
                }
            },
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # ==========================================
    # GROUP 5: AIRLINER & ENVIRONMENT
    # ==========================================
    
    # 21. Auto Brake
    
    "set_auto_brake": {
        "templates": [
            "set autobrake {level}", 
            "autobrake {level}", 
            "autobrakes {level}"
            ],
        "slots": {
            "level": {
                "type": "categorical",
                "values": ["rto", "off", "1", "2", "3", "max"],
                "synonyms": {
                    "rto": ["rejected takeoff"], 
                    "max": ["maximum"]
                }
            }
        }
    },

    # 22. Seatbelt sign

    "toggle_seatbelt_sign": {
        "templates": [
            "seatbelt sign {state}", 
            "fasten seatbelts {state}", 
            "seatbelts {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off", "auto"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 23. De-icing

    "toggle_deice": {
        "templates": [
            "{system} de-ice {state}", 
            "{system} anti-ice {state}", 
            "pitot heat {state}"
            ],
        "slots": {
            "system": {
                "type": "categorical", 
                "values": ["wing", "engine", "pitot", "propeller", "all"], 
                "synonyms": {
                    "pitot": ["probe"]
                }
            },
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 24. Wipers

    "toggle_wipers": {
        "templates": [
            "wipers {state}", 
            "windshield wipers {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # ==========================================
    # GROUP 6: SWITCHES & ENGINES
    # ==========================================
    
    # 25. Auto-pilot
    
    "toggle_autopilot_master": {
        "templates": [
            "autopilot {state}", 
            "autopilot master {state}", 
            "ap {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 26. Flight Director

    "toggle_flight_director": { 
        "templates": [
            "flight director {state}", 
            "fd {state}"
            ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # 27. Engines

    "toggle_engine": {
        "templates": [
            "engine {engine_id} {state}", 
            "{state} engine {engine_id}"
            ],
        "slots": {
            "engine_id": {
                "type": "categorical", 
                "values": ["1", "2", "all"]
                },
            "state": {
                "type": "categorical", 
                "values": ["on", "off"], 
                "synonyms": {
                    "on": SYN_ON, 
                    "off": SYN_OFF
                }
            }
        }
    },

    # ==========================================
    # GROUP 7: CONVERSATIONAL & NULL
    # ==========================================
    
    "ask_time": {
        "templates": [
            "what time is it", 
            "current time", "time check"
            ], 
        "slots": {}
        },

    "ask_status_generic": {
        "templates": [
            "what's our status", 
            "report status", 
            "how are we doing"
            ], 
        "slots": {}
        },

    "chit_chat_greeting": {
        "templates": [
            "hello vimaan", 
            "hi copilot", 
            "good morning"
            ], 
        "slots": {}
        },

    "None": {
        "templates": [
            "what's the weather", 
            "make me a sandwich", 
            "open the pod bay doors",
            "call mom", 
            "play some music", 
            "how was your day", 
            "tell me a joke"
        ],
        "slots": {}
    }
}