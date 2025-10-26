SCHEMA = {
    "HEADING": {
        "base_cmds": [
            "heading {degrees}",
            "set heading {degrees}",
            "change heading to {degrees}",
            "turn to {degrees} degrees",
            "fly heading {degrees}"
        ],
        "slots": {
            "degrees": {
                "type": "numerical",
                "values": [str(i) for i in range(0, 361, 1)]
            }
        }
    },
    "ALTITUDE": {
        "base_cmds": [
            "altitude {altitude}",
            "set altitude {altitude}",
            "climb to {altitude} feet",
            "descend to {altitude} feet",
            "fly at {altitude}"
        ],
        "slots": {
            "altitude": {
                "type": "numerical",
                "values": [str(i) for i in range(100, 40001, 100)]
            }
        }
    },
    "FLIGHT_LEVEL": {
        "base_cmds": [
            "flight level {flight_level}",
            "climb to flight level {flight_level}",
            "maintain flight level {flight_level}",
            "request flight level {flight_level}"
        ],
        "slots": {
            "flight_level": {
                "type": "numerical",
                "values": [str(i) for i in range(100, 401, 10)]
            }
        }
    },
    "COM_FREQUENCY": {
        "base_cmds": [
            "com {com_port} to {frequency}",
            "set com {com_port} to {frequency}",
            "tune com {com_port} {frequency}",
            "frequency {frequency} on com {com_port}"
        ],
        "slots": {
            "com_port": {
                "type": "categorical",
                "values": ["1", "2"]
            },
            "frequency": {
                "type": "numerical",
                "values": ["<DYNAMIC>"]
            }
        }
    },
    "LANDING_GEAR": {
        "base_cmds": [
            "gear {state}",
            "{state} landing gear",
            "landing gear {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["up", "down"],
                "synonyms": {
                    "up": ["raise", "retract"], 
                    "down": ["lower", "extend", "deploy"]
                }
            }
        }
    },
    "FLAPS": {
        "base_cmds": [
            "flaps {state}", 
            "{state} flaps", 
            "set flaps {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["up", "down"],
                "synonyms": {
                    "up": ["retract", "raise"], 
                    "down": ["extend", "lower", "deploy"]
                }
            }
        }
    },
    "AUTOPILOT_1": {
        "base_cmds": [
            "autopilot 1 {state}", 
            "{state} autopilot 1", 
            "ap one {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["engage", "activate"], 
                    "off": ["disengage", "deactivate"]
                }
            }
        }
    },
    "AUTOPILOT_2": {
        "base_cmds": [
            "autopilot 2 {state}", 
            "{state} autopilot 2", 
            "ap two {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["engage", "activate"], 
                    "off": ["disengage", "deactivate"]
                }
            }
        }
    },
    "FLIGHT_DIRECTOR_1": {
        "base_cmds": [
            "flight director 1 {state}", 
            "{state} flight director 1", 
            "fd one {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["enable", "activate"], 
                    "off": ["disable", "deactivate"]
                }
            }
        }
    },
    "FLIGHT_DIRECTOR_2": {
        "base_cmds": [
            "flight director 2 {state}", 
            "{state} flight director 2", 
            "fd two {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["enable", "activate"], 
                    "off": ["disable", "deactivate"]
                }
            }
        }
    },
    "PARKING_BRAKE": {
        "base_cmds": [
            "parking brake {state}", 
            "{state} parking brake"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["engage", "set"], 
                    "off": ["release", "disengage"]
                }
            }
        }
    },
    "ENGINE_1": {
        "base_cmds": [
            "engine 1 {state}", 
            "{state} engine 1"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["start", "ignite"], 
                    "off": ["stop", "shut down", "kill"]
                }
            }
        }
    },
    "ENGINE_2": {
        "base_cmds": [
            "engine 2 {state}", 
            "{state} engine 2"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": ["on", "off"],
                "synonyms": {
                    "on": ["start", "ignite"], 
                    "off": ["stop", "shut down", "kill"]
                }
            }
        }
    }
}