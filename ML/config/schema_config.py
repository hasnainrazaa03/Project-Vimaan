# -----------------
# Clean Schema Configuration for Project Vimaan NLU Dataset Generation
# -----------------


# ============================================================================
# SYNONYMS
# ============================================================================


# Push/Pull actions for mode switching
SYN_PUSH_ACTIONS = ["push", "push to", "engage", "activate managed"]
SYN_PULL_ACTIONS = ["pull", "pull to", "disengage", "activate selected"]


# Mode activation verbs
SYN_MANAGE_ACTIONS = ["manage", "managed mode", "put in managed"]
SYN_SELECT_ACTIONS = ["select", "selected mode", "put in selected"]


# Control synonyms
SYN_SPD = ["SPD", "speed", "speed knob", "speed control"]
SYN_HDG = ["heading", "HDG", "heading knob", "heading control"]
SYN_ALT = ["altitude", "ALT", "altitude knob", "altitude control"]
SYN_VS = ["vertical speed", "VS", "VS knob", "V/S", "climb rate"]


# Brake control
SYN_BRAKE_ON = ["on", "engage", "set", "apply", "activate"]
SYN_BRAKE_OFF = ["off", "release", "disengage", "deactivate"]
SYN_PARKING_BRAKE = ["parking brake", "park brake", "brake", "park"]


# Flap positions
SYN_FLAP_ZERO = ["0", "zero", "up", "retracted"]
SYN_FLAP_ONE = ["1", "one"]
SYN_FLAP_TWO = ["2", "two"]
SYN_FLAP_THREE = ["3", "three"]
SYN_FLAP_FULL = ["full", "full flaps", "all the way"]


# Speed/heading values
SYN_SET = ["set", "change", "go to", "adjust to", "maintain"]
SYN_KNOTS = ["knots", "kts", "K", "knot"]
SYN_DEGREES = ["degrees", "degree" "degs", "deg"]


# ============================================================================
# SCHEMA
# ============================================================================


SCHEMA = {
    # ========================================================================
    # BOOLEAN MODES (push=managed, pull=selected)
    # ========================================================================
    
    "spd_mode": {
        "intent": "SPD_MODE",
        "type": "boolean_mode",
        "control": "SPD",
        "templates": [
            "{action} {control}",
            "{state} {control}",
            "{action} the {control}",
            "{control} {state}",
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["managed", "selected"]
            }
        },
        "placeholder_mapping": {
            "action": {
                "push": {
                    "synonyms": SYN_PUSH_ACTIONS,
                    "state": "managed"
                },
                "pull": {
                    "synonyms": SYN_PULL_ACTIONS,
                    "state": "selected"
                }
            },
            "control": {
                "synonyms": SYN_SPD
            },
            "state": {
                "managed": {
                    "synonyms": SYN_MANAGE_ACTIONS
                },
                "selected": {
                    "synonyms": SYN_SELECT_ACTIONS
                }
            }
        }
    },
    
    "hdg_mode": {
        "intent": "HDG_MODE",
        "type": "boolean_mode",
        "control": "HDG",
        "templates": [
            "{action} {control}",
            "{state} {control}",
            "{action} the {control}",
            "{control} {state}",
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["managed", "selected"]
            }
        },
        "placeholder_mapping": {
            "action": {
                "push": {
                    "synonyms": SYN_PUSH_ACTIONS,
                    "state": "managed"
                },
                "pull": {
                    "synonyms": SYN_PULL_ACTIONS,
                    "state": "selected"
                }
            },
            "control": {
                "synonyms": SYN_HDG
            },
            "state": {
                "managed": {
                    "synonyms": SYN_MANAGE_ACTIONS
                },
                "selected": {
                    "synonyms": SYN_SELECT_ACTIONS
                }
            }
        }
    },
    
    "alt_knob_mode": {
        "intent": "ALT_MODE",
        "type": "boolean_mode",
        "control": "ALT",
        "templates": [
            "{action} {control}",
            "{state} {control}",
            "{action} the {control}",
            "{control} {state}",
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["managed", "selected"]
            }
        },
        "placeholder_mapping": {
            "action": {
                "push": {
                    "synonyms": SYN_PUSH_ACTIONS,
                    "state": "managed"
                },
                "pull": {
                    "synonyms": SYN_PULL_ACTIONS,
                    "state": "selected"
                }
            },
            "control": {
                "synonyms": SYN_ALT
            },
            "state": {
                "managed": {
                    "synonyms": SYN_MANAGE_ACTIONS
                },
                "selected": {
                    "synonyms": SYN_SELECT_ACTIONS
                }
            }
        }
    },
    
    "vs_mode": {
        "intent": "VS_MODE",
        "type": "boolean_mode",
        "control": "VS",
        "templates": [
            "{action} {control}",
            "{state} {control}",
            "{action} the {control}",
            "{control} {state}",
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["managed", "selected"]
            }
        },
        "placeholder_mapping": {
            "action": {
                "push": {
                    "synonyms": SYN_PUSH_ACTIONS,
                    "state": "managed"
                },
                "pull": {
                    "synonyms": SYN_PULL_ACTIONS,
                    "state": "selected"
                }
            },
            "control": {
                "synonyms": SYN_VS
            },
            "state": {
                "managed": {
                    "synonyms": SYN_MANAGE_ACTIONS
                },
                "selected": {
                    "synonyms": SYN_SELECT_ACTIONS
                }
            }
        }
    },
    
    # ========================================================================
    # BINARY STATE (on/off)
    # ========================================================================
    
    "park_brake": {
        "intent": "PARK_BRAKE",
        "type": "binary_state",
        "control": "PARK_BRAKE",
        "templates": [
            "{action} {control}",
            "{control} {state}",
            "{state} {control}",
            "{action} the {control}",
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": ["on", "off"]
            }
        },
        "placeholder_mapping": {
            "action": {
                "on": {
                    "synonyms": SYN_BRAKE_ON,
                    "state": "on"
                },
                "off": {
                    "synonyms": SYN_BRAKE_OFF,
                    "state": "off"
                }
            },
            "control": {
                "synonyms": SYN_PARKING_BRAKE
            },
            "state": {
                "on": {
                    "synonyms": ["on", "engaged"]
                },
                "off": {
                    "synonyms": ["off", "released"]
                }
            }
        }
    },
    
    # ========================================================================
    # DISCRETE VALUE (specific positions)
    # ========================================================================
    
    "sflp": {
        "intent": "SFLP",
        "type": "discrete_value",
        "control": "flaps",
        "templates": [
            "{control} {position}",
            "set {control} to {position}",
            "{control} position {position}",
            "{position} {control}",
        ],
        "slots": {
            "position": {
                "type": "categorical",
                "values": ["0", "1", "2", "3", "full"]
            }
        },
        "placeholder_mapping": {
            "control": {
                "synonyms": ["flaps", "flap setting", "flap position", "flap"]
            },
            "position": {
                "0": {
                    "synonyms": SYN_FLAP_ZERO
                },
                "1": {
                    "synonyms": SYN_FLAP_ONE
                },
                "2": {
                    "synonyms": SYN_FLAP_TWO
                },
                "3": {
                    "synonyms": SYN_FLAP_THREE
                },
                "full": {
                    "synonyms": SYN_FLAP_FULL
                }
            }
        }
    },
    
    # ========================================================================
    # NUMERIC VALUES (ranges)
    # ========================================================================
    
    "spd_value": {
        "intent": "SPD_VALUE",
        "type": "numeric_value",
        "control": "speed",
        "templates": [
            "{action} {control} {value} {unit}",
            "{control} {value} {unit}",
            "{value} {unit} {control}",
            "{action} {value}",
        ],
        "slots": {
            "value": {
                "type": "numeric",
                "min": 100,
                "max": 350,
                "unit": "knots"
            }
        },
        "placeholder_mapping": {
            "action": {
                "synonyms": SYN_SET
            },
            "control": {
                "synonyms": SYN_SPD
            },
            "unit": {
                "synonyms": SYN_KNOTS
            }
        }
    },
    
    "hdg_value": {
        "intent": "HDG_VALUE",
        "type": "numeric_value",
        "control": "heading",
        "templates": [
            "{action} {control} {value} {unit}",
            "{control} {value} {unit}",
            "{value} {unit} {control}",
            "{action} {value}",
        ],
        "slots": {
            "value": {
                "type": "numeric",
                "min": 0,
                "max": 359,
                "unit": "degrees"
            }
        },
        "placeholder_mapping": {
            "action": {
                "synonyms": SYN_SET
            },
            "control": {
                "synonyms": SYN_HDG
            },
            "unit": {
                "synonyms": SYN_DEGREES
            }
        }
    },
}
