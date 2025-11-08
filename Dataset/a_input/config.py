SCHEMA = {
    "LANDING_GEAR": {
        "description": "To Be Added",
        "command_templates": [
            "gear {state}",
            "landing gear {state}",
            "{state} gear",
            "{state} landing gear"
        ],
        "slots": {
            "state": {
                "type": "categorical",
                "values": {
                    "UP": ["up", "raise", "retract"],
                    "DOWN": ["down", "lower", "extend", "drop"]
                }
            }
        }
    },
    "FLAPS": {
        "description": "To Be Added",
        "command_templates": [
            "flaps {state}", 
            "{state} flaps", 
            "set flaps {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": {
                    "UP": ["up", "raise", "retract"],
                    "DOWN": ["down", "lower", "extend", "drop"]
                }
            }
        }
    },
    "AUTOPILOT_1": {
        "command_templates": [
            "autopilot 1 {state}", 
            "{state} autopilot 1", 
            "ap one {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": {
                    "ON": ["engage", "activate"], 
                    "OFF": ["disengage", "deactivate"]
                }
            }
        }
    },
    "AUTOPILOT_2": {
        "command_templates": [
            "autopilot 2 {state}", 
            "{state} autopilot 2", 
            "ap two {state}"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": {
                    "ON": ["engage", "activate"], 
                    "OFF": ["disengage", "deactivate"]
                }
            }
        }
    },
    # "FLIGHTDIRECTOR_1": {
    #     "command_templates": [
    #         "flight director 1 {state}", 
    #         "{state} flight director 1", 
    #         "fd one {state}"
    #     ],
    #     "slots": {
    #         "state": {
    #             "type": "categorical", 
    #             "values": {
    #                 "ON": ["enable", "activate"], 
    #                 "OFF": ["disable", "deactivate"]
    #             }
    #         }
    #     }
    # },
    # "FLIGHTDIRECTOR_2": {
    #     "command_templates": [
    #         "flight director 2 {state}", 
    #         "{state} flight director 2", 
    #         "fd two {state}"
    #     ],
    #     "slots": {
    #         "state": {
    #             "type": "categorical", 
    #             "values": {
    #                 "ON": ["enable", "activate"], 
    #                 "OFF": ["disable", "deactivate"]
    #             }
    #         }
    #     }
    # },
    # "PARKING_BRAKE": {
    #     "command_templates": [
    #         "parking brake {state}", 
    #         "{state} parking brake"
    #     ],
    #     "slots": {
    #         "state": {
    #             "type": "categorical", 
    #             "values": {
    #                 "ON": ["engage", "set"], 
    #                 "OFF": ["release", "disengage"]
    #             }
    #         }
    #     }
    # },
    "ENGINE_1": {
        "command_templates": [
            "engine 1 {state}", 
            "{state} engine 1"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": {
                    "ON": ["start", "ignite"], 
                    "OFF": ["stop", "shut down", "kill"]
                }
            }
        }
    },
    "ENGINE_2": {
        "command_templates": [
            "engine 2 {state}", 
            "{state} engine 2"
        ],
        "slots": {
            "state": {
                "type": "categorical", 
                "values": {
                    "ON": ["start", "ignite"], 
                    "OFF": ["stop", "shut down", "kill"]
                }
            }
        }
    },
    "HEADING": {
        "command_templates": [
            "set heading {state}",
            "change heading to {state}",
            "turn to {state} degrees",
            "fly heading {state}"
        ],
        "slots": {
            "state": {
                "type": "numerical",
                "values": {str(i) for i in range(0, 361, 1)}
            }
        }
    },
    # "ALTITUDE": {
    #     "command_templates": [
    #         "set altitude {state}",
    #         "climb to {state} feet",
    #         "descend to {state} feet",
    #         "fly at {state}"
    #     ],
    #     "slots": {
    #         "state": {
    #             "type": "numerical",
    #             "values": {str(i) for i in range(100, 40001, 100)}
    #         }
    #     }
    # },
#     "FLIGHT_LEVEL": {
#         "command_templates": [
#             "climb to flight level {state}",
#             "maintain flight level {state}",
#             "request flight level {state}"
#         ],
#         "slots": {
#             "state": {
#                 "type": "numerical",
#                 "values": {str(i) for i in range(100, 401, 10)}
#             }
#         }
#     }
}