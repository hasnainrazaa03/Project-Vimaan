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
    }
}