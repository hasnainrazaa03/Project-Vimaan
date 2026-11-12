"""simtest: drive Vimaan in a running X-Plane from the terminal (Phase A).

Talks to the plugin's test bridge (``ML/vimaan_nlu/sim_bridge.py``) through
files in ``~/Vimaan_Logs/test/``. The bridge does nothing until it's switched
on (``python -m simtest bridge on``), and the plugin says when it is.

    python -m simtest launch                        # X-Plane through Steam
    python -m simtest fly baron                     # in the air over Seattle, running
    python -m simtest bridge on
    python -m simtest status
    python -m simtest say "autopilot on"            # generated voice, through recognition
    python -m simtest say --text "autopilot on"     # straight to the model
    python -m simtest read sim/cockpit2/autopilot/servos_on
    python -m simtest voices "descend at one thousand five hundred feet per minute"
    python -m simtest quit

Development tool only: never shipped into X-Plane.
"""
