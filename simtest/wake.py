"""Phase L.5.3: the wake word in the sim, on each aircraft.

Tier 1's air start, hands off. Each step is heard on the open microphone, as
the wake word listens (the bridge's "hear"), never pressed on the talk key:

- in one breath ("hey copilot heading two seven zero") and after a pause
  ("hey copilot ... heading one eight zero": " ... " is a second's pause);
- the wake word alone: no command, nothing done;
- talk without it ("autopilot on, heading mode"; "copilot, check the flaps",
  "copilot" alone): no wake at all;
- a question after it: answered, nothing done.

``python -m simtest record wake all`` records your voice for each phrase.
"""

from __future__ import annotations

from .relative import START_WORDS
from .scenario import Step, equals
from .tier1 import SETUPS, THOUSANDS, hands_off

AP = "sim/cockpit2/autopilot/"
BUG = AP + "heading_dial_deg_mag_pilot"
ALT_BUG = AP + "altitude_dial_ft"
# A heading is "pending" on the A330 and the GFC-700 aircraft (their knob turns).
SET_OR_TURNING = ("executed", "no_op", "pending")
DONE_OR_SET = ("executed", "no_op")
# A switch is stepped over frames on some aircraft ("pending").
SWITCHED = ("executed", "no_op", "pending")
NO_CHANGE = {"no_actions": True, "vimaan_checks": False}
# Ordinary cockpit talk, never woken: with command words, a call to the
# tower, and "copilot" without "hey".
TALK = (
    ("command words", "autopilot on, heading mode, gear down"),
    ("a radio call", "tower, ready for departure, runway one zero right"),
    ('"copilot" alone', "copilot, check the flaps for me"),
    ("chat", "okay, autopilot is on and we are climbing"),
)


def steps(aircraft: str) -> list[Step]:
    s = SETUPS[aircraft]
    lower = s.alt_ft - 1000
    return hands_off(aircraft) + [
        Step(
            "one breath: heading 270",
            say="hey copilot heading two seven zero",
            wake=True,
            outcomes=SET_OR_TURNING,
            sim=(equals("heading bug 270", BUG, 270, tol=0.5, within_sec=10),),
        ),
        Step(
            f"one breath: altitude {lower}",
            say=f"hey copilot altitude {THOUSANDS[lower]}",
            wake=True,
            outcomes=DONE_OR_SET,
            sim=(equals(f"altitude bug {lower}", ALT_BUG, lower, tol=0.5),),
        ),
        Step(
            "one breath: landing lights on",
            say="hey copilot landing lights on",
            wake=True,
            outcomes=SWITCHED,
        ),
        Step(
            "after a pause: heading 180",
            say="hey copilot ... heading one eight zero",
            wake=True,
            outcomes=SET_OR_TURNING,
            sim=(equals("heading bug 180", BUG, 180, tol=0.5, within_sec=10),),
        ),
        Step(
            f"after a pause: altitude {s.alt_ft}",
            say=f"hey copilot ... altitude {START_WORDS[s.alt_ft]}",
            wake=True,
            outcomes=DONE_OR_SET,
            sim=(equals(f"altitude bug {s.alt_ft}", ALT_BUG, s.alt_ft, tol=0.5),),
        ),
        Step(
            "after a pause: landing lights off",
            say="hey copilot ... landing lights off",
            wake=True,
            outcomes=SWITCHED,
        ),
        Step(
            "the wake word alone: no command",
            say="hey copilot",
            wake=True,
            outcomes=("no_command",),
            sim=(equals("heading bug still 180", BUG, 180, tol=0.5),),
            **NO_CHANGE,
        ),
        *(
            Step(
                f"talk without it: {name}", say=said, wake=True, outcomes=("no_wake",), **NO_CHANGE
            )
            for name, said in TALK
        ),
        Step(
            "a question after it",
            say="hey copilot what's our altitude",
            wake=True,
            outcomes=("answered",),
            **NO_CHANGE,
        ),
    ]
