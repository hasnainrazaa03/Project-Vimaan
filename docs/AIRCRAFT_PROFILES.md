# Aircraft profiles and the probe

How Vimaan works out what the loaded aircraft can do. The code is in
`ML/vimaan_nlu/aircraft.py` (the profile), `ML/vimaan_nlu/probe.py` (the
probe's rules) and the plugin (reading the sim). This file is the research
record behind them (Tier 1, step A1) and the instructions for the probe.

## The principle: nothing is keyed to an aircraft's name

A profile is resolved in layers, from most general to most specific. An
aircraft no one has seen before still gets layers 1, 2 and 5 automatically,
and layer 4 after one probe.

| Layer | Source | Example |
| --- | --- | --- |
| 1. Standard controls | X-Plane's `sim/...` commands and datarefs | `sim/autopilot/heading` for heading mode |
| 2. What X-Plane reports | datarefs read when the aircraft loads | `acf_en_type` = 7 (jet) gives a turbine start |
| 3. Overrides | used only if the aircraft *has* the custom control | `laminar/A333/autopilot/a_thr_toggle` exists, so A/THR presses it |
| 4. Probe results | measured on this aircraft, saved per aircraft file | V/S mode "no change" means "not on this aircraft" |
| 5. Verification | read-back after every action | "Autopilot 1 did not respond" |

**Why overrides aren't matched on ICAO or author.** Laminar's 737 lists
"Alex Unruh" as author, and its A330 lists five individuals, not "Laminar
Research". Add-ons reuse ICAO codes too: the Zibo 737 is also `B738`. A
control's existence is what decides, because that is what the action needs.

## Layer 2: what X-Plane reports (read on every aircraft load)

| Field | Dataref | Used for |
| --- | --- | --- |
| Engine count, type per engine | `sim/aircraft/engine/acf_num_engines`, `sim/aircraft/prop/acf_en_type` (0/1 piston, 3 electric, 5/7 jet, 9/10 turboprop) | engine start sequence |
| Flap detents, angles | `sim/aircraft/controls/acf_flap_detents`, `acf_flap_dn` | flap positions (Part B) |
| Retractable gear | `sim/aircraft/gear/acf_gear_retract` | "This aircraft has fixed gear" |
| Autopilot type | `sim/aircraft/autopilot/preconfigured_ap_type`, `single_axis_autopilot` | logged; probe context |
| V-speeds | `sim/aircraft/view/acf_Vso`, `acf_Vfe`, `acf_Vno`, `acf_Vne`, `acf_Mmo` | speed and flap safety rules (Part B) |
| Identity | `acf_ICAO`, `acf_author`, `acf_descrip`, `xp.getNthAircraftModel(0)` | logs; the probe file's name |

Each load is logged as an `aircraft` event with the full profile.

**Flap angles are not handle labels.** `acf_flap_dn[0]` is 0 (up) and
`[1..detents]` are the modelled surface angles. The 737's are 3, 5, 8, 10, 12,
14, 18, 40, but its handle reads 1, 2, 5, 10, 15, 25, 30, 40: "flaps 15" is
notch 5, at 12°. Degrees can name a notch only where the labels *are* degrees
(Baron: 13, 30). Airliner labels need an override (Part B).

## Findings per aircraft (X-Plane 12.4.3, verified 2026-11-07)

Sources: `Resources/plugins/DataRefs.txt` and `Commands.txt`; each aircraft's
`.acf` and xlua scripts (`plugins/xlua/scripts/*/*.lua`); the POH PDFs in the
aircraft folders. `scripts/check_xplane_refs.py` re-checks every name.

**Beechcraft Baron 58** (`BE58`, piston ×2, flap detents 2)
- POH "Autopilot Operation" lists AP, HDG, ALT, NAV, BC, APPR, YAW and a DN/UP rocker.
- ALT "will level the aircraft at the current altitude, and hold this". In flight, pressing ALT replaced the selected altitude with the current one.
- No V/S button, no altitude preselect or capture, no autothrottle; the glideslope is not followed.
- The xlua scripts (`B58_*`) replace no autopilot commands.
- Unknown until probed: whether the generic V/S and altitude-arm commands do anything.

**Boeing 737-800** (`B738`, jet ×2, flap detents 8)
- The MCP uses the standard `sim/autopilot/...` commands; the scripts replace none.
- The baro has its own controls (`laminar/B738/EFIS_control/{capt,fo}/push_button/std_press`, `baro_in_hpa_up/dn`), and the scripts replace `sim/instruments/barometer_up/down`. Override `b738_efis_baro` covers STD.
- The fuel cutoff is the condition levers (`laminar/B738/engines/condition_lever_{1,2}_{idle,cutoff}`). Override `b738_condition_levers` covers a turbine start.
- The start switches are `laminar/B738/knob/starter{1,2}_{up,dn}`. Not used yet; the generic starter is tried first and logged.

**Airbus A330-300** (`A333`, jet ×2, flap detents 4)
- The scripts wrap `servos_toggle`/`servos2_toggle`, `altitude_hold`, `NAV`, `approach`, `knots_mach_toggle`, `barometer_std` and the up/down dials.
- In flight, `servos_on` and `servos2_on` were ignored. The fix: press the toggle once the state has been read, on every aircraft.
- The FCU knobs: `laminar/A333/autopilot/{speed,heading,altitude,vertical}_knob_{pull,push}`. Pull = selected, push = managed.
- A/THR: `laminar/A333/autopilot/a_thr_toggle`. Override `airbus_fcu` covers both.
- The ENG MODE selector is `laminar/A333/switch/eng_mode_{left,right}`. In flight, a restart cranked 20 s with the fuel lever at cutoff; the turbine start now opens it at 20% N2.

**Not yet checked, for the generalization test:** King Air C90B (`BE9L`,
turboprop) and Citation X (`C750`, jet). Neither has an override, so they
exercise layers 1, 2, 4 and 5 only.

## Engine start by type

| Type | Sequence | Limit | Failure message |
| --- | --- | --- | --- |
| Piston | starter held until running | 20 s | "did not start. Check fuel, mixture and ignition." |
| Jet | starter; fuel opens at N2 20% (the aircraft's own lever command if it has one, else `mixture_ratio` = 1); starter held until running and N2 ≥ 50% | 60 s | "did not spool up. N2 reached X percent. Check bleed air and the engine start switch." or "did not light ... Check fuel and ignition." |
| Turboprop | the same, on N1 (gas generator), fuel at 13% | 60 s | the same, with N1 |
| Electric | refused: "is electric and has no starter" | — | — |

Every step is logged: `crank_begin` (start type, fuel lever, core speed),
`fuel_on` (how it was opened), and `crank_end` (peak core speed, fuel on, result).

## Override and cancel

After a safety hold ("Confirm gear up, low altitude. Say override, or repeat
to confirm."):
- **override / confirm / affirmative** runs the held command. It skips only
  the rules overridden for it, so another rule that applies still holds it.
- **cancel / negative / disregard / abort** drops it. It also drops an open
  "to or two?" question.

Out-of-range values and commands the aircraft can't perform can't be
overridden: they are impossible, not risky.

## Running the probe

Bind two keys in X-Plane (Settings → Keyboard, search "Vimaan"):
**Vimaan: probe this aircraft (one step per press)** and **Vimaan: stop the
probe**. The probe is never started by voice, because it moves the
autopilot.

**On the ground** (3 steps: altimeter STD, altimeter setting, flap handle):
1. Parked, parking brake set, engines in any state.
2. Press the probe key once. Vimaan says "Probe ready. 3 ground steps…".
3. Press once per step. Each takes about 4 s and is put back afterwards.

**In the air** (15 steps: 10 autopilot modes, autothrottle, knots/Mach, speed / V/S / altitude dials):
1. Above 1,500 ft AGL, in smooth air, with the autopilot on in **HDG and ALT hold**.
2. Press once, then once per step. Each step presses one control for 2 s and puts it back. Expect brief mode changes, for example a few seconds in pitch mode.
3. If a step can't put things back, the probe stops and says "Check the autopilot". Fly normally and look at the mode lights.
4. You can stop at any time with the stop key. What was measured is saved.

Results are saved to `~/Vimaan_Logs/aircraft/<folder>__<file>.json`, with
the ground and air runs merged. They are used from the next aircraft load.
Delete the file to forget a probe.
