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
| 3. Overrides | used only if the aircraft *has* the custom control, and its own scripts are loaded now | `laminar/A333/autopilot/a_thr_toggle` exists and `laminar/A333/autopilot/ap1_pos` is registered, so A/THR presses it |
| 4. Probe results | measured on this aircraft, saved per aircraft file | V/S mode "no change" means "not on this aircraft", for X-Plane's generic commands. The aircraft's own control (an override) measured as "no change" is still tried and verified: it exists, and did nothing in that moment |
| 5. Verification | read-back after every action | "Autopilot 1 did not respond" |

**Why overrides aren't matched on ICAO or author.** Laminar's 737 lists
"Alex Unruh" as author, and its A330 lists five individuals, not "Laminar
Research". Add-ons reuse ICAO codes too: the Zibo 737 is also `B738`. A
control's existence is what decides, because that is what the action needs.

**Why a live dataref too (2026-11-12).** An aircraft's custom *commands* stay
registered for the rest of the X-Plane session after it unloads; its
*datarefs* go with it. In the A2 run, an A330 loaded after the 737 picked up
the 737's overrides and Vimaan pressed its CMD A, so the A330's autopilot never
engaged. Each override now also names a dataref its aircraft's scripts create
(`create_dataref`): 737 `laminar/B738/autopilot/cmd_a_status`, A330
`laminar/A333/autopilot/ap1_pos`. It applies only while that is registered,
and the profile is looked at again 5 s and 15 s after a load, in case the
scripts register late. `scripts/check_xplane_refs.py` checks those names too.

## Layer 2: what X-Plane reports (read on every aircraft load)

| Field | Dataref | Used for |
| --- | --- | --- |
| Engine count, type per engine | `sim/aircraft/engine/acf_num_engines`, `sim/aircraft/prop/acf_en_type` (0/1 piston, 3 electric, 5/7 jet, 9/10 turboprop) | engine start sequence |
| Flap detents, angles | `sim/aircraft/controls/acf_flap_detents`, `acf_flap_dn` | flap positions (Part B) |
| Retractable gear | `sim/aircraft/gear/acf_gear_retract` | "This aircraft has fixed gear" |
| Autopilot type | `sim/aircraft/autopilot/preconfigured_ap_type`, `single_axis_autopilot` | logged; probe context |
| V-speeds | `sim/aircraft/view/acf_Vso`, `acf_Vfe`, `acf_Vno`, `acf_Vne`, `acf_Mmo` | speed and flap safety rules (Part B) |
| Flap speed limits | the `.acf`: `_Vfe1_kts` (first notch, which X-Plane publishes as `acf_Vfe`) and `_Vfem_kts` (full flaps, not published), read in the background at load | the flap-speed hold, per notch |
| Identity | `acf_ICAO`, `acf_author`, `acf_descrip`, `xp.getNthAircraftModel(0)` | logs; the probe file's name |
| Autothrottle in the cockpit | the aircraft's own files: `.acf`, then scripts, then objects (not liveries), searched for "autothrottle", in a background thread at load | refusing "autothrottle on" where the aircraft has none |

**Why the files, for autothrottle:** X-Plane's generic autothrottle runs on any
aircraft, so the probe said "works" on the Baron and King Air, which have none.
An aircraft that has one uses the sim's autothrottle datarefs or commands
somewhere in its panel, 3D cockpit or scripts. Checked on 12.4.3: the 737 and
A330 (in the .acf) and MD-82 (cockpit objects and scripts) do; the Baron, King
Air C90B, C172, SR22 and Citation X don't. An aircraft's own A/T control (an
override) always counts.

Each load is logged as an `aircraft` event with the full profile.

**Flap limits per notch (2026-11-12).** `acf_Vfe` was taken as the full-flap
limit, but it is the first notch's: the A2 run's 737 took full flaps at about
240 kt. The `.acf` has both:

| Aircraft | First notch `_Vfe1_kts` | Full `_Vfem_kts` |
| --- | --- | --- |
| Baron 58 | 152 | 122 |
| King Air C90B | 190 | 150 |
| 737-800 | 250 | 162 |
| A330-300 | 215 | 180 |

The hold uses the first-notch limit at notch 1, the full-flap limit at the
last, and evenly between (`aircraft.flap_limit_kt`). For the 737 that's 250,
237, 225, 212, 200, 187, 175, 162 kt for flaps 1 to 40, against a placard of
250, 250, 250, 210, 200, 190, 175, 162. Until the file is read, `acf_Vfe`.
"Override" still goes through.

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
- **A/T ARM switch:** X-Plane's own `sim/autopilot/autothrottle_arm` / `autothrottle_hard_off` (its cockpit object), state `autothrottle_enabled` (-1 disarmed, 0 armed). Its script's `laminar/B738/autopilot/autothrottle_arm_toggle` and `autothrottle_arm_pos` are commented out, so they don't exist in the sim. A first override used them; the name check matched the commented lines. With the switch off, "autothrottle on" did nothing (A2 runs, 2026-11-12). Now any aircraft whose autothrottle is hard off (-1, and "on" means holding a speed) is armed with `sim/autopilot/autothrottle_arm` first ("Arming the autothrottle"), then engaged half a second later. "Autothrottle off" leaves it armed. The A330 is excluded: its A/THR counts armed as on.

- **Probed 2026-11-07**, all 18 steps (sessions `vimaan_20261107_211836` and `_224940`):
  - **Works:** HDG, VOR/LOC, APP, back course, ALT HOLD, ALT ARM, V/S, LVL CHG, autothrottle, knots/Mach, the speed, V/S and altitude dials, and the flap handle.
  - **No change:** LNAV and VNAV (`lnav_press`, `vnav_press`). Both need an FMC flight plan to engage; whether one was loaded isn't logged.
  - **Altimeter writes don't hold:** its scripts own the baro. Vimaan turns the knob instead (`sim/instruments/barometer_up/down`, which the 737's scripts handle): one click per frame, reading back between clicks. STD was inconclusive: the altimeter was already 29.92.
  - **Side effects the probe now handles:** LVL CHG turns the autothrottle on, and the probe turns it back off. Moving the altitude dial arms altitude capture and leaves it armed; the probe notes that and carries on.
  - Engaged with no modes (roll and pitch hold), ALT HOLD couldn't be put back: there's no button for pitch hold. The probe now asks for HDG and ALT first.
- **Its MCP buttons** (`laminar/B738/autopilot/cmd_a_press`, `hdg_sel_press`, `vorloc_press`, `app_press`, `alt_hld_press`, `vs_press`, `lvl_chg_press`, `lnav_press`, `vnav_press`) call the same sim commands, and also update the 737's own `CMD_status` / `AP_status`, which drive its CMD light and annunciators.
  - With the generic engage, its `cmd_a_status` read off while the servos flew.
  - Override `b738_mcp` makes "autopilot on" and the probe's mode steps use those buttons.
- **In the first flight the autopilot disconnected ten times**, each time 1–50 s after engaging, with no modes selected (roll and pitch hold), and with the flight director left on. That's how X-Plane's airliner autopilot disconnects on control-column input.
  - Neither Vimaan nor the 737's scripts switched it off.
  - X-Plane's log showed two Xbox controllers registered as joysticks, unconfigured, with every null zone at 0%. A thumbstick at rest is never exactly centred.
  - The Baron's simpler autopilot stayed engaged through its whole probe.
  - In the second flight, with the 737's own CMD A, it never disconnected.

**Airbus A330-300** (`A333`, jet ×2, flap detents 4)
- The scripts wrap `servos_toggle`/`servos2_toggle`, `altitude_hold`, `NAV`, `approach`, `knots_mach_toggle`, `barometer_std` and the up/down dials.
- In flight, `servos_on` and `servos2_on` were ignored. The fix: press the toggle once the state has been read, on every aircraft.
- The FCU knobs: `laminar/A333/autopilot/{speed,heading,altitude,vertical}_knob_{pull,push}`. Pull = selected, push = managed. Heading push engages GPS steering (managed NAV); altitude push engages `sim/autopilot/FMS` (managed vertical), read from `fms_vnav`. Override `airbus_fcu` maps LNAV and VNAV to them.
- A/THR: `laminar/A333/autopilot/a_thr_toggle`. Override `airbus_fcu` covers both.
- The ENG MODE selector is `laminar/A333/switch/eng_mode_{left,right}`. In flight, a restart cranked 20 s with the fuel lever at cutoff; the turbine start now opens it at 20% N2.

- **Probed 2026-11-07**, all 18 steps (session `vimaan_20261107_231419`):
  - **Works:** HDG, NAV, APP, back course, ALT HOLD, ALT ARM, V/S, LVL CHG, knots/Mach, the speed, V/S and altitude dials, STD, the altimeter setting, and the flap handle.
  - **No change:** GPS steering and VNAV. Managed lateral and vertical guidance come from the FMS.
  - **No change: A/THR.** `a_thr_toggle` flips `autothrottle_enabled` between -1 (hard off) and 0 (armed). But `A333_autothrust()` holds it at 0 every frame while both engines' `ENGN_fadec_pow_req` is 2 or more (thrust levers at FLX/MCT or TOGA). Re-test it with the levers in CL. Vimaan counts armed as on, and an "autothrottle off" that doesn't take says to move the levers to CL.
  - The first air run started in TRK (lateral mode 18) and TO/GA (vertical mode 10). No button puts either back, and the probe warned about it.
  - **The autopilot disconnected about nine times.** Vimaan pressed nothing at five of those moments, and no A330 script turns the servos off, so these were X-Plane's own disconnects on control input. Two controllers, the USB Xbox controller and a "Virtual" Xbox 360 one, were both mapped to pitch, roll and yaw, with null zones at 0%.
  - A pause left APP armed, and the next step (back course) couldn't restore. A pause now puts the pressed control back first.

**Beechcraft King Air C90B** (`BE9L`, turboprop ×2, flap detents 2, Rockwell/Collins FCS-65, no overrides)
- **Probed 2026-11-07**, all 18 steps in one run with no disconnect (session `vimaan_20261107_232456`). It exercised layers 1, 2, 4 and 5 only.
  - **Works:** HDG, NAV, APP, back course, GPS steering, ALT HOLD, ALT ARM, V/S, LVL CHG (the FCS-65's IAS mode), the three dials, the altimeter setting, and the flap handle.
  - **No change:** VNAV.
  - **"Works", but it's X-Plane, not the aircraft:** the autothrottle and knots/Mach. X-Plane's own autothrottle runs on any aircraft. The C90B has none in its cockpit: its `.acf` references no autothrottle dataref, while the 737's and the A330's do. Whether Vimaan should allow "autothrottle on" here is an open Part B decision.
  - STD was inconclusive: the altimeter was already at 29.92.

**Not yet checked:** the Citation X (`C750`, jet, no override).

## Tier 2 findings: radios and transponder (X-Plane 12.4.3, 2026-11-12)

From each aircraft's xlua scripts, comment lines skipped (the A/T ARM lesson
above). Not yet checked in the sim: that's the Tier 2 runner's job.

X-Plane's own names, all in DataRefs.txt / Commands.txt:
`radios/actuators/com{1,2}_standby_frequency_hz_833` (kHz),
`nav{1,2}_frequency_hz` and `nav{1,2}_standby_frequency_hz` (10 kHz:
110.50 MHz is 11050), `nav{1,2}_obs_deg_mag_pilot`, `transponder_code`
(the code's digits as a number: 0120 is 120), `transponder_mode` (off 0,
standby 1, on/mode A 2, alt/mode C 3, test 4, ground 5, TA only 6, TA/RA 7),
`radios/indicators/transponder_id` (1 while identing); commands
`sim/radios/{com,nav}{1,2}_standy_flip` (X-Plane's spelling) and
`sim/transponder/transponder_{off,standby,on,alt,ident}`.

| Aircraft | Radios and course | Transponder | Override |
| --- | --- | --- | --- |
| Baron 58 | only reads the NAV 1/2 frequencies (for its DME) | standard | none |
| King Air C90B | writes the OBS only in its course-sync handler; its flip buttons call X-Plane's | replaces `sim/transponder/transponder_up/dn` with its knob, which writes the mode only on those presses, so the generic mode commands hold | none |
| 737-800 | its NAV transfer buttons call `sim/radios/nav{N}_standy_flip`; writes no frequency | the mode is set only by its selector, `laminar/B738/knob/transponder_mode_up/dn`, position `laminar/B738/knob/transponder_pos`: 0 TEST, 1 STBY, 2 ALT RPTG OFF, 3 XPNDR, 4 TA ONLY, 5 TA/RA. X-Plane's mode reads 2 for all of 2–5, so the selector, not the mode, is checked. No off. Ident `laminar/B738/push_button/transponder_ident_dn` (calls X-Plane's). The code is not written by its scripts | `b738_transponder` |
| A330-300 | its COM flip calls X-Plane's | `A333.transponder` recomputes X-Plane's mode every frame from three selectors, so a written mode doesn't hold: `auto_on_knob_pos` (-1 STBY, 0 AUTO, 1 ON; `auto_on_off_left/right`), `alt_rpt_knob_pos` (0/1; `alt_rpt_off/on`), `ta_ra_knob_pos` (TCAS 0 STBY, 1 TA, 2 TA/RA; `ta_ra_left/right`), plus AC power, gear and airspeed. TCAS STBY also leaves the transponder in standby. AUTO replies only once rolling. No off. It wraps X-Plane's ident and code digits; the code is written only at quick start, so a write holds | `a333_transponder` |

**Turning selectors** (`aircraft.Knob`): one click, then wait for the
position to move before the next, so a click the aircraft handles a frame late
can't overshoot; at most 12 clicks or 10 s, and "Transponder not set: the
selector did not move" if 30 frames pass without a change. A target is a
range, so a selector already in it is left alone: on the 737 "transponder on"
accepts ALT RPTG OFF through TA/RA, "alt" accepts XPNDR through TA/RA; on the
A330 "on" is AUTO or ON with TCAS TA or TA/RA, "alt" adds altitude reporting.

**"On" means replying.** On any aircraft, a transponder already reporting
altitude (alt, TA, TA/RA) answers "Transponder already on" instead of being
turned down to mode A. Ident is refused in standby ("Say transponder on
first").

## Tier 3 findings: switches (X-Plane 12.4.3, 2026-11-14)

From each aircraft's cockpit `.obj` (which dataref animates a switch, which
command clicks it) and its xlua scripts, comment lines skipped. Not yet
checked in the sim. Everything an aircraft's own scripts create with
`create_dataref` and no handler is read-only to other plugins (xlua
`init.lua`), so its switch positions are read and its commands clicked.

X-Plane's own names, all in DataRefs.txt / Commands.txt: lights
`sim/lights/{landing,taxi,beacon,strobe,nav}_lights_on/_off` (and
`landing_01/02_light_on/off`) and `cockpit2/switches/landing_lights_on`,
`taxi_light_on`, `beacon_on`, `strobe_lights_on`, `navigation_lights_on`,
`landing_lights_switch[16]`, `generic_lights_switch[128]`; signs
`sim/systems/seatbelt_sign_toggle`, `no_smoking_toggle` (toggles only);
`yaw_damper_on/off`; `controls/speedbrake_ratio` (-0.5 armed, 0 retracted,
1 fully extended) and `aircraft/parts/acf_sbrkEQ` (has speed brakes: 0 on the
Baron and King Air); `switches/auto_brake_level` (0 RTO, 1 off, 2-5 levels).
No dataref says whether an aircraft has an autobrake: X-Plane's is offered on
jets only.

**Part A (lights, signs, yaw damper, speed brake, autobrake)**, as built in
`vimaan_nlu/switches.py`:

| Aircraft | Lights | Signs | Yaw damper | Speed brake | Autobrake | Override |
| --- | --- | --- | --- | --- | --- | --- |
| Baron 58 | X-Plane's; landing L/R are `landing_01/02` (`landing_lights_on` is only the first); "wing" is the wing-ice light, `generic_lights_switch[0]` | none | X-Plane's (its flight start turns it off once) | none | none | `b58_switches` |
| King Air C90B | X-Plane's, landing L/R as the Baron; wing-ice light `laminar/c90/lighting/switch/wing_ice_up/_dn` | one 3-position switch, `no_smoke_fsb_pos` (1 NO SMOKING + SEAT BELT, 0 OFF, -1 SEAT BELT), `no_smoke_fsb_up/_dn`. Its handler leaves the seat belt sign lit going from -1 to 0, so Vimaan toggles it out after. No smoking alone isn't a position | X-Plane's | none | none | `c90_switches` |
| 737-800 | landing: four switches, `landing_lights_switch[0..3]` written (belly [1] [2], wing root [0] [3]); taxi: `taxi_light_brightness_pos` 0-2, whose `_pos_up` turns it down; strobes and nav share the position switch `position_light_pos` (-1 STEADY, 0 OFF, 1 STROBE & STEADY), so strobes off goes to STEADY; logo / wing / runway turnoff `generic_lights_switch[1]` / `[0]` / `[2]` `[3]`. Nothing reads `taxi_light_on` | seat belt `seatbelt_sign_pos` 0 OFF / 1 AUTO / 2 ON, whose `_up` turns it toward OFF; no smoking not modelled | X-Plane's | the lever `laminar/B738/flt_ctrls/speedbrake_lever` (writable): DOWN 0, ARMED 0.0889, FLIGHT DETENT 0.667, UP 1.0. Disarming through X-Plane's ratio leaves the lever at ARMED. In the air it stops at the flight detent | X-Plane's, the knob follows (RTO / OFF / 1 / 2 / 3 / MAX) | `b738_switches` |
| A330-300 | one LAND switch (`landing_01`); no taxi light, "taxi" is the nose light (`landing_lights_switch[1]` 0 / 0.5 TAXI / 1 T.O., `laminar/A333/switch/lighting/taxi_light_off/_taxi`); strobe `laminar/a333/switches/strobe_pos` 0 OFF / 1 AUTO / 2 ON (`strobe_pos_off/_auto/_on`) and nav & logo `nav_pos` 0-2 (`nav_light_pos_up/_dn`), both rewritten every frame; runway turnoff `rwy_turn_off_on/_off`; wing `generic_lights_switch[1]` | seat belt and no smoking each 0 OFF / 1 AUTO / 2 ON (`seatbelt_signs_up/_dn`, `smoking_signs_up/_dn`), the generic sign datarefs rewritten every frame | none (fly-by-wire) | X-Plane's ratio | LO / MED / MAX buttons: X-Plane's `brakes_1_auto` (level 3), `brakes_2_auto` (4), `brakes_rto_auto` (0 on the ground, 5 in the air), replaced by the A330; each toggles, so pressing the lit one turns it off. Needs nosewheel steering on. No "3" | `a333_switches` |

**Part B (anti-ice, electrical, fuel pumps)**, as built (T3.0b). Vimaan
answers with what a switch has: "Pitot heat on or auto?" on the A330, "The
boots cycle once each time. Say boots on." on the King Air. The 737's
"battery off" opens the guard, then switches off, and leaves it open (closing
it would put the battery back on). Generator and fuel pump steps for engines
an aircraft doesn't have are left out.

- **Baron:** pitot L/R, window heat and avionics are X-Plane's. Prop heat is one switch for both props, so use `sim/ice/prop_heat_on/off`: `prop_heat0` heats the left only. The boots are AUTO / OFF / MAN; the aircraft replaces `wing_boot*`, so `wing_boot_on` sets 2 and `_off` sets 0. There's no inlet heat. The two boost pumps are on/off only.
- **King Air:** pitot, battery, avionics and boost pumps are X-Plane's. Its own commands are needed for:
  - windshield L/R, three positions (NORMAL / OFF / HIGH): `windshield_antiice_{up,dn}_{L,R}`;
  - boots: `surface_deice_up`, one 7-second cycle, with no latched on;
  - prop heat: `prop_deice_up/_dn`; its 90-second timer undoes a write;
  - generators: RESET / ON / OFF, `generator{1,2}_up/_dn`. Its generator 2 RESET branch checks generator 1.

  Engine anti-ice is the ice vanes, `anti_ice_engine_air[0/1]`. Inlet heat is automatic and rewritten every frame. Its flight start turns off windshield, boots, prop and inlet heat.
- **737:**
  - **Probes and heat:** probe heat A/B uses `capt_probes_pos_on/off` and `fo_probes_pos_on/off`. Window heat ×4 is X-Plane's `window_heat`, `window2..4`. Wing anti-ice is X-Plane's `wing_heat_on/off` (the surface heat, not bleed air). Engine anti-ice is `inlet_heat0/1`.
  - **Battery:** it is under a guard (`guard_cover00`, a toggle; state `guard_cover_pos[0]`), and the guard logic forces it on every frame while closed. So: open the guard, then switch off. Battery off also shuts the APU down if nothing else carries the load.
  - **Generators:** GEN 1/2 are momentary `gen1_up` (OFF) / `gen1_dn` (ON).
  - **Avionics:** no avionics master; it is forced every frame.
  - **Fuel pumps:** paired per tank, `fuel_tank_pump_on[0 L, 1 C, 2 R]`.
- **A330:**
  - **Probes and window heat:** one button, AUTO / ON only, with no off: `probe_window_heat_toggle`.
  - **Anti-ice:** wing anti-ice through the wrapped `sim/ice/wing_heat_tog`, engine anti-ice through `inlet_heat0/1_tog`, both with a state check.
  - **Batteries:** `battery_1/2_toggle` (replaced) and `APU_batt_toggle`. Read the button state; `battery_on` isn't reliable.
  - **Generators:** GEN 1 `generator_1_on/_off/_toggle`, GEN 2 `generator_2_toggle` only.
  - **Avionics:** no avionics master.
  - **Fuel pumps:** 8 buttons, `laminar/A333/switches/{left1,left2,left_stby,right1,right2,right_stby,center_left,center_right}_pump_toggle`. **Never use X-Plane's `sim/fuel/fuel_pump_N_*` here:** its `fuel_pump_on[0/1]` is set by the engine master switches (`A333.switches.lua` 6069-6094), i.e. it is the engine's fuel feed.

**Switch steps** (`vimaan_nlu/switches.py`, run by the plugin one per frame, as
the transponder selectors are):
- **Knob:** click toward a position range.
- **Button:** press unless the state already reads the value, then wait.
- **Write:** write a dataref and check it holds.
- **Check:** first step only; the setting is already made when a dataref reads a value.

A button is pressed, and a dataref written, once per step: a toggle pressed
twice would undo itself (an A330 button reads 1.2 while pressed). A switch
that doesn't move in 30 frames answers "... not set: the switch did not move".

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
- It can come with the command: "override landing gear up" overrides a held
  gear up. A rest that names a different command ("override flaps up") or
  contradicts the held one ("override gear down") is not an override. A
  garbled rest still is, since the hold already said what's held. With
  nothing held, "override gear up" is just "gear up", and it gets held as
  usual. Vimaan always says what it's holding before it acts.
- **cancel / negative / disregard / abort** drops it. It also drops an open
  "to or two?" question.

Out-of-range values and commands the aircraft can't perform can't be
overridden: they are impossible, not risky.

## Running the probe

Bind keys in X-Plane (Settings → Keyboard, search "Vimaan"). The probe is
never started by voice, because it moves the autopilot.

| Key | Does |
| --- | --- |
| **Vimaan: probe this aircraft (one step per press)** | starts a run, then runs one step per press |
| **Vimaan: stop the probe (press twice to start over)** | stops after the current step; pressed again with no probe running, it forgets where the run stopped |
| **Vimaan: probe, skip to the next step** | moves on without running the step (optional) |
| **Vimaan: probe, go back one step** | moves back, for example to retry one step (optional) |

Skip and back wrap around the ends, so any single step can be reached
without running the ones in between.

**Joysticks and gamepads:** give every axis a null zone, so a stick at rest
doesn't count as pilot input (X-Plane: Settings → Joystick → select the device
→ Calibrate, then raise "Null zone"; 10–15% for a gamepad thumbstick). With a
0% null zone an airliner autopilot disconnects within seconds.

**On the ground** (3 steps: altimeter STD, altimeter setting, flap handle):
1. Parked, parking brake set, engines in any state. Set the altimeter away from 29.92 first, or STD shows nothing.
2. Press the probe key once. Vimaan says "Probe ready. 3 ground steps…".
3. Press once per step. Each takes about 4 s and is put back afterwards.

**In the air** (15 steps: 10 autopilot modes, autothrottle, knots/Mach, speed / V/S / altitude dials):
1. Above 1,500 ft AGL, in smooth air, with the autopilot on in **HDG and ALT hold**. Without them a step may not be able to put the mode back (there's no button for roll or pitch hold), so the probe says so once. Select them, or press again to run anyway.
2. Press once, then once per step. Each step presses one control for 2 s and puts it back. Expect brief mode changes, for example a few seconds in pitch mode.
3. Putting things back is tried in two rounds: first the mode that was on before, then an altitude arm or autothrottle the step changed. Altitude capture left armed on its own is harmless, and the probe carries on with a note.
4. If the autopilot disconnects during a step, the probe puts back the control it pressed, then pauses there ("The autopilot disconnected during …"). Re-engage it with HDG and ALT, then press again to retry that step. Nothing is recorded for the interrupted try.
5. If a step still can't put things back, the probe stops and says what didn't ("After level change, the autothrottle did not go back. Check the autopilot."). Fly normally and look at the mode lights.

**Nothing is lost when a run stops.** Each step is saved the moment it's
measured, along with where the run has got to. The next press of the probe key
carries on from there ("Probe ready. Continuing at V NAV, step 10 of 15"), even
after X-Plane restarts. A finished run starts from the first step next time.
To start over early, press stop twice.

Results are saved to `~/Vimaan_Logs/aircraft/<folder>__<file>.json`, with the
ground and air runs merged. A later "inconclusive" never replaces a definite
result. They are used from the next aircraft load. Delete the file to forget a
probe.
