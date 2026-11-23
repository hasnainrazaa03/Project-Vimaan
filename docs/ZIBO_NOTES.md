# Zibo 737-800 (parked 2026-11-22)

The Zibo mod's 737-800 (v4.05.33, free) was started in phase L.2 and parked
by the owner until the existing aircraft are production ready. The aircraft is
kept at `~/Vimaan_Logs/zibo_parked/B737-800X` (quarantine already cleared);
move it back into `X-Plane 12/Aircraft/` with X-Plane closed. Measuring
scripts and their output are in `~/Vimaan_Logs/measuring/zibo/`.

## What applies already

Vimaan's 737 overrides match by the commands they need, and the Zibo has most
of Laminar's 737 names. On load it got `b738_mcp`, `b738_efis_baro`,
`b738_transponder`, `b738_switches` and `b738_apu`; not
`b738_condition_levers`. The switch scan found no probe heat (`ice:pitot`).

## What differs (from its scripts and `b738_cockpit.obj` manipulators)

| Function | Laminar 737 (Vimaan now) | Zibo |
| --- | --- | --- |
| Fuel cutoff | `engines/condition_lever_1_idle` / `_cutoff` | `engine/mixture1_idle` / `_cutoff` |
| Generators | `switch/gen1_dn` / `_up` | `toggle_switch/gen1_dn` / `_up`; `electrical/gen1_pos` |
| Probe heat | `toggle_switch/capt_probes_pos_on` / `_off` | `toggle_switch/capt_probes_pos` (a toggle) |
| Battery | guard `guard_cover00` | `switch/battery_up` / `_dn`, cover `button_switch_cover02` |
| Landing lights | `landing_lights_switch[0..3]` | wing root `switch/land_lights_left_on` / `_off` (and right); belly `land_lights_ret_left_*` |
| Taxi light | brightness knob | `toggle_switch/taxi_light_brigh_toggle` |
| Logo, wing, runway turnoff | generic light switches | `switch/logo_light_on` / `_off`, `toggle_switch/wing_light`, `rwy_light_left` / `_right` |
| No smoking | none | `toggle_switch/no_smoking_up` / `_dn` |
| Anti-ice | X-Plane's | toggles `toggle_switch/wing_heat`, `eng1_heat`, `eng2_heat`, `window_heat_*`; positions `ice/*_pos` |
| Fuel pumps | X-Plane's tank pumps | six: `fuel/fuel_tank_pos_lft1` .. `rgt2` |
| Autobrake | X-Plane's | `knob/autobrake_up` / `_dn`, or `_off`, `_1` .. `_max`, `_rto` |

Some names come from its plugin's strings: check in the running sim whether
each is a command or a dataref before using it.

## First Tier 1, by text: 15/33

- The autopilot never engaged: CMD A pressed, servos stayed off, and every
  autopilot step after failed. To probe first: the IRS alignment, the flight
  director switches, or another part of its own start state.
- The heading and altitude bugs and the altimeter were put back: its scripts
  own them, so it needs its MCP and EFIS knob commands (as the A330 does).
- Radios, lights, flaps, gear and the safety holds passed.
- Loading it is slow: X-Plane crashed on its first load (recovered), and the
  second load took longer than the runner's 30 s request timeout (the load
  went on). The flight-load request needs a longer timeout for it.
