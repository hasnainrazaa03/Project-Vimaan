# Phase plan: offline speech + command expansion

Written 2026-10-30, after the first full in-sim flight with the voice worker
process (session `vimaan_20261029_235410`). Supersedes the "Medium term"
section of [ROADMAP.md](ROADMAP.md).

## Where we are

Phase 0 (PR #24) is merged: voice control works end to end in X-Plane 12.

| Measured in-sim | Value |
| --- | --- |
| Frame hitches from Vimaan | 0 stalls, 0 overruns (≤0.41 ms/frame) |
| Speech end → action | p50 784 ms, p95 1.17 s |
| — of which Google STT | p50 460 ms, p95 843 ms |
| — of which NLU | ~11 ms |
| Commands executed / read-back verified | 11 / 11 |
| Failures | all speech recognition ("slaps", "Secom", empty transcript) |

The model is not the bottleneck. Speech recognition is: half the latency,
every observed error, and the only part that needs internet. And the command
set is small: **17 intents, 6 slot types**.

Two tracks run in parallel. They meet at one point: the offline recogniser's
vocabulary is generated from the command schema, so when commands are added,
the recogniser learns their words automatically.

---

## Track S: offline speech recognition

**Choice: Vosk** (Kaldi, ~40 MB English model, `vosk` 0.3.44 has a universal2
macOS wheel). Why not Whisper: Whisper only starts work after you release the
key and is 0.3–1 s on CPU for a short clip; it also invents text on silence.
Vosk streams: it decodes *while you talk*, so after release only ~50 ms of
work remains. And Vosk accepts a **grammar**: a list of the only words it may
output. With ~300 cockpit words instead of 100,000, "slaps" and "Secom" become
impossible outputs.

| Step | What | Done when |
| --- | --- | --- |
| **S1** | Save every captured command's audio (`~/Vimaan_Logs/audio/<session>_<cmd>.wav`) with its transcript, outcome and what the pilot meant. Capped in size; off with `VIMAAN_SAVE_AUDIO=0`. | Every flight from now on grows a real-voice test set. |
| **S2** | `vosk` backend in the voice worker: stream audio into the recogniser during capture; grammar built from the command schema plus number words; model shipped by `install_plugin.py` (download once, SHA-256 checked). | Works with Wi-Fi off; release → action < 400 ms on the bench. |
| **S3** | `scripts/stt_benchmark.py`: replay the saved WAVs through Google, Vosk-grammar, Vosk-open and (optionally) Whisper; report word error rate on the command, **end-to-end intent+slot accuracy**, and latency. | A table that decides the default with data, not guesses. |
| **S4** | Make Vosk the default if it matches or beats Google end to end; keep Google selectable. Fallback chain if the grammar rejects everything ("[unk]"): say "say again". | Default switched; runbook updated. |

Risks: grammar too strict for free phrasing (mitigation: grammar is generated
from the same templates the model trains on, plus filler words; S3 measures
it); Vosk small-model accuracy on accented speech (S3 measures it; the larger
`vosk-model-en-us-0.22-lgraph` is the step up, 128 MB).

---

## Track E: command expansion

### E0. Schema v2: one source of truth

Today each new command is added by hand in five places (schema, generator,
plugin handlers, validators, self-test). Schema v2 is one declarative table
per intent:

```python
"set_vertical_speed": Intent(
    templates=["vertical speed {vs} feet per minute", "{direction} at {vs} fpm", ...],
    slots={"vs": Number(100, 6000, step=100, unit="fpm"),
           "direction": Enum(climb=["climb", "up"], descend=["descend", "down"])},
    sim=Write("sim/cockpit2/autopilot/vvi_dial_fpm"),
    readback="Vertical speed {vs}",
    safety=None,
)
```

From that table we generate: training data, slot validators, the plugin's
dispatch + read-back, the Vosk grammar, self-test probes,
`check_xplane_refs.py` coverage, and the command list in the docs.

**Families instead of numbered intents.** `toggle_engine_1` / `toggle_engine_2`
become `toggle_engine {engine: 1|2|both, state}`; same for autopilot, flight
director, radios. Fewer, better-separated intents; "both engines" and
"engine 3" come for free; adding an aircraft with four engines needs no
retraining.

**Slot value types.** `Number(min, max, step, unit)`, `Enum(...)` with
synonyms, `Bool` (on/off with synonyms: engage/arm/set/activate ↔
off/disengage/release), `Instance` (1, 2, left/right, both, all),
`Digits(n, base)` for squawk codes, `Relative(direction, delta)` for
"turn left 20 degrees" / "climb another thousand".

### Commands to add

Every dataref/command below was checked against X-Plane 12.4.3's
`DataRefs.txt` / `Commands.txt`.

**Tier 1: autopilot and radios (values)**

| Intent | Slots | Example | X-Plane binding |
| --- | --- | --- | --- |
| `set_autopilot_speed` | speed (kt / mach) | "speed one six zero knots" | `autopilot/airspeed_dial_kts_mach` |
| `set_vertical_speed` | vs, direction | "descend at one thousand five hundred" | `autopilot/vvi_dial_fpm` |
| `set_autopilot_mode` | mode (hdg/nav/apr/alt/vs/flc), state | "heading mode", "arm approach" | `sim/autopilot/heading`, `NAV`, `approach`, `altitude_hold`, `vertical_speed`, `level_change` |
| `set_nav_frequency` | radio (1/2), frequency 108.00–117.95 | "nav one one one zero decimal five" | `radios/actuators/nav1_frequency_hz` (10 kHz) |
| `set_standby_frequency` | radio (com/nav), port, frequency | "com one standby one two one five" | `com1_standby_frequency_hz_833` |
| `swap_frequency` | radio, port | "flip com one" | `sim/radios/com1_standy_flip` (sic) |
| `set_course` | nav port, degrees | "course zero niner zero" | `nav1_obs_deg_mag_pilot` |
| `set_transponder_code` | squawk (4 octal digits) | "squawk four two one seven" | `radios/actuators/transponder_code` |
| `set_transponder_mode` | mode (off/standby/on/alt), ident | "transponder alt", "ident" | `sim/transponder/transponder_alt`, `_standby`, `_ident` |
| `set_altimeter` | inHg 28.00–31.00 / hPa 950–1050 / "standard" | "altimeter two niner niner two" | `gauges/actuators/barometer_setting_in_hg_pilot`, `sim/instruments/barometer_2992` |
| `toggle_autothrottle` | state | "autothrottle on" | `sim/autopilot/autothrottle_on/_off` |
| `toggle_yaw_damper` | state | "yaw damper on" | `sim/systems/yaw_damper_on/_off` |

**Tier 2: switches (booleans)**

| Intent | Slots | Example | X-Plane binding |
| --- | --- | --- | --- |
| `toggle_lights` | light (landing/taxi/beacon/strobe/nav/all), state | "landing lights on" | `switches/landing_lights_on`, `taxi_light_on`, `beacon_on`, `strobe_lights_on`, `navigation_lights_on` |
| `toggle_anti_ice` | system (pitot/window/wing), state | "pitot heat on" | `ice/ice_pitot_heat_on_pilot`, … |
| `toggle_electrical` | system (battery/avionics/generator), instance, state | "avionics master on" | `switches/avionics_power_on`, `electrical/generator_on[]` |
| `toggle_fuel_pump` | engine, state | "fuel pump one on" | `engine/actuators/fuel_pump_on[]` |
| `toggle_signs` | sign (seatbelt/no smoking), state | "seatbelt sign on" | `switches/fasten_seat_belts`, `no_smoking` |

**Tier 3: flight controls (positions)**

| Intent | Slots | Example | X-Plane binding |
| --- | --- | --- | --- |
| `set_flaps` (replaces `toggle_flaps`) | up/down **or** notch N **or** degrees | "flaps one", "flaps fifteen", "flaps full" | `controls/flap_handle_request_ratio` + `acf_flap_detents` |
| `set_speed_brake` | arm/extend/retract | "speed brakes armed" | `controls/speedbrake_ratio` (−0.5 = armed) |
| `set_autobrake` | level (off/1/2/3/max/RTO) | "autobrake three" | `switches/auto_brake_level` |

**Tier 4: questions (read-only, zero risk)**

`ask_altitude`, `ask_heading`, `ask_airspeed`, `ask_vertical_speed`,
`ask_fuel`, `ask_frequency {radio, port}`, `ask_autopilot_status`. Answered by
reading the gauge datarefs (`gauges/indicators/altitude_ft_pilot`,
`heading_AHARS_deg_mag_pilot`, `airspeed_kts_pilot`, `vvi_fpm_pilot`,
`fuel/fuel_quantity`). The existing `ask_time` gets a real answer from
`sim/time/zulu_time_sec`.

**Tier 5: later, needs its own design**

Relative commands ("turn left twenty", "climb another thousand"), two
commands in one breath ("heading two seven zero, climb five thousand"),
checklists ("before takeoff checklist" read and verify), per-aircraft profiles
for add-on aircraft whose custom datarefs differ.

### Safety rules for the new commands

Same gate as today (confirm by repeating within 12 s):
transponder 7500/7600/7700; battery, avionics or generator off in flight;
speed brakes extended below 1,000 ft AGL; flaps beyond the placard speed
(`acf_flap_detents` + IAS); autothrottle off on short final is *not* gated
(pilots need it instantly).

### Steps

| Step | What | Done when |
| --- | --- | --- |
| **E0** | Schema v2 + types above; port the 17 existing intents into families (`toggle_engine {engine}` etc.) without changing behaviour. Generator, validators, self-test and dispatch read from it. | All 410 existing tests pass on the ported schema. |
| **E1** | Generator v2: many templates per intent, numbers rendered as pilots say them ("one two zero", "one twenty", "niner", "flight level two four zero"), near-miss negatives for `None`, ASR-style noise (dropped words, homophones seen in real logs). Drop the Pegasus/Flan-T5 paraphrase stage; its sentences ("One can request ap.") are not how pilots talk. Real-transcript eval set built from S1 logs. | Dataset + manifest; per-intent counts balanced; held-out templates (not just held-out rows). |
| **E2** | Retrain **v12** (same DistilBERT joint model, MPS). | Held-out intent acc ≥ 98.5%, slot F1 ≥ 0.99, ≥ v11 on the old intents, 100% on the real-transcript set, NLU < 25 ms, ONNX export still works. |
| **E3** | Plugin handlers Tier 1 → 4, each with read-back verification, validators, safety rules and self-test probes. Intents the model knows but the plugin hasn't implemented yet say "not available yet". | `check_xplane_refs.py` covers every binding; fake-sim tests per intent. |
| **E4** | Aircraft profiles. Detect the aircraft (`acf_ICAO`, engine count, flap detents, has autothrottle) and pick a profile: **default** (X-Plane's generic `sim/...` bindings — every Laminar aircraft), **jet** (engine start = starter + fuel/cutoff lever and wait for N2, not "starter until running"), **Airbus** (flaps by position 1/2/3/FULL, FCU modes), and **add-on** profiles for aircraft that replace X-Plane's systems with their own datarefs (Zibo 737, ToLiss A3xx, FlightFactor) — written per aircraft from its published dataref list. Unsupported → "No autothrottle on this aircraft" instead of a silent no-op. | Tested in-sim on the Laminar Baron 58, Cessna 172, Boeing 737-800 and Airbus A330-300 (all installed). |
| **E5** | In-sim checklist per tier; saved audio from those flights feeds S3 and the real-transcript set. | Every new intent executed and verified once in X-Plane. |

**One retrain, not five.** Train v12 once on the full Tier 1–4 schema; the
plugin then grows tier by tier without retraining. Tier 5 is the next model.

---

## Order of work

```
now   S1 save audio ──► S2 Vosk backend ──► (fly) ──► S3 benchmark ──► S4 default
        └────────────► E0 schema v2 ──► E1 generator ──► E2 train v12 ──► E3 handlers ──► E4 ──► E5 fly
                                              ▲
                     S2's grammar is generated from the schema, so it follows E0/E1 automatically
```

S1 first because it is small and every flight after it produces test data
for both tracks.
