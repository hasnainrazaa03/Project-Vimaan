# Vimaan build plan

The single plan for finishing Vimaan. Every phase is broken into numbered
steps, each with a concrete "done when". Work happens strictly in the order of
the status board; a phase starts only when the previous one has passed its
gate. After each step: tick it here, commit, and keep the board current.

Last updated 2026-10-30.

---

## Status board

| # | Phase | State | PR |
| --- | --- | --- | --- |
| 0 | In-sim readiness: worker process, verified bindings, telemetry | ✅ merged | #24 |
| S | Offline speech recognition (Vosk), saved audio, benchmark | ✅ built, flown twice; awaiting merge | #25 |
| O | In-sim overlay (live transcript, meaning, outcome) | 🟡 built; needs in-sim check | #26 |
| E0 | Command table: one source of truth (no behaviour change) | ⬜ next | |
| E1 | Training data v2 + command families → model **v12** | ⬜ | |
| T1 | Tier 1: autopilot core (speed, V/S, modes, A/THR, altimeter, flap positions) + aircraft profiles → **v13** | ⬜ | |
| T2 | Tier 2: radios & navigation → **v14** | ⬜ | |
| T3 | Tier 3: switches (lights, anti-ice, electrical, fuel pumps, yaw damper, signs, speed brakes, autobrake) → **v15** | ⬜ | |
| T4 | Tier 4: questions (read-only) → **v16** | ⬜ | |
| L | Later: relative commands, two commands in one breath, checklists, add-on aircraft profiles | ⬜ | |

Measured so far (in-sim, Vosk): speech end → action **p50 330 ms**;
recognition 2–3 ms; model ~11 ms; 0 frame stalls. Real-voice benchmark
(33 labelled clips): Vosk 30/33, Google 32/33.

---

## Rules for every phase

**Definition of done for a PR** (no exceptions):

1. Unit tests for every new behaviour; whole suite green (`pytest`).
2. `ruff check`, `ruff format --check`, `mypy` clean.
3. `python scripts/check_xplane_refs.py --xplane "$X"`: every dataref and
   command the plugin uses exists in X-Plane 12.4.3.
4. `python scripts/build_stt_vocabulary.py --check`: grammar in sync.
5. CI green; squash-merge; feature branch per phase/tier.
6. Installed into X-Plane (`scripts/install_plugin.py`) and flown with the
   phase's checklist; the session read with `scripts/vimaan_log_report.py`
   and `scripts/stt_benchmark.py`.
7. This board updated; project memory updated.

**Never:** change behaviour and retrain in the same step without a gate in
between; ship a model that is worse than the previous one on any earlier gate;
act on a command without a read-back where one is possible.

**Retraining costs ~20–40 min** on this Mac (v11: 70k rows, 1,220 s, early
stop at epoch 6), so each tier gets its own model and its own flight.

---

## Phase S: offline speech recognition ✅

Built and flown (A330, twice). Remaining:

- [ ] S4.1 Merge #25.
- [ ] S4.2 After each future flight: `stt_benchmark.py --seed-labels`, correct
  `~/Vimaan_Logs/audio/labels.jsonl` by hand, re-run the comparison. Keep Vosk
  the default while it is within 2 commands of Google per 30 clips and all its
  misses are safe (rejected / asked again, never a wrong action).
- [ ] S4.3 If accented speech underperforms: try `vosk-model-en-us-0.22-lgraph`
  (128 MB) through the same benchmark.

---

## Phase O: in-sim overlay 🟡

Built (#26): bottom bar with REC (live words) → HEARD → MEANT → outcome
(DONE / NO CHANGE / CONFIRM? / REPEAT / REJECTED + reason / NOT HEARD /
NO RESPONSE), fades 6 s after the result; `vimaan/toggle_overlay` cycles
auto / pinned / off.

- [ ] O.1 Merge #26 (after #25; retarget to `main`).
- [ ] O.2 In-sim check: the log report's `Overlay:` line says `imgui`; worst
  draw time < 1 ms; no `sim_stall` / `main_thread_overrun` while it's visible.
  If ImGui fails on Metal it falls back to `native` — acceptable, note it.
- [ ] O.3 Usability pass in-sim: readable at 1080p and on a pop-out window;
  adjust `IMGUI_FONT_SCALE`, width, colours if needed.
- [ ] O.4 E0 replaces the overlay's hard-coded intent/slot display names with
  the command table's (step E0.9).

---

## Phase E0: command table (no behaviour change)

**Goal.** Today a command lives in ~8 places (schema, generator, plugin
handler table, required slots, validators, command words, safety, read-back,
self-test, overlay names, vocabulary, dataref checker). After E0 there is one
table, `ML/vimaan_nlu/commands.py`, and everything else is derived from it.
It lives in `vimaan_nlu` because that is what the installer ships to X-Plane.
**Intents, data and model are unchanged** (still v11), so this phase is a pure
refactor, proven by the existing tests.

### Steps

- [ ] **E0.1 Types.** In `commands.py`, frozen dataclasses:
  - Slot types: `Number(lo, hi, step=1, unit=None, wrap=False, min_digits=1)`,
    `Choice(values, synonyms)` (on/off, up/down, modes, lights…),
    `Instance(count, words={"left": 1, "right": 2, "first": 1, …}, all_words=("both", "all"))`,
    `Digits(n, alphabet="01234567")` (squawk: kept as a string, leading zeros
    preserved), `Frequency(lo, hi, spacing)` (COM 8.33 kHz, NAV 50 kHz).
    Each type knows: validation (replaces `validators.SLOT_VALIDATORS`), how
    to render values as speech for training (`speak(value) -> [variants]`),
    and the words it contributes to the recogniser's vocabulary.
  - Bindings (what the sim does): `WriteValue(dataref, kind, scale, array_index)`,
    `CommandPair(on, off, state_dataref)` (state-checked: no-op if already in
    the asked state), `CheckedToggle(command, state_dataref)`,
    `Custom(handler_name)` (engine start, flaps, gear, autopilot).
  - `Intent(name, display, templates, slots, required, command_words,
    binding, readback, verify, safety_rule, self_test, profiles={})`.
- [ ] **E0.2 Port the 17 intents** exactly as they are (names, templates,
  synonyms, value ranges, read-back wording). Test: a snapshot of today's
  `SCHEMA` equals the one derived from the table.
- [ ] **E0.3 `config/schema_config.py` becomes a derived view**
  (`SCHEMA = training_schema(COMMANDS)`) so the current generator and
  training scripts run unchanged.
- [ ] **E0.4 Validators** — `SLOT_VALIDATORS`, `COMMAND_WORDS`,
  `REQUIRED_SLOTS` derived from the table. Existing validator/postprocessor
  tests pass unmodified.
- [ ] **E0.5 Safety** — each intent names its rule; `safety.py` keeps the rule
  logic, the intent → rule mapping moves to the table. `test_safety.py` passes.
- [ ] **E0.6 Read-backs** — wording from the table (`readback="Heading {degrees:digits}"`),
  formatting helpers in `readback.py`. Plugin tests' spoken strings unchanged.
- [ ] **E0.7 Plugin dispatch** — `_setup_intent_handlers` is built from the
  table. Generic handlers: `_write_value`, `_command_pair`, `_checked_toggle`;
  the existing custom ones (heading, altitude, flight level, COM, gear,
  flaps, parking brake, autopilot/FD, engine) stay as `Custom` until a tier
  replaces them. Read-back verification declared per intent (`verify=`).
- [ ] **E0.8 Self-test probes** from the table (one per intent that declares
  `self_test=`); still 10/10.
- [ ] **E0.9 Overlay names** — `overlay_state.INTENT_NAMES` / `SLOT_NAMES`
  replaced by `display` fields from the table.
- [ ] **E0.10 Dataref checker** — `check_xplane_refs.py` enumerates every
  binding in the table (plus profile bindings); `fake_xplane.py` gets a test
  that it knows every binding too.
- [ ] **E0.11 Vocabulary** — `build_stt_vocabulary.py` reads the table
  (templates, choice words, synonyms, number words) instead of `SCHEMA`.
- [ ] **E0.12 Generated reference** — `scripts/build_command_docs.py` writes
  `docs/COMMANDS.md` (every command, example phrases, values, sim binding,
  safety rule); a test fails if it's stale.

**Gate E0.** All existing tests pass with only import changes; v11 held-out
predictions identical to before (4,000-row comparison, as done for the number
fix); in-sim smoke flight: self-test 10/10 plus heading, altitude, COM, gear,
autopilot.

---

## Phase E1: training data v2 + command families → v12

**Goal.** Replace the data pipeline before adding commands, and prove the new
model is at least as good as v11 on the *same* commands. Then every later
tier only adds data.

### Steps

- [ ] **E1.1 Spoken-number renderer** (`ML/data/spoken.py`, unit-tested):
  every value type renders the ways pilots say it —
  digit-by-digit ("two seven zero"), pilot groups ("two seventy",
  "one eighty"), cardinals ("seven thousand five hundred",
  "twenty five hundred"), "niner", leading zeros ("zero niner zero"),
  decimals ("one two one decimal five", "one twenty one point five"),
  flight levels ("flight level two four zero", "FL two forty"). The
  normalizer must turn every rendering back into the right digits: a
  property test renders each value every way and round-trips it.
- [ ] **E1.2 Templates**: ≥15 per intent from the table — imperative
  ("gear up"), ATC/pilot phrasing ("set heading two seven zero",
  "fly heading…"), polite, terse ("flaps fifteen"), with the value first or
  last. The Pegasus/Flan-T5 paraphrase stage is dropped.
- [ ] **E1.3 Negatives** (the `None` class), ~10% of the data:
  out-of-scope chatter; command-word salad without a coherent command
  (seen in-sim: "you parking me the are"); fragments ("seventy", "on",
  "eighty two on"); questions about unsupported things.
- [ ] **E1.4 Recognition noise** on ~20% of rows, from real logs: to/two,
  for/four, do/two, a spurious leading word ("feet engine two off"), a
  dropped trailing word, "the" for "brake".
- [ ] **E1.5 Splits**: hold out whole *templates* (≈15% per intent) for the
  held-out set, so the score measures generalisation to new phrasing, not
  memorised rows. Also keep v11's held-out file for comparison.
- [ ] **E1.6 Command families** (labels change, so it belongs with this
  retrain):
  - `toggle_engine_1/2` → `toggle_engine {engine: 1|2|both, state}`
  - `toggle_autopilot_1/2` → `toggle_autopilot {unit: 1|2, state}`
  - `toggle_flight_director_1/2` → `toggle_flight_director {unit, state}`
  - `toggle_flaps` → `set_flaps {position}` (up/down only in E1; positions in T1)

  Update postprocessor (drop `NUMBERED_INTENT_RE` / family-confidence
  summing in `inference.py`), plugin handlers (engine "both" = sequence),
  safety tokens, overlay names, gold tests.
- [ ] **E1.7 Real-voice set as a test**: `tests/test_real_voice.py`
  (`requires_torch`) runs `labels.jsonl` transcripts through NLU and the
  plugin's command-word guard; v12 must not lose a clip v11 got right.
- [ ] **E1.8 Gate script** `scripts/model_gate.py vNEW --baseline vOLD`:
  per-intent accuracy and slot F1 on the template-held-out set; v11 held-out
  regression; gold NLU; real-voice clips; negatives (share that ends as
  `None`, low confidence or no command word); NLU p95 latency. Prints PASS/FAIL
  per gate and writes `gate.json` next to the model.
- [ ] **E1.9 Train v12** (`--max-length 40`, same DistilBERT, MPS).
- [ ] **E1.10 Gate v12**; ONNX export + int8 quantization still work.
- [ ] **E1.11 Install + fly** the E0 checklist again, plus "both engines".

**Gate E1.** Template-held-out intent ≥ 98.5%, slot F1 ≥ 0.99 for every intent;
no regression on v11 held-out, gold, or real-voice clips; ≥ 95% of negatives
rejected; NLU p95 < 25 ms.

---

## The per-tier procedure (T1–T4)

Every tier runs these 12 steps in order. Tier sections below list what is
specific to each tier.

1. **Research bindings.** For each intent: find the dataref/command in
   `DataRefs.txt` / `Commands.txt`; read enum meanings from the dataref
   descriptions; check what the Laminar A330 (`plugins/xlua/scripts/A333.*`)
   replaces or adds, and the Laminar 737. Record each finding next to the
   intent in `commands.py` (comment with source).
2. **Table entries**: templates (≥15), slots, binding per profile, read-back,
   verify, safety rule, command words, display names, self-test probe.
3. **Normalizer**: spoken forms specific to the tier, with round-trip tests.
4. **Vocabulary**: regenerate; `build_stt_vocabulary.py --check-vosk` fails
   on words the Vosk model lacks; add split forms and rejoin rules
   (e.g. "auto throttle" → autothrottle).
5. **Data**: regenerate the full dataset (all tiers so far) + manifest.
6. **Train** the next version.
7. **Gate** with `model_gate.py` against the previous model. Fail → fix data
   or templates and retrain; never relax a gate.
8. **Plugin**: handlers (generic where possible), safety rules, read-backs,
   verification, aircraft-profile variants.
9. **Tests**: fake-sim test per intent (right binding, value, read-back,
   verify, no-op when already set); safety tests; overlay display;
   `check_xplane_refs.py` passes.
10. **PR**, CI, review.
11. **Install and fly** the tier checklist on the Baron 58, Boeing 737-800 and
    Airbus A330. Note anything the aircraft ignored.
12. **Read the logs** (report + benchmark + label new clips), fix, merge,
    update this board.

---

## Tier 1: autopilot core + aircraft profiles → v13

Used in every phase of every flight.

### Commands

| Intent | Slots (type, range) | Say | Default binding | Verify | Safety |
| --- | --- | --- | --- | --- | --- |
| `set_autopilot_speed` | `speed` Number 60–400 kt **or** `mach` Number 0.40–0.95 | "speed one eight zero", "one eighty knots", "mach point seven eight" | ensure `autopilot/airspeed_is_mach` matches (`sim/autopilot/knots_mach_toggle`), write `autopilot/airspeed_dial_kts_mach` | dial ±0.5 kt / ±0.005 M | — |
| `set_vertical_speed` | `vs` Number 0–6000 step 100; `direction` Choice climb/up, descend/down | "descend at one thousand five hundred", "vertical speed plus eight hundred" | write `autopilot/vvi_dial_fpm` (descend = negative) | dial ±50 | — |
| `set_autopilot_mode` | `mode` Choice heading / nav / approach / altitude hold / vertical speed / level change / vnav / lnav; `state` on/off (default on) | "heading mode", "arm approach", "level change", "v nav" | toggle commands `sim/autopilot/heading`, `NAV`, `approach`, `altitude_hold`, `vertical_speed`, `level_change`, `vnav`, `gpss`, state-checked via `autopilot/heading_status`, `nav_status`, `approach_status`, `altitude_hold_status`, `vvi_status`, `speed_status`, `vnav_status`, `gpss_status` (0 off, 1 armed, 2 active) | status ≥ 1 / = 0 | — |
| `toggle_autothrottle` | `state` | "autothrottle on", "A thr off" | `sim/autopilot/autothrottle_on` / `_off` | `autopilot/autothrottle_enabled` | — |
| `set_altimeter` | `pressure` Number: 28.00–31.50 inHg **or** 940–1060 hPa (unit from the range); or `standard` | "altimeter two niner niner two", "q n h one zero one three", "set standard" | write `gauges/actuators/barometer_setting_in_hg_pilot` + `_copilot` (hPa × 0.02953); standard → `sim/instruments/barometer_2992` | ±0.01 inHg | — |
| `set_flaps` (positions) | `position` Choice up/down(=one notch)/full, **or** `notch` Number 0–detents, **or** `degrees` matched to `aircraft/controls/acf_flap_dn` | "flaps fifteen", "flaps one", "flaps full", "flaps up" | write `controls/flap_handle_request_ratio` = notch / `acf_flap_detents` | `flap_handle_request_ratio` | beyond notch 1 above the flap speed → confirm (find the Vfe dataref in step 1) |

Also in T1:

- **Fix: autopilot engage on Airbus.** The A330 replaces
  `sim/autopilot/servos_toggle` / `servos2_toggle` but not `servos_on`, which
  it ignored in flight. Engage/disengage = `CheckedToggle(servos_toggle,
  servos_on)` for every aircraft.
- **Aircraft profiles** (`ML/vimaan_nlu/aircraft.py`): detect from
  `sim/aircraft/view/acf_ICAO`, engine count, `acf_flap_detents`, and
  whether the aircraft has autothrottle; pick a profile:
  - `default`: everything above.
  - `jet`: engine start = open the fuel lever
    (`engine/actuators/mixture_ratio` → 1), engage starter, wait for
    `engine/indicators/N2_percent` to stabilise, not "starter until running"
    (the A330 restart cranked 20 s with mixture 0).
  - `airbus` (Laminar A330 first): flap positions 1 / 2 / 3 / FULL;
    selected vs managed = `laminar/A333/autopilot/{speed,heading,altitude,vertical}_knob_pull/push`
    after writing a value; A/THR = `laminar/A333/autopilot/a_thr_toggle`
    (state-checked); engine mode `laminar/A333/switch/eng_mode_left/right`.
  - Unsupported command → "No autothrottle on this aircraft", never a
    silent no-op.
- **Vocabulary**: Vosk lacks *autothrottle, vnav, lnav, flch, qnh,
  hectopascal(s), millibars, fpm, localizer, glideslope*. Use split forms
  ("auto throttle", "v nav", "l nav", "level change", "q n h", "feet per
  minute", "loc", "glide slope") and rejoin them in the normalizer. Letters
  v, l, q, h, n, a exist.

### Tier 1 flight checklist (Baron, 737, A330)

Speed 160 / mach .78 (jets) · V/S descend 1500 · heading mode, nav, approach,
altitude hold, level change, VNAV (737) · autothrottle on/off (jets) ·
altimeter 2992, QNH 1013, standard · flaps 1 / 15 / full / up · autopilot on
and off on the A330 (the fix) · "both engines" start on the ground (jet).

---

## Tier 2: radios and navigation → v14

| Intent | Slots | Say | Binding | Verify | Safety |
| --- | --- | --- | --- | --- | --- |
| `set_standby_frequency` | `radio` com/nav; `port` 1–2; `frequency` (COM 118.000–136.975 at 8.33 kHz; NAV 108.00–117.95) | "com one standby one two one decimal five" | `radios/actuators/com{N}_standby_frequency_hz_833` (kHz) / `nav{N}_standby_frequency_hz` (10 kHz) | read-back | — |
| `swap_frequency` | `radio`, `port` | "flip com one", "swap nav two" | `sim/radios/com{N}_standy_flip`, `nav{N}_standy_flip` (X-Plane's spelling) | active = previous standby | — |
| `set_nav_frequency` | `port`, `frequency` 108.00–117.95 | "nav one one one zero decimal five" | `radios/actuators/nav{N}_frequency_hz` | read-back | — |
| `set_course` | `port` (default 1), `degrees` 0–360, 3 digits | "course zero niner zero" | `radios/actuators/nav{N}_obs_deg_mag_pilot` | ±0.5° | — |
| `set_transponder_code` | `squawk` Digits(4, 0–7) | "squawk four two one seven", "squawk one two zero zero" | `radios/actuators/transponder_code` | read-back | 7500 / 7600 / 7700 → confirm |
| `set_transponder_mode` | `mode` off/standby/on/alt; `ident` | "transponder alt", "squawk standby", "ident" | `sim/transponder/transponder_off`, `_standby`, `_on`, `_alt`, `_ident` | `radios/actuators/transponder_mode` (enum from step 1) | — |

Specific work: squawk codes keep leading zeros and reject 8/9 (normalizer
must not join them into other numbers); NAV port absorbed into the frequency
("nav one one one zero…") handled like COM; A330 has its own transponder
script (`A333.transponder`): check in step 1.

Checklist: tune COM1 standby + flip, NAV1 110.50 + course 090 (ILS), squawk
4217 then 1200, transponder alt / standby / ident, emergency code asks to
confirm.

---

## Tier 3: switches by phase of flight → v15

| Intent | Slots | Say | Binding | Verify | Safety |
| --- | --- | --- | --- | --- | --- |
| `toggle_lights` | `light` landing/taxi/beacon/strobe/nav/all; `state` | "landing lights on", "all lights off" | `sim/lights/{landing,taxi,beacon,strobe,nav}_lights_on/_off` | `switches/landing_lights_on`, `taxi_light_on`, `beacon_on`, `strobe_lights_on`, `navigation_lights_on` | — |
| `toggle_anti_ice` | `system` pitot/window/wing/engine; `instance`; `state` | "pitot heat on", "wing anti ice on" | `ice/ice_pitot_heat_on_pilot` (+copilot), `sim/ice/window_heat_on/off`, `sim/ice/wing_heat_on/off`, `ice/ice_inlet_heat_on_per_engine[i]` | the same datarefs | — |
| `toggle_electrical` | `system` battery/avionics/generator; `instance`; `state` | "avionics master on", "generator two off" | `sim/electrical/battery_1_on/off`, `sim/systems/avionics_on/off`, `sim/electrical/generator_{N}_on/off` | `electrical/battery_on[i]`, `switches/avionics_power_on`, `electrical/generator_on[i]` | off while airborne → confirm |
| `toggle_fuel_pump` | `engine`; `state` | "fuel pump one on" | `sim/fuel/fuel_pump_{N}_on/off` | `engine/actuators/fuel_pump_on[i]` | off while airborne → confirm |
| `toggle_yaw_damper` | `state` | "yaw damper on" | `sim/systems/yaw_damper_on/_off` | `switches/yaw_damper_on` | — |
| `toggle_signs` | `sign` seatbelt/no smoking; `state` | "seatbelt sign on" | `CheckedToggle(sim/systems/seatbelt_sign_toggle, switches/fasten_seat_belts)`, same for `no_smoking` | the state dataref | — |
| `set_speed_brake` | `position` armed/extend/half/retract | "speed brakes armed", "speed brake extend" | write `controls/speedbrake_ratio` (armed −0.5, retract 0, half 0.5, extend 1) | read-back | extend below 1,000 ft AGL airborne → confirm |
| `set_autobrake` | `level` off/1/2/3/max/RTO | "autobrake three" | write `switches/auto_brake_level` (enum mapping from step 1) | read-back | — |

Vocabulary: "speed brake(s)", "auto brake", "r t o" (Vosk lacks the joined
forms). Checklist: preflight battery/avionics/beacon/nav, taxi lights,
landing + strobes on takeoff, pitot heat, fuel pump, yaw damper, seatbelt
sign, speed brakes armed for landing, autobrake 3, all lights off after
shutdown.

---

## Tier 4: questions (read-only) → v16

| Intent | Answer from | Spoken |
| --- | --- | --- |
| `ask_altitude` | `gauges/indicators/altitude_ft_pilot` | "Altitude seven thousand four hundred" (nearest 10 ft) |
| `ask_heading` | `gauges/indicators/heading_AHARS_deg_mag_pilot` | "Heading two seven zero" |
| `ask_airspeed` | `gauges/indicators/airspeed_kts_pilot` | "Airspeed one six five knots" |
| `ask_vertical_speed` | `gauges/indicators/vvi_fpm_pilot` | "Descending one thousand two hundred" (nearest 50) |
| `ask_fuel` | `flightmodel/weight/m_fuel_total` (kg) | "Fuel two thousand three hundred kilograms" |
| `ask_frequency` | `radio`, `port`, active/standby → the radio datarefs | "COM one one two one decimal five" |
| `ask_autopilot_status` | servos + `*_status` datarefs | "Autopilot one on, heading and altitude hold" |
| `ask_time` (existing) | `time/zulu_time_sec` | "Time one four three five zulu" |
| `ask_status_generic` (existing) | altitude, speed, heading | short summary |

No safety rules, no writes; the overlay shows the answer. Checklist: ask each
in cruise and compare against the instruments.

---

## Later (L)

Designed only after T4: relative commands ("turn left twenty", "climb
another thousand"), two commands in one breath ("heading two seven zero,
climb five thousand"), checklists read and verified, profiles for add-on
aircraft (Zibo 737, ToLiss A3xx: per-aircraft dataref lists), wake word.

---

## Risks and how they're handled

| Risk | Mitigation |
| --- | --- |
| A tier's new phrases confuse old intents | Gate compares every intent with the previous model; real-voice clips must not regress |
| Vosk grammar grows and accuracy drops | Benchmark on saved real audio after each tier; larger Vosk model as fallback |
| Aircraft ignore generic bindings (A330 did) | Step 1 reads each aircraft's scripts; read-back failures are spoken and logged |
| Confident wrong values (seen: "one eighty" → 81) | Round-trip tests for every spoken number form; command-word guard; per-type plausibility (3-digit headings) |
| Squawk/frequency digits merged with other numbers | Type-aware normalisation (`Digits` keeps zeros; port absorption rules for COM and NAV) |
