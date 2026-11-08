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
| S | Offline speech recognition (Vosk), saved audio, benchmark | ✅ merged | #25 |
| O | In-sim overlay (live transcript, meaning, outcome) | ✅ merged and flown (ImGui, worst draw 1.1 ms) | #26, #27 |
| E0 | Command table: one source of truth (no behaviour change) | ✅ merged; smoke-flown on v11 | #28 |
| E1 | Training data v2 + command families → **v17** | 🟡 all gates and ONNX parity pass; flown twice; installed for general use; awaiting merge | #29 |
| T1 | Tier 1: autopilot core (speed, V/S, modes, A/THR, altimeter, flap positions) + aircraft profiles → **v18** | ⬜ next | |
| T2 | Tier 2: radios & navigation → **v19** | ⬜ | |
| T3 | Tier 3: switches (lights, anti-ice, electrical, fuel pumps, yaw damper, signs, speed brakes, autobrake) → **v20** | ⬜ | |
| T4 | Tier 4: questions (read-only) → **v21** | ⬜ | |
| L | Later: relative commands, two commands in one breath, checklists, add-on aircraft profiles | ⬜ | |

Measured so far (in-sim, Vosk): speech end → action **p50 330 ms**;
recognition 2–3 ms; model ~11 ms; 0 frame stalls. Real-voice benchmark
(33 labelled clips): Vosk 30/33, Google 32/33. Real-voice gate set: 118
hand-labelled clips from five flights; v17 and v11 both 91 right / 26
rejected / 1 wrong.

Model versions: E1 was planned as v12; v12–v16 were intermediate runs and
v17 is the result, so the tiers continue from v18.

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

- [x] S4.1 Merge #25.
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

- [x] O.1 Merge #26. (It merged into `feat/offline-stt` after #25 had landed,
  so it reaches `main` through #27.)
- [x] O.2 In-sim check (2026-10-30): ImGui backend, worst draw 0.55 ms, no stalls.
  Original check: the log report's `Overlay:` line says `imgui`; worst
  draw time < 1 ms; no `sim_stall` / `main_thread_overrun` while it's visible.
  If ImGui fails on Metal it falls back to `native` — acceptable, note it.
- [x] O.3 Usability pass: moved to the bottom left; badge pills, outcome
  stripe, pulsing record dot (#27). Original item: readable at 1080p and on a pop-out window;
  adjust `IMGUI_FONT_SCALE`, width, colours if needed.
- [x] O.4 E0 replaces the overlay's hard-coded intent/slot display names with
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

- [x] **E0.1 Types.** *Done as built:* `Command` + `Slot` dataclasses, with
  `Range`/`Enum` moved to `slot_types.py`. The richer value types (`Number`
  with speech rendering, `Choice`, `Instance`, `Digits`, `Frequency`) arrive
  in E1, where the data generator first needs them, and the binding types
  (`WriteValue`, `CommandPair`, `CheckedToggle`) in T1 with the first generic
  handlers. E0 moved the existing binding data (autopilot/FD/engine/COM) into
  `Command.params` + `Command.bindings`. Original plan: In `commands.py`, frozen dataclasses:
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
- [x] **E0.2 Port the 17 intents** exactly as they are (names, templates,
  synonyms, value ranges, read-back wording). Test: a snapshot of today's
  `SCHEMA` equals the one derived from the table.
- [x] **E0.3 `config/schema_config.py` becomes a derived view**
  (`SCHEMA = training_schema(COMMANDS)`) so the current generator and
  training scripts run unchanged.
- [x] **E0.4 Validators** — `SLOT_VALIDATORS`, `COMMAND_WORDS`,
  `REQUIRED_SLOTS` derived from the table. Existing validator/postprocessor
  tests pass unmodified.
- [x] **E0.5 Safety** — each intent names its rule; `safety.py` keeps the rule
  logic, the intent → rule mapping moves to the table. `test_safety.py` passes.
- [ ] **E0.6 Read-backs** *Deferred to T1:* the existing read-backs carry
  logic ("already up", digit spelling) and stay in their custom handlers;
  read-back templates come with the generic handlers. Original plan: — wording from the table (`readback="Heading {degrees:digits}"`),
  formatting helpers in `readback.py`. Plugin tests' spoken strings unchanged.
- [x] **E0.7 Plugin dispatch** *Done as built:* handler map, required slots,
  non-actionable replies, autopilot/FD/engine/COM bindings all derived from
  the table; generic handlers wait for T1 (see E0.1). Original plan: — `_setup_intent_handlers` is built from the
  table. Generic handlers: `_write_value`, `_command_pair`, `_checked_toggle`;
  the existing custom ones (heading, altitude, flight level, COM, gear,
  flaps, parking brake, autopilot/FD, engine) stay as `Custom` until a tier
  replaces them. Read-back verification declared per intent (`verify=`).
- [x] **E0.8 Self-test probes** from the table (one per intent that declares
  `self_test=`); still 10/10.
- [x] **E0.9 Overlay names** — `overlay_state.INTENT_NAMES` / `SLOT_NAMES`
  replaced by `display` fields from the table.
- [x] **E0.10 Dataref checker** — `check_xplane_refs.py` enumerates every
  binding in the table (plus profile bindings); `fake_xplane.py` gets a test
  that it knows every binding too.
- [x] **E0.11 Vocabulary** — `build_stt_vocabulary.py` reads the table
  (templates, choice words, synonyms, number words) instead of `SCHEMA`.
- [x] **E0.12 Generated reference** — `scripts/build_command_docs.py` writes
  `docs/COMMANDS.md` (every command, example phrases, values, sim binding,
  safety rule); a test fails if it's stale.

**Gate E0 result (2026-10-30):** ✅ derived schema identical to v11's
(fixture test); 4,000 held-out predictions byte-identical; 93/93 X-Plane
names; all tests, ruff, mypy green. Test changes: the self-test COM probe is
picked by intent instead of position. Remaining: in-sim smoke flight.

**Gate E0.** All existing tests pass with only import changes; v11 held-out
predictions identical to before (4,000-row comparison, as done for the number
fix); in-sim smoke flight: self-test 10/10 plus heading, altitude, COM, gear,
autopilot.

---

## Phase E1: training data v2 + command families → v17

**Goal.** Replace the data pipeline before adding commands, and prove the new
model is at least as good as v11 on the *same* commands. Then every later
tier only adds data.

### Steps

- [x] **E1.1 Spoken-number renderer** (`ML/data/spoken.py`, unit-tested):
  every value type renders the ways pilots say it —
  digit-by-digit ("two seven zero"), pilot groups ("two seventy",
  "one eighty"), cardinals ("seven thousand five hundred",
  "twenty five hundred"), "niner", leading zeros ("zero niner zero"),
  decimals ("one two one decimal five", "one twenty one point five"),
  flight levels ("flight level two four zero", "FL two forty"). The
  normalizer must turn every rendering back into the right digits: a
  property test renders each value every way and round-trips it.
- [x] **E1.2 Templates**: ≥15 per intent from the table — imperative
  ("gear up"), ATC/pilot phrasing ("set heading two seven zero",
  "fly heading…"), polite, terse ("flaps fifteen"), with the value first or
  last. The Pegasus/Flan-T5 paraphrase stage is dropped.
- [x] **E1.3 Negatives** (the `None` class), ~10% of the data:
  out-of-scope chatter; command-word salad without a coherent command
  (seen in-sim: "you parking me the are"); fragments ("seventy", "on",
  "eighty two on"); questions about unsupported things.
- [ ] **E1.4 Recognition noise** *Partly done:* stray leading word, dropped filler word, "the" for "brake". Not done: to/two and for/four (they change the number, so the row is no longer labelled correctly and the round-trip check drops it); do/two (removed from the grammar instead); dropped trailing word. Original plan: on ~20% of rows, from real logs: to/two,
  for/four, do/two, a spurious leading word ("feet engine two off"), a
  dropped trailing word, "the" for "brake".
- [x] **E1.5 Splits**: hold out whole *templates* (≈15% per intent) for the
  held-out set, so the score measures generalisation to new phrasing, not
  memorised rows. Also keep v11's held-out file for comparison.
- [x] **E1.6 Command families** (labels change, so it belongs with this
  retrain):
  - `toggle_engine_1/2` → `toggle_engine {engine: 1|2|both, state}`
  - `toggle_autopilot_1/2` → `toggle_autopilot {unit: 1|2, state}`
  - `toggle_flight_director_1/2` → `toggle_flight_director {unit, state}`
  - `toggle_flaps` → `set_flaps {position}` (up/down only in E1; positions in T1)

  Update postprocessor (drop `NUMBERED_INTENT_RE` / family-confidence
  summing in `inference.py`), plugin handlers (engine "both" = sequence),
  safety tokens, overlay names, gold tests.
- [x] **E1.7 Real-voice set as a test**: `tests/test_real_voice.py`
  (`requires_torch`) runs `labels.jsonl` transcripts through NLU and the
  plugin's command-word guard; v12 must not lose a clip v11 got right.
- [x] **E1.8 Gate script** `scripts/model_gate.py vNEW --baseline vOLD`:
  per-intent accuracy and slot F1 on the template-held-out set; v11 held-out
  regression; gold NLU; real-voice clips; negatives (share that ends as
  `None`, low confidence or no command word); NLU p95 latency. Prints PASS/FAIL
  per gate and writes `gate.json` next to the model.
- [x] **E1.9 Train** *Done as v17* (`--max-length 40`, same DistilBERT, MPS; v12–v16 were intermediate runs).
- [x] **E1.10 Gate; ONNX export + int8 quantization still work.** *Done (2026-11-07):*
  - all six gates pass (`gate.json`);
  - ONNX export works (opset 17, 253 MB, 10/10 logit argmax parity);
  - `scripts/backend_parity.py` runs all 15,305 gate texts through torch int8, torch fp32 and ONNX. ONNX makes the same decision as fp32 on every text.
  - int8 quantization is really applied, not a silent fp32 fallback. It decides 56 texts differently: int8 6 right / 50 rejected, ONNX 8 / 48, 0 wrong for either.
  - Latency p50: ONNX 3.3–4.1 ms, fp32 6.2–7.2 ms, int8 8.7–9.1 ms. int8 is slower on qnnpack, as it was on v9.
  - The longest text is 21 tokens, so the runtime's 32-token limit (v17 trained at 40) truncates nothing.
- [ ] **E1.11 Install + fly** the E0 checklist again, plus "both engines". *Mostly done:* two controlled v17 flights (2026-10-30, 2026-11-07); installed for general use on 2026-11-07, by the owner's decision. Still to fly: the five "frequency … on com two" phrases (fixed offline) and "both engines". Both go into the T1 flight.

**E1 status (2026-10-30):** v17 passes all six gates after a one-line scorer fix (the COM 1 default for an unspoken port now applies to the prediction as well as the expected action); see `gate.json` beside the model. Not installed: X-Plane stays on v11 until #27 and #28 merge, and v17 also needs this branch's plugin code. Known misses: 5 of 490 explicit-port COM rows tune COM 1 ("frequency … on com two" with a one-digit decimal; v11 got all 490 right), "parking bring on engine" starts engine 1, "fly at one thousand" is rejected.

**v17 flight and follow-up (2026-10-30):** five commands set a wrong target because "to" and "two" tie in recognition ("altitude to 1,000" → 21,000), on v11 too. Added on this branch, not flown: ask when both readings are valid and differ; delayed read-back of written values; logging of modes, actual heading and altitude, COM values, autopilot changes and aircraft type. Re-scored with no retraining, thresholds unchanged: all six gates pass; real voice 91/26/1 for both models on 118 clips. X-Plane is back on v11. Open: COM 1 observation unexplained (inconclusive), autopilot off between 702 s and 854 s unexplained, "heading to seven zero" can set 270 because digit answers skip the check, "two five hundred" takes 2,500 because 500 is below the minimum.

**Second v17 flight and fixes (2026-11-07):** session `20261107_010639`, Baron, 39 commands, 0 main-thread overruns (six sim stalls during a flight reload). The to/two ask worked 3 of 3. COM 1 held 118.750 at +1, +10 and +30 s. Heading followed the bug with HDG on (88.6° → 179.9°). ALT hold held altitude, but on the Baron pressing ALT replaces the selected altitude with the current one, so a voice altitude target never drives the aircraft. "Frequency 133.1 on com two" tuned COM 1 in flight. Cause: that phrasing is a held-out template, v17 left the port untagged, and the postprocessor only corrected tagged ports. Fixed on this branch with no retraining and thresholds unchanged:
- a port named once in the text fills an untagged one;
- a bare-number reply to the to/two question is acted on only when it is one of the two values offered, within 12 s, and any other bare number is answered aloud;
- "two five hundred" now asks (500 or 2,500) instead of setting 2,500;
- autopilot-change logging names the last command that reached the sim;
- the flight level is read back as three digits.

Re-gated: all six pass. The only change is COM slot F1, 0.9967 → 1.0000, and all five "on com two" flight phrases now give COM 2. Old data, gold, real voice, negatives and the v11 baseline are unchanged.

**Unresolved: COM 2 regression on v17**, fixed offline and not yet flown. It stays open until the five "frequency … on com two" phrases tune COM 2 in flight. Still open: "heading to seven zero" sets 270 (an acoustic tie; say headings without "to"), "parking bring on engine" starts engine 1, and the autopilot went off at about 87 kt during stall practice with no Vimaan command (cause unconfirmed).

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

## Tier 1: autopilot core + aircraft profiles → v18

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

### Decisions (owner, 2026-11-07)

1. **Two PRs:** Part A (aircraft profiles and fixes, no retraining), then Part B (new commands, v18).
2. **Probe and research flight before any training data.** It must generalize to *every* aircraft, not just the three test aircraft.
3. **Altitude on aircraft that can't capture one:** (a) by default, set the target and say how to get there; (b) optionally, Vimaan captures it (it engages altitude hold at the target). Option (b) is off unless the pilot turns it on.
4. **Safety rules for the new commands as proposed**, plus an **override** command.

### How it generalizes

No behaviour is keyed to an aircraft's name. Every action resolves through five layers, from most general to most specific:

1. **X-Plane's standard controls** (`sim/...`). Most default and add-on aircraft honour them.
2. **Capabilities read when the aircraft loads**, from X-Plane's own datarefs:
   - engine count and type per engine (`acf_en_type`: piston / turboprop / jet / electric);
   - flap detents and angles;
   - retractable gear;
   - autopilot type (`preconfigured_ap_type`) and single-axis flag;
   - V-speeds (`acf_Vso`, `acf_Vfe`, `acf_Vno`, `acf_Vne`).
3. **Overrides that apply only when the loaded aircraft has the custom control.** The A330's `laminar/A333/autopilot/a_thr_toggle` is used only if that command exists. Matching on ICAO or author is unreliable: Laminar's 737 and A330 list individual authors, and add-ons reuse ICAO codes.
4. **Probe results** saved per aircraft file (`~/Vimaan_Logs/aircraft/<aircraft>.json`). If the probe measured that a mode command does nothing on this aircraft, Vimaan says so instead of trying.
5. **Verification after every action.** On an aircraft nobody has probed, a command the aircraft ignores is reported aloud, never assumed to have worked.

An aircraft never seen before gets layers 1, 2 and 5 automatically, and layer 4 after one probe flight.

### Part A — profiles and fixes (PR 1, no retraining)

- [x] **A1 Research record** (`docs/AIRCRAFT_PROFILES.md`). Every Tier 1 control, with its source: X-Plane 12.4.3 `DataRefs.txt` / `Commands.txt`, the aircraft's xlua scripts, and the POH. *Verified 2026-11-07:*
  - all Tier 1 names exist;
  - the Baron autopilot has no VS mode, no altitude capture and no A/T;
  - the 737 uses the standard autopilot commands;
  - the A330 wraps servos, altitude hold, NAV, approach, knots/Mach and baro STD, and adds knob push/pull and A/THR commands.
- [x] **A2 `vimaan_nlu/aircraft.py`** (pure Python, unit-tested):
  - `AircraftInfo` (identity plus layer-2 capabilities);
  - an override table (layer 3), each override declaring the command it requires;
  - `resolve(info, available_commands, probe)` returns a `Profile`.
  - The plugin logs the identity, the capabilities and the resolved profile on every aircraft load.
- [x] **A3 Autopilot engage on every aircraft:** read `servos_on`; if it needs changing, press `servos_toggle` and verify. The A330 ignored `servos_on` in flight but wraps the toggle. Use the explicit `servos_on` command only when the state can't be read.
- [x] **A4 Engine start by engine type:**
  - Piston: hold the starter until the engine runs (as now).
  - Jet: starter on, open the fuel (`mixture_ratio` → 1) once N2 passes 20%, hold until it runs, 60 s limit.
  - Turboprop: the same, with N1 / Ng.
  - Electric: no starter, so say so.
  - N1/N2 is logged at every step, and a failure says which step stalled.
- [x] **A5 Override and cancel** (matched on the transcript, never sent to the model):
  - "override" / "confirm" / "affirmative" executes the command a safety rule is holding, and the pilot hears which rule was overridden.
  - "cancel" / "negative" / "disregard" clears a held command or an open "to or two?" question.
  - Validators and "not on this aircraft" refusals can't be overridden; they are impossible values, not risks.
  - The words are added to the recognizer's grammar.
- [x] **A6 Probe** (`vimaan/probe_aircraft`, bound to a key, never voice; `vimaan/probe_stop`).
  - The first press records identity, capabilities and which candidate controls exist, with nothing actuated. Each later press runs **one** step.
  - A step snapshots the autopilot, presses one control or writes one dial, reads back after 2 s, classifies it (works / no change / missing), and restores the previous state.
  - Restoring means pressing the button for the previous mode (mode datarefs are read-only without an override) and writing old dial values back.
  - If restoring fails, the probe stops and says so.
  - Airborne steps need autopilot on and 1,500 ft AGL; ground steps (flaps) need the parking brake set.
  - The results file is layer 4.
- [x] **A7 Model input length from the model's manifest** (v17 trained at 40, runtime 32), with dynamic padding. Measure the gate before and after: decisions must not change. *Done:*
  - On all 15,305 gate texts, int8 decisions change on 28, and all 28 keep their outcome: 27 rejected either way (for a different reason), 1 right either way.
  - p50 8.7 → 4.4 ms.
  - Gate re-run: every v17 number unchanged except p95, 9.7 → 5.9 ms.
  - v11 as the fallback: unchanged on old data and real voice; on new phrasings, intent 91.64% → 91.56%.
- [x] **A8 `check_xplane_refs.py`** also checks aircraft-specific names (`laminar/...`) against that aircraft's scripts. *Done:* 223/223 X-Plane names, 8/8 aircraft-specific controls.
- [ ] **A9 Tests** (fake sim per profile), docs, PR. The owner merges. *Tests and docs done (1,124 tests); PR open.*

**Research flight (after A):** run the probe on the Baron, 737 and A330, plus one aircraft with no override (King Air C90B, a turboprop, or Citation X, a jet). Also check engine start by type on the ground. The results set Part B's per-aircraft data.

### Part B — new commands → v18 (PR 2)

- [x] **B1 Table entries** for the six commands above:
  - templates, slots, bindings per layer;
  - read-back templates and verification (E0.6);
  - safety rules, command words, display names, self-test probes.
- [x] **B2 Normalizer:** "point seven eight", "two niner niner two" → 29.92, "one thousand five hundred" (rate), plus/minus, and rejoining "q n h", "v nav", "l nav", "auto throttle", "level change", "feet per minute". All with round-trip tests.
- [x] **B3 Vocabulary:** split forms for words Vosk lacks; `build_stt_vocabulary.py --check-vosk` passes.
- [x] **B4 "To or two?"** extended to speed and vertical speed.
- [x] **B5 Safety:**
  - flaps beyond the first notch above that notch's speed limit → confirm;
  - a speed target outside the aircraft's range (`acf_Vso`…`acf_Vne` / `acf_Mmo`) → refuse;
  - mode changes → no confirmation;
  - everything holdable can be overridden (A5).
- [ ] **B6 Altitude without capture:**
  - (a) default: set the target, then say "This aircraft can't capture altitude. Climb, and say altitude hold at the target."
  - (b) optional `capture` setting (`VIMAAN_CAPTURE=1`): Vimaan watches the altitude and presses altitude hold within a band scaled to vertical speed. It cancels on "cancel", on any pilot mode change, or if the target moves.
- [ ] **B7 Data:** regenerate everything, and train **v18** at the same length as A7.
- [ ] **B8 Gate v18 vs v17**, same bars, plus Tier 1 gold commands.
- [x] **B9 Handlers** through the resolved profile; fake-sim tests per command and per profile.
- [ ] **B10 PR.** The owner merges.

*Part B status (2026-11-07):*
- **B1:** 5 new commands plus flap settings. 19 intents, 15 actionable. Read-backs are in the handlers.
- **B2:** most forms already parsed. Mach needed one rule ("mach point 78" → 0.78). Split forms ("auto throttle", "v nav", "q n h") go to the model as Vosk says them, so nothing is rejoined.
- **B3:** 41 grammar words added; a test checks every grammar word against the Vosk model.
- **B4:** the "to or two?" check covers speed, V/S and flap settings. A speed reading below the aircraft's Vso isn't asked ("speed two eighty" on a jet).
- **B5:** flap setting or full above `acf_Vfe` → confirm; speed below Vso, or above Vne / Mmo → refuse.
- **B7:** dataset `aviation_v2_v7` generated: 67,320 rows, 11,485 held out, 12 dropped by the round-trip check. Training waits for the research flight.
- **B9:** handlers go through the profile: knots/Mach toggle, A330 knob pull, V/S engaged before the rate is written, mode state checked, A/T via the aircraft's own control, flap labels.
- **Still to do:** B6 capture, B7 training, B8 gold commands and gate.

### Part C — Tier 1 flight (Baron, 737, A330, plus one aircraft with no override)
On each aircraft:
- speed 160, and Mach .78 on jets;
- V/S descend 1500;
- heading mode, nav, approach, altitude hold, level change, and VNAV (737);
- autothrottle on/off (jets);
- altimeter 2992, QNH 1013, standard;
- flaps 1 / 15 / full / up;
- autopilot on and off;
- "both engines" start on the ground (jets);
- an override of one safety hold;
- the capture option, once, on the Baron;
- **from E1:** the five "frequency … on com two" phrases.

Then logs, fixes, merge.

---

## Tier 2: radios and navigation → v19

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

## Tier 3: switches by phase of flight → v20

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

## Tier 4: questions (read-only) → v21

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
