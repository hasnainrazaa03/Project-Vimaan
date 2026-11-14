# Vimaan build plan

The single plan for finishing Vimaan. Every phase is broken into numbered
steps, each with a concrete "done when". Work happens strictly in the order of
the status board; a phase starts only when the previous one has passed its
gate. After each step: tick it here, commit, and keep the board current.

Last updated 2026-11-11.

---

## Status board

| # | Phase | State | PR |
| --- | --- | --- | --- |
| 0 | In-sim readiness: worker process, verified bindings, telemetry | ✅ merged | #24 |
| S | Offline speech recognition (Vosk), saved audio, benchmark | ✅ merged | #25 |
| O | In-sim overlay (live transcript, meaning, outcome) | ✅ merged and flown (ImGui, worst draw 1.1 ms) | #26, #27 |
| E0 | Command table: one source of truth (no behaviour change) | ✅ merged; smoke-flown on v11 | #28 |
| E1 | Training data v2 + command families → **v17** | ✅ merged; flown twice | #29 |
| T1 | Tier 1: autopilot core (speed, V/S, modes, A/THR, altimeter, flap positions) + aircraft profiles → **v19** | ✅ closed 2026-11-12: Part C flown unattended (121/121 on four aircraft) and the owner's real-voice check passed | #30–#34, #38–#42 |
| D | Flight review: autopilot-disconnect diagnostics (cause, log, callout) + dashboard Flights tab | ✅ merged, installed; callout checked in the sim (switch); stick pull still to check | #35 |
| A | Automated testing in the sim: a test bridge, recorded voice, scenarios, then takeoff and landing, so the tier checklists run without a pilot | 🟡 A1 and A2 done: Tier 1 passes 121/121 unattended on four aircraft (2026-11-12); A3 and A4 later | #36–#40 |
| T2 | Tier 2: radios & navigation → **v25** | 🟡 groundwork, runner scenarios, data and model done: v25 passes all six gates (2026-11-13); installed 2026-11-13; runner in the sim 95/95 on four aircraft; next: the owner's real-voice check | #43, #46 |
| T3 | Tier 3: switches (lights, anti-ice, electrical, fuel pumps, yaw damper, signs, speed brakes, autobrake) → **v26** | 🟡 T3.0a and T3.0b built (#50, #51); the first v26 failed the gate; data v15 fixes the causes, v26-d/e/f training | #50, #51, #52, #54 |
| T4 | Tier 4: questions (read-only) → **v27** | ⬜ | |
| F | Flight context: phase of flight and envelope margins, every second; safety rules and callouts read it | ⬜ after T4 | |
| C | Conversational copilot: free speech answered and acted on through the same handlers, safety rules and read-backs | ⬜ after F | |
| L | Later: relative commands, two commands in one breath, checklists, add-on aircraft profiles | ⬜ | |

Measured so far (in-sim, Vosk): speech end → action **p50 330 ms**;
recognition 2–3 ms; model ~11 ms; 0 frame stalls. Real-voice benchmark
(33 labelled clips): Vosk 30/33, Google 32/33. Real-voice gate set: 118
hand-labelled clips from five flights, as the shipped grammar hears them:
v11, v17 and v19 88 right / 28 rejected / 2 wrong, v25 89 / 28 / 1
(re-heard 2026-11-13; with the E1 grammar it was 91 / 26 / 1).

Order from here (owner, 2026-11-11): D (#35) → A1 → A2, which runs the Tier 1
flight checklist and so closes T1 → T2.0 groundwork → A3 → A4 → T2 build,
train and gate, tested by the runner → T3 → T4 → F → C (F and C added
2026-11-12, at the owner's question about flight-envelope awareness and a
talking copilot).

Model versions: E1 was planned as v12; v12–v16 were intermediate runs and
v17 is the result. Tier 1's first model, v18, failed the gate and v19 passed,
so the later tiers moved up one (T2 → v20). Tier 2 took v20–v25 (v25 is the
average of v22–v24; see Tier 2), so T3 → v26 and T4 → v27.

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

## Tier 1: autopilot core + aircraft profiles → v19

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
- [x] **B6 Altitude capture.** The probe showed altitude arm working on all four probed aircraft, so the aircraft's own capture comes first:
  - after V/S engages (or a new altitude is set while climbing or descending), altitude arm is pressed 1.2 s later, only if not armed already: the 737 arms it by itself, and the arm command is a toggle;
  - (b) where the probe says arm does nothing: Vimaan's own capture, if switched on (`VIMAAN_CAPTURE=1`, or the **Vimaan: altitude capture by Vimaan on/off** key). It presses altitude hold within a band scaled to vertical speed (6 s ahead, at least 60 ft), and stands down on "cancel", a pilot mode change, a moved target or the autopilot off;
  - (a) otherwise the read-back says "This aircraft can't capture altitude. Say altitude hold at <target>."
- [x] **B7 Data and training.** *v18 (data v7, 67,320 rows) failed the gate; **v19** (data v8) passes. Both trained at length 40, 10 epochs, patience 4, batch 16, lr 5e-5, watched live on the dashboard.*
  - v18's first gate: new phrasings (Mach and altimeter-inch slots F1 0.49 / 0.80; COM intent 0.95), old data (96.68% vs 97.21%), gold 39/41.
  - Two decoding bugs, fixed in code: `reconstruct_slot_value` turned "0", ".", "78" into "0. 78" (since v1; COM frequencies were repaired from the text, which hid it), and the choice canonicalizer matched "nav" inside "v nav". Re-gated, v18 still failed on COM.
  - The data cause: COM's only portless training rows were "tune ..." ("frequency {f}" is a held-out template), so "frequency one eighteen point four" was read as the new speed intent, whose Mach values also have decimals. Data v8 adds "new frequency", "change frequency to" and "go to frequency" (all outside the held-out set, which stays the same): 843 portless training rows instead of 248.
- [x] **B8 Gate v19 vs v17**, same bars, plus Tier 1 gold commands. **All six pass:**
  - new phrasings (11,376): intent 1.0000, slot F1 1.0000 (v17 0.6891 / 0.6667, which lacks the new commands);
  - old data (7,170): 97.25% right, 0 wrong (v17 97.21%, 0 wrong);
  - gold: 41/41 (17 Tier 1 phrases plus the E1 COM 2 phrase), 8/8 fragments not acted on;
  - real voice (118): 91 / 26 / 1, same as v17 (the one wrong is the same recording);
  - negatives: 873/873 rejected;
  - latency p95 6.2 ms.
  - Backend parity: fp32 = ONNX on all 8,874 texts; int8 differs on 2, right either way. ONNX p95 3.2 ms.
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
- **From the probes (2026-11-07/08):**
  - an aircraft's own control measured as "no change" is tried and verified, not refused (737 LNAV/VNAV need a flight plan; A330 A/THR held at FLX);
  - FMS modes on airliners explain a failure ("needs a flight plan in the F M C"); A330 managed modes push the FCU knobs;
  - autothrottle is refused where the aircraft's own files never use one (X-Plane's generic A/T "worked" on the Baron and King Air); the A330's armed A/THR counts as on;
  - the 737's altimeter is turned with its knob, since it puts written values back.
- **Done since:** B10 merged (#34); v19 installed (2026-11-08), v17 and v11 kept for rollback.
- **Still to do:** Part C flight.

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

## Phase D: flight review 🟡

Flight 203741 (Baron) lost the autopilot 17 min 38 s in, with nothing in the
log to say why: Vimaan's last action was 31 s earlier, and the aircraft had
gone from level to +1,250 ft/min in altitude hold a second before. The 737
flights had the same. No stick position or button presses were logged.

- [x] D.1 Diagnostics (`vimaan_nlu/disconnect.py`): the plugin keeps 6 s of
  pilot stick input (`sim/joystick/yoke_*_ratio`, not the servo output) and
  watches every autopilot-off command (X-Plane's, and the 737's disengage bar
  and CMD A/B) whoever fires it. When the autopilot drops it logs
  `autopilot_disconnect` with a cause (reload, probe, Vimaan, no power,
  failure, switch or button, stick, unknown), the stick peak, the null zone,
  power and fail flags, and says "Autopilot disconnected" plus the cause,
  except when Vimaan or the probe did it or a flight was just loaded.
  The joystick setup (null zone) is logged per aircraft.
- [x] D.2 Dashboard Flights tab: every session in `~/Vimaan_Logs` (or
  `VIMAAN_LOG_DIR`) as a timeline: each command with what was heard and
  understood, Vimaan's reply, what was sent to X-Plane and checked, outcome,
  speech-to-action time and the recording; autopilot changes and whether
  Vimaan made them; startup, joystick, probe and problems. A flight in
  progress updates live.
- [ ] D.3 Install the plugin update (owner's approval) and check, on the next
  flight, that a deliberate stick pull with the autopilot on is called out as
  "Stick moved".

## Phase A: automated testing in the sim 🟡

Goal: the tier checklists run with no one at the controls. The owner starts
or approves a run and reads the report; a short check with the real
microphone and voice stays manual.

What X-Plane 12.4 offers (checked 2026-11-11 in the installed build):
- starting a flight from a file (`--new_flight_json`): `ramp_start`,
  `runway_start` (with `final_distance` in nm, i.e. on final approach),
  `lle_ground_start`, `lle_air_start` (airborne at a position and altitude);
- plugin calls `setUsersAircraft`, `placeUserAtAirport`, `placeUserAtLocation`
  (airborne, engines running, at a speed);
- overrides for the stick, throttle and brakes, so a script can fly and a
  stray joystick can't interfere;
- the local web API (port 8086, on in this install).

Rules: a run happens only when the owner starts or approves it, never while
they fly. The bridge is off unless switched on, says so when it is, and
gives the controls back at the end of a run.

- [x] **A1 Test bridge and recorded voice.** *Done (#36), and run live 2026-11-12, session 005223, details below.*
  - Plugin bridge, off by default; on while `~/Vimaan_Logs/test/bridge.on`
    exists (announced). Requests are files in `test/inbox/`, results in
    `test/outbox/`: say (a WAV through the voice engine, or text through the
    model only), read, write, command, place in the air, airport, aircraft,
    state. A say result carries the outcome, replies and actions.
  - Voice worker: a `wav` request runs a recording through recognition and
    the model exactly as a push-to-talk recording would.
  - Runner `simtest` (terminal): status, say (speech generated with macOS
    `say`, or a recorded clip), read / write / command, place.
  - Launching X-Plane straight into a flight from a file: check whether the
    Steam build allows a direct launch.
  - Done when: with X-Plane running and no one at the controls, "autopilot on"
    in a generated voice engages the autopilot and the result comes back to
    the terminal.
  - **Live run (2026-11-12, session 005223), no one at the controls:**
    - X-Plane was started through Steam. The flight was started through the
      web API: the Baron in the air over Seattle, 6,000 ft, engines running.
    - In the generated voice "Daniel", every command was heard exactly and
      done, with its read-back checked: "autopilot on", "heading mode",
      "altitude hold", "altitude five thousand", and "descend at one thousand
      five hundred feet per minute" (read back "Descend 1500 feet per minute,
      to 5000"). Speech to action 21–111 ms.
    - The descent held about −1,450 ft/min with altitude capture armed,
      switched to altitude hold at about 5,200 ft, and settled at 4,996 ft.
    - A disconnect pressed through the bridge (`servos_fdir_off`) was
      diagnosed as "switch or button", and Vimaan said so. D.3 still needs a
      stick pull; no joystick was connected.
    - Clean shutdown, no errors.
  - **Learned about X-Plane, now in `simtest launch` / `fly` / `quit`:**
    - The Steam edition starts only through Steam (`steam://run/2014780`),
      and Steam asks for a launch option (X-Plane / Plane Maker / Airfoil
      Maker) unless a default is set.
    - The first start came up in Safe Mode (plugins off) for no clear
      reason; the second was normal.
    - `POST /api/v3/flight` starts the flight but leaves the Flight
      Configuration screen open over it, paused; `toggle_flight_config`
      closes it.
    - `sim/operation/quit` asks to confirm unless "Don't ask again" was
      ticked once.
- [x] **A2 In-air runner for Tier 1.** *Done 2026-11-12 (#37, fixes #38–#40): the final pass on `main` 9ffae19 passed every step on all four aircraft, 121/121.* One scenario per aircraft (Baron, 737,
  A330, King Air): placed in the air, set up, then each checklist command is
  said and the sim is checked: targets, modes, the level-off at the selected
  altitude, COM 2, altimeter, flaps. Pass or fail per step in a report the
  dashboard shows. Done when: all four run unattended and their failures are
  the ones a pilot would see. This closes T1 Part C, plus one short real-voice
  check by the owner.
  - **First runs (2026-11-12, session 013647):** Baron 24/27, King Air 27/27,
    737 27/30, A330 17/30. The Baron's and the 737's failures were the
    scenario's own (fixed). They found three Vimaan bugs (fix PR):
    1. **Aircraft overrides leaked across a switch.** They were detected by
       command existence, and the 737's commands stay registered after it
       unloads, so the A330 got the 737's autopilot buttons and never engaged.
    2. **The flap-speed hold used the first notch's limit.** X-Plane's
       `acf_Vfe` is `_Vfe1_kts`; the full-flap limit is `_Vfem_kts` in the
       .acf (Baron 152 / 122, King Air 190 / 150, 737 250 / 162,
       A330 215 / 180 kt). The 737 took full flaps at about 240 kt.
    3. **No callout for a pilot disconnect within 5 s of "autopilot on"**:
       Vimaan's own "autopilot …" counted as the cause, on or off.
  - **Re-run on #38 (2026-11-12, 02:37):** Baron 27/27, King Air 27/27, 737
    27/28, A330 27/30. All three fixes held. The failures were the runner's:
    the jets' thrust wasn't managed (A330 at 337 kt, Vmo 330; 737 A/T arm
    switch off). Fixed in #37/#38: thrust levers to CL on the A330, A/T engaged
    first, overspeed callout.
  - **Third run (2026-11-12, 21:00, main 0b0684d):** Baron 27/27, King Air
    27/27, A330 **31/32**, 737 23/29.
    - The 737's A/T arm override used names its script has commented out, so
      it never applied: no A/T, the speed decayed to 161 kt, and the descent
      failed. Fix: arm a hard-off autothrottle with X-Plane's own
      `sim/autopilot/autothrottle_arm` (the 737's switch), on any aircraft. The
      name checker now skips commented-out lines.
    - The A330's one failure: "level change" at the selected altitude (an
      Airbus doesn't change mode then). The runner now selects a new altitude
      first.
  - **Fourth run, jets only (2026-11-12, 21:25, main eb559a2):** 737 **31/32**,
    A330 **32/33**. The autothrottle armed and engaged on the 737. Both had
    the same one failure, a Vimaan bug: after level change, altitude hold is
    armed (status 1, capture at the selected altitude), and "altitude hold"
    said "already on" while they kept climbing. Fix: for altitude hold, on
    means holding (2).
  - **Final pass (2026-11-12, 21:38, main 9ffae19, session 213808): Baron
    28/28, King Air 28/28, 737 32/32, A330 33/33.** No errors; clean shutdown.
  - By voice (Daniel): 25–29 of 27–31 understood per aircraft. The same two
    phrases missed on every aircraft and every run: "q n h one zero one
    three" (heard "zero one three") and "tune com one to one two one decimal
    five" (heard "decimal five"). They're for the owner's real-voice check.
  - **Owner's real-voice check (2026-11-12, 22:03, A330, session 220331,
    EarPods):** all six checklist phrases were understood and done, 315–342 ms
    speech to action. Both phrases the generated voice missed worked:
    "q n h one zero one three" was heard "q in h one zero one three", and the
    COM 1 phrase "doing com one two one two one decimal five". One digit was
    misheard: "133.1" as "133.5", which was read back, so the pilot hears it.
    Frequencies are Tier 2's core: measure digit accuracy there. "alt mode" and
    "altitude mode" got "Which mode?"; they're now altitude hold (#42).
    **Tier 1 closed.** "q n h one
    zero one three" and "tune com one to one two one decimal five" were
    misheard every time; check them with a real voice.
- [ ] **A3 Ground and takeoff.** Start with engines off ("start both
  engines"), taxi (parking brake at speed, override), take off (gear up below
  1,000 ft with a positive rate, flaps up), climb to the A2 start.
- [ ] **A4 Approach and landing.** Start on final (`runway_start` with a final
  distance). Autoland where the aircraft has it (737 and A330: to confirm); a
  scripted approach and flare for the Baron and King Air, tuned per aircraft.

---

## Tier 2: radios and navigation → v25

Eight intents. The first design had six, with one standby and one swap intent
taking a `radio` com/nav slot; separate COM and NAV intents keep each one's
slots small and its frequency range single, as `set_com_frequency` already
is (2026-11-12).

| Intent | Slots | Say | Binding | Verify | Safety |
| --- | --- | --- | --- | --- | --- |
| `set_com_standby` | `com_port` 1–2 (default 1); `frequency` 118.000–136.975 | "com one standby one two one decimal five" | `radios/actuators/com{N}_standby_frequency_hz_833` (kHz) | still set 1 s later | — |
| `swap_com_frequency` | `com_port` | "flip com one" | `sim/radios/com{N}_standy_flip` (X-Plane's spelling) | active = the previous standby | — |
| `set_nav_frequency` | `nav_port` 1–2 (default 1); `nav_frequency` 108.00–117.95 | "tune nav one to one one zero decimal five" | `radios/actuators/nav{N}_frequency_hz` (10 kHz: 110.50 → 11050) | still set | — |
| `set_nav_standby` | `nav_port`; `nav_frequency` | "nav two standby one one three decimal niner" | `radios/actuators/nav{N}_standby_frequency_hz` | still set | — |
| `swap_nav_frequency` | `nav_port` | "swap nav two" | `sim/radios/nav{N}_standy_flip` | active = the previous standby | — |
| `set_course` | `degrees` 0–360, three digits; `nav_port` | "course zero niner zero" | `radios/actuators/nav{N}_obs_deg_mag_pilot` | ±0.5° | — |
| `set_transponder_code` | `squawk`: `Digits(4, 0–7)`, leading zeros kept | "squawk four two one seven", "squawk twelve hundred" | `radios/actuators/transponder_code` (0120 → 120) | still set | 7500 / 7600 / 7700 → confirm |
| `set_transponder_mode` | `xpdr_mode` off / standby / on / alt / ident | "transponder alt", "squawk standby", "ident" | `sim/transponder/transponder_{off,standby,on,alt,ident}`; the 737 and A330 turn their own selectors | `transponder_mode`; 737 and A330: the selector positions; ident: `indicators/transponder_id` | — |

Behaviour:
- **"On" means replying.** A transponder already on alt, TA or TA/RA says
  "Transponder already on": it is never turned down to mode A.
- **A number without its start is asked again** (all commands): "thousand"
  or "hundred" with no number before it ("altitude thousand", from "altitude
  eight thousand") answers "Say the whole number". **V/S goes in steps of 50**
  ("twenty minute", from "we should be there in twenty minutes", was a V/S of
  20). Measured with v25 on the same transcripts (2026-11-13): wrong actions
  13 → 8 on the checklist voices and 13 → 8 on six phrasings of every command
  in three voices, non-commands acted on 10 → 6 of 160, nothing right lost
  (real voice 90 / 28 / 0 either way); all six gates pass unchanged. The six
  left: "maintain five thousand" four times (a real altitude instruction),
  "the engine off" (confirmed in flight) and "turn left ident".
- **Read-backs name the radio**, said or not: "course zero niner zero" answers
  "NAV 1 course zero niner zero", so a lost "nav two" can't pass unnoticed.
- **Ident** only when the transponder is on ("Transponder is in standby. Say
  transponder on first.").
- The 737 and A330 have **no off** ("This transponder has no off. Say standby.").
- The slot is `xpdr_mode`, not `mode`, which is the autopilot's.
- Templates may name one value with `{slot=word}`: bare "ident" is
  `{xpdr_mode=ident}`. A template with no slot would repeat the same text.
- Vosk lacks "localizer", "preselect" and "VOR", so no template uses them
  (`tests/test_vosk_vocabulary.py`); "ILS" and "nav" are in it.

Per-aircraft findings: `AIRCRAFT_PROFILES.md`, "Tier 2 findings".

Checklist: tune COM1 standby + flip, NAV1 110.50 + course 090 (ILS), squawk
4217 then 1200, transponder alt / standby / ident, emergency code asks to
confirm.

### T2.0 Groundwork (no training)
- [x] **Research** on all four aircraft from their scripts: the 737 and A330
  own their transponder mode; everything else is X-Plane's standard controls.
- [x] **Slots and normalization:** `Digits(4, 0–7)` squawk codes keep leading
  zeros and reject 8 and 9; a NAV port absorbed into the frequency ("nav
  1110.5" → "nav 1 110.5") is split like COM; the NAV port and squawk code are
  taken from the text when the model leaves them untagged.
- [x] **Handlers and safety:** the eight handlers with read-back
  verification; overrides `b738_transponder` and `a333_transponder` turn the
  selectors one click a frame (`aircraft.Knob`); 7500 / 7600 / 7700 ask to
  confirm, and a repeat of a different code doesn't confirm it.
- [x] **Command table:** at least 15 templates each; data renderers for NAV
  frequencies and squawk codes, round-trip tested on every NAV channel and all
  4,096 codes; STT vocabulary and `COMMANDS.md` regenerated.
- [x] **Self-test** probes only the intents the loaded model knows (from the
  worker's ready message), so v19 still reports 15 of 15.
- [x] **Gold set** `GOLD_TIER2` (15 phrases): skipped for older models,
  required by the gate.
- [x] **Tier 2 scenarios** for the A2 runner (`simtest/tier2.py`, `simtest run
  tier2 <aircraft>`): the checklist above, 23 steps on the Baron and King Air,
  24 on the 737 and 25 on the A330 (their thrust steps), with the transponder
  checked the way each aircraft shows it. They run end to end on the fake sim in CI (Baron, 737 selector, A330
  selectors), so the expectations are right before X-Plane. The runner
  refuses to start on a model that predates Tier 2.
- [x] Then the per-tier procedure: data, train, gate (below).

### T2.1 Data, training and gate
Every run: DistilBERT, length 40, batch 16, lr 5e-5, 4,000 rows per command,
gated against v19 with the bars unchanged. Gate 1 ("new phrasings") asks every
intent to reach 98.5% on phrasings held out of training (2–3 per command).

| Model | Data | Result |
| --- | --- | --- |
| v20 | v9 | ✗ gates 1–2: bare "identify" was held out, so the word was never trained; "speed eighty" lost to the leading-noise word "eighty" |
| v21 | v10 | ✗ gate 1: transponder 0.875, altitude 0.979. Found a data bug: `{slot=word}` placeholders leaked into the negatives |
| v22 | v11 | ✗ gate 1: transponder 0.925 ("ident" inside word-salad negatives). From here 10 epochs with linear warmup and decay |
| v23 | v12 | ✗ gate 1: transponder 0.901, altitude 0.967 |
| v24 | v13 | ✗ gate 1: autopilot mode 0.967, swap NAV 0.856. First run with a development set |
| **v25** | — | ✅ **all six gates**: the uniform average of v22, v23 and v24 (`ML/average_models.py`) |

Fixes along the way, all in #43: placeholders out of the negatives, the same
data from the same seed in every process, no whole command ("ident") inside
word salad, the linear learning-rate schedule, and the development set (one
more held-out phrasing per command, which picks the checkpoint).

Why an average: after v21 every run missed 2–3 of the 87 held-out phrasings
and never the same ones (v22–v24: no phrasing failed twice), so which run
passed was luck. More data patches would have been aimed at the test
phrasings. The three models start from the same base and sit 5% apart, so
their average is one model of the same size and speed that evens out what
each got wrong by chance. The development set scored 100% on every run and
did not tell the runs apart; it still keeps the checkpoint choice off the
trained phrasings.

v25 against v19: new phrasings 0.9999 intent / 1.0000 slot F1 (every command
1.000); old data 10,063 right / 225 rejected / 0 wrong, the same as v19; gold
56/56 and 8/8 fragments; real voice 92 / 26 / 0 (v19 91 / 26 / 1: "parking
brake on", heard as "parking bring on engine", now sets the parking brake
instead of starting an engine); negatives 100%; p95 6.6 ms.

Real-voice set re-heard (2026-11-13). It had been transcribed with the E1
grammar and never rebuilt, while Tier 1 and 2 grew the grammar, so every gate
since graded old transcripts. With the shipped grammar 11 of 118 clips come
out differently, mostly worse ("autopilot two on" as "autopilot new on", which
acts on autopilot 1; "parking brake off" as "mach go"). Re-gated: all six
still pass, real voice 89 / 28 / 1 against v19's 88 / 28 / 2. The set now
records the grammar's fingerprint (`real_voice.meta.json`); a CI test and
the gate refuse a set heard with another grammar.

- [x] Installed v25 with #43's plugin at 9ab507c (2026-11-13), checked byte
  for byte; v19, v17 and v11 kept for rollback.

- [x] `simtest run tier2` on all four aircraft (2026-11-13, v25, main at
  26e7481): **95/95 steps** (Baron 23, King Air 23, 737 24, A330 25), every
  one checked in the sim. By voice 82 of 94 understood; the same three
  phrases were misheard on every aircraft and passed when redone as text:
  - "tune nav one to one one zero decimal five" heard "zero decimal five":
    rejected, "Please repeat that".
  - "transponder alt" heard "transponder hold": rejected with the question.
  - "nav two course one eight zero" heard "course one eight zero": NAV 1's
    course was set and the read-back said only "Course one eight zero". The
    course read-back now names the radio (#46).
  Cause: the Vosk grammar is the command words plus `[unk]`, which is dropped,
  so a stretch it can't place vanishes. "Tune" was lost in all six test
  voices; "transponder altitude" was heard right in all of them.
- [ ] The owner's real-voice check of Tier 2 in flight.
- [x] Why Vosk drops words like "tune" (2026-11-13). Measured on the 118
  real clips, the checklist phrases in six voices, six phrasings of every
  intent in three voices, and 160 non-command sentences, end to end with v25:
  - `[unk]` swallows stretches it can't place ("tune nav one to one one" as
    one `[unk]`), and the British test voice's "tune" and "alt" aren't heard
    even with Vosk's full vocabulary. Vosk's grammar is a bag of words:
    phrase entries don't constrain word order.
  - Dropping `[unk]`: real voice the same; 6 commands fixed and 6 new wrong
    actions elsewhere (forced words make plausible wrong commands), so it stays.
  - Dropping the 41 chit-chat-only words: chit-chat goes (0/54), commands
    don't gain.
  - Vosk's 10 best: falling back to an alternative when the best is rejected
    gives real voice +6 right but +8 wrong (it slips past the "to"/"two"
    check). Not used.
  - The larger `vosk-model-en-us-0.22-lgraph`: real voice 79 / 32 / 7, worse
    than the small model's 90 / 28 / 0 (same harness).
  - The one gap: "oh" for zero, in 753 clean training phrasings, wasn't in
    the grammar ("one oh nine" could only be heard as "one nine"). Added
    (#48), with a test that every word of a clean command phrasing is in the
    grammar. Real voice unchanged (v25 89 / 28 / 1).

---

## Tier 3: switches by phase of flight → v26

| Intent | Slots | Say | Binding | Verify | Safety |
| --- | --- | --- | --- | --- | --- |
| `toggle_lights` | `light` landing/taxi/beacon/strobe/nav/logo/wing/runway/all; `switch` on/off/auto | "landing lights on", "strobes auto", "all lights off" | per aircraft (`vimaan_nlu/switches.py`); X-Plane's `sim/lights/*_on/_off` elsewhere | the switch's own position or state | — |
| `toggle_anti_ice` | `ice` pitot (probe) / window (windshield) / wing (boots) / engine / prop; `switch` on/off/auto | "pitot heat on", "wing anti ice on", "boots on" | per aircraft; X-Plane's `sim/ice/*_on/_off` elsewhere (prop heat on propeller aircraft only) | the switch's position or state | — |
| `toggle_electrical` | `power` battery/avionics/generator (alternator); `instance` 1/2 for a generator; `switch` on/off | "avionics master on", "generator two off" | per aircraft; X-Plane's battery / avionics / generator commands elsewhere | the switch's state | off while airborne → confirm (`power_off_in_flight`) |
| `toggle_fuel_pump` | `instance` 1/2 (none: all, center included); `switch` on/off | "fuel pumps on", "fuel pump one off" | per aircraft; X-Plane's `sim/fuel/fuel_pump_{N}_on/off` elsewhere (never on the A330) | the pump's state | off while airborne → confirm (`fuel_pump_off_in_flight`) |
| `toggle_yaw_damper` | `state` | "yaw damper on" | `sim/systems/yaw_damper_on/_off` | `switches/yaw_damper_on` | — |
| `toggle_signs` | `sign` seat belt/no smoking; `switch` on/off/auto | "seat belt sign on", "seat belt sign auto" | per aircraft; `seatbelt_sign_toggle` / `no_smoking_toggle` pressed only when the state differs elsewhere | the switch's position or state | — |
| `set_speed_brake` | `speedbrake` retract/armed/half/extend (up = extend, down = retract) | "speed brake armed", "stow the speed brakes" | write `controls/speedbrake_ratio` (armed −0.5, retract 0, half 0.5, extend 1); the 737 its lever | read-back | extend or half below 1,000 ft AGL airborne → confirm |
| `set_autobrake` | `autobrake` off/r t o/max/low/medium, or `autobrake_level` 1-3 | "auto brake two", "auto brake max" | write `switches/auto_brake_level` (0 RTO, 1 off, 2-5); the A330 its LO / MED / MAX buttons | read-back | — |

Vocabulary: "speed brake(s)", "auto brake", "r t o" (Vosk lacks the joined
forms and "detent"). Per-aircraft findings: `AIRCRAFT_PROFILES.md`, "Tier 3
findings". Where an aircraft lacks a switch Vimaan says so ("This aircraft has
no auto brake", "There is no no smoking sign switch on this aircraft"). The
owner chose (2026-11-14): the core five lights plus logo, wing and runway
turnoff; AUTO as a spoken setting; the groundwork in two PRs, then one
training run (v26).

### T3.0a Lights, signs, yaw damper, speed brake, autobrake (no training)
- [x] **Research** on all four aircraft from their cockpit objects and scripts;
  every name checked against the X-Plane 12.4.3 install (416/416 X-Plane
  names, 70/70 aircraft controls).
- [x] **Switch steps** (`vimaan_nlu/switches.py`): Knob, Button, Write, Check,
  run one per frame by the plugin (the transponder selectors now use them).
  Plans for any aircraft, overridden by `b58_switches`, `c90_switches`,
  `b738_switches` and `a333_switches`.
- [x] **Five commands** with handlers, read-backs and the speed brake safety
  rule; `GOLD_TIER3` (17 phrases, skipped for older models), and "auto" and
  "armed" alone never acted on.
- [x] **Grammar:** 245 → 276 words. Re-heard, the real-voice set showed
  "light" taking "flight" ("light level two forty", "light director on", six
  clips) and "collision" and "lever" turning up unsaid, so the singular
  "light", "anti collision", "speed brake lever" and "yaw dampers" phrasings
  were dropped. After that 5 of 118 clips change, all already rejected; real
  voice is unchanged (v25 89 / 28 / 1, v19 and v11 88 / 28 / 2). No bare
  "{light} {switch}": "nav on" is the autopilot's NAV mode.
- [ ] Not installed: the larger grammar helps nothing until v26.

### T3.0b Anti-ice, electrical, fuel pumps (no training)
- [x] **Three commands** with handlers and per-aircraft plans from the part B
  findings: the 737's battery opens its guard first, its tank pumps are
  written; the A330's fuel pumps, batteries and generators are its own
  buttons (never `sim/fuel/fuel_pump_N_*`, its engine feed), its probe and
  window heat ON or AUTO; the King Air's windshield and generators are
  stepped, its boots run one cycle a press, its engine anti-ice is the ice
  vanes. Every name checked: 500/500 X-Plane names, 119/119 aircraft
  controls (its guard command is made in a loop: the checker now reads those).
- [x] **Safety:** power and fuel pumps off in flight ask to confirm.
- [x] **Grammar:** 276 → 297 words, real voice unchanged (v25 89 / 28 / 1);
  no "de ice" (the recogniser put "de" in unsaid).
- [x] `GOLD_TIER3` grows to 31 phrases.

### T3.1 Data, training, gate, runner
- [x] **Runner scenarios** (`simtest/tier3.py`, `simtest run tier3 <aircraft>`):
  26 / 29 / 31 / 34 steps on the Baron / King Air / 737 / A330, each switch
  checked the way its aircraft shows it; power and fuel pumps off held and
  cancelled. End to end on the fake sim in CI: the whole Baron checklist, and
  the lights, signs, anti-ice, speed brake and autobrake parts on the 737 and
  A330.
- [x] **Data v14:** 137,720 rows, 35 intents (8 new, 4,000 each): train
  105,448, dev 7,845, test 24,427; no text with two labels.
- [x] **First v26 failed the gate** (2026-11-14). It was the average of three
  runs of the v24 recipe on data v14 (`--name v26-a/b/c --run-seed 1/2/3`),
  kept as `v26-soup1`. Gates 2, 5 and 6 passed (old data 15,938/308/0, the
  same as v25). The failures:
  - **Gate 1** (anti-ice 0.662, lights 0.737) and **gate 3** (gold "boots on"
    went to the flaps, "beacon off" to an autopilot mode) had one cause.
    `{ice=boots} {switch}` and `{light=beacon} {switch}` were the only
    templates pinning those words, and the hash held both out, so the model
    never saw them. Every other held-out template scored 100%.
  - **Gate 4** (real voice 87/30/1 vs 89/28/1): every right lost had a stray
    word beside a whole command, a word that since Tier 3 belongs to another
    command: "ice director on", "speed autopilot on", "fuel com two to …"
    (plus a soup-only low confidence on "v confirm engine one off").
  - The clean dev set was saturated (1.0, even on confidence), so it couldn't
    show any of this. Ideas were compared on a "heard" dev set instead:
    1,600 dev rows spoken by four generated voices and decoded with the
    grammar. It is not a gate.
- [x] **Data v15** (137,720 rows: train 105,723, dev 7,716, test 24,281):
  - pinned words get training templates of their own, and a test holds every
    command to it;
  - a "stray" noise (7,461 rows): a command word that can't make its own
    command with the rest of the sentence, before, in place of the first
    word of, or after a command;
  - "propeller" for prop heat: two of four generated voices hear "prop heat"
    as "probe heat" (the pitot); none mishear "propeller".
  - **Grammar:** "anti ice", "fuel pump(s)" and "fuel boost" are entered whole.
    Vosk makes a bigram model of the lines, so the pair is likely, though
    the words can still be heard alone. Real voice is rebuilt: **v25 91 / 27 /
    0** (the grammar fixed its one wrong, "autopilot new on" → "two"). The
    heard dev set is unchanged.
- [ ] **v26** = the average of `v26-d/e/f` (run seeds 1/2/3) on data v15;
  then the gate against v25, install (owner's approval) and the Tier 3
  checklist in the sim.
- Found on the way, not part of v26:
  - **#54:** a held switch could be confirmed by another switch command
    (the token lacked the switch slots).
  - **Clipped first word:** the mic opens ~85 ms after the key press. With
    150 ms cut from the generated clips, wrong actions went 29 → 76: "vertical
    speed thirty two hundred" → "speed …" (320 kt), "nav two transfer" →
    COM swap, "deactivate autopilot" → "activate". No model fixes a heard
    text that is another valid command. Pre-roll audio would; it needs the
    owner's decision (the mic stays open).

Checklist: preflight battery/avionics/beacon/nav, taxi lights,
landing + strobes on takeoff, pitot heat, fuel pump, yaw damper, seatbelt
sign, speed brakes armed for landing, autobrake 3, all lights off after
shutdown.

---

## Tier 4: questions (read-only) → v27

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

## Phase F: flight context → after T4

Today Vimaan knows the aircraft only when a command asks (each safety rule
reads its own datarefs) and logs its state at intervals. Phase F keeps a
model of the flight, updated every second:

- **Phase of flight**: parked, taxi, takeoff roll, initial climb, climb,
  cruise, descent, approach, landing, rollout, from on-ground, ground speed,
  vertical speed, height above ground, gear, flaps and thrust.
- **Envelope margins**: airspeed against stall, flap, gear and maximum speeds
  (from the `.acf`, as the flap limits already are), Mach, bank, sink rate,
  configuration for the height (gear down by 1,000 ft on approach).
- **Uses**: safety rules read the phase instead of separate datarefs;
  optional callouts ("Speed", "Gear", "Approaching flap limit"), ranked and
  rate-limited so they don't chatter; Tier 4 answers and phase C get it as
  context. Phase changes are logged and shown on the dashboard's Flights tab.
- **Tests**: recorded flights (the A3 takeoffs and A4 landings) replayed
  through the tracker; the transitions must match. Pure Python, no training.

## Phase C: conversational copilot → after F

The trained model stays the fast path for commands (6–11 ms, offline,
predictable). What it doesn't take as a command (questions, conversation, low
confidence) can go to a language model given the phase F context and tools
mapped to the existing handlers. Every action it proposes goes through the
same validators, safety rules and read-backs: it never writes to the sim
itself. Also a better speaking voice and free-speech recognition (Whisper).
Opt-in. Decisions for the owner: which model (cloud, or local if X-Plane leaves
GPU room), what is sent (the transcript and flight state only), cost, and the
latency allowed.

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
