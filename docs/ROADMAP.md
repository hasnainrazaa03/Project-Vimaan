# Roadmap

A prioritized list of features and improvements. Status legend: 🟢 ready to pick up · 🟡 needs design · 🔵 research · ✅ done.

---

## Near term (next milestone)

### 🟢 R-01 — Threaded speech recognition
Move `recognize_google()` off the X-Plane main thread (see [BUGS B-006](BUGS.md#-b-006--microphone-callback-can-block-indefinitely)). Use a `queue.Queue` and an `xp.createFlightLoop` poller.

### 🟢 R-02 — Explicit on/off semantics
Resolve [B-005](BUGS.md#-b-005--toggle-off-intents-map-to-a-toggle-command-not-an-explicit-off-command). Read the current dataref; only toggle when state differs from request. Applies to autopilot, FD, engines, parking brake, gear (where applicable).

### 🟢 R-03 — `requirements.txt` honoured everywhere
Strip auto-`pip install` calls from `ML/core/normalization.py` and `ML/train_nlu_model.py` (B-003 / B-004). Document a single venv setup.

### 🟢 R-04 — Rename `core` → `vimaan_nlu`
Eliminate the `sys.path` hack and namespace collision risk (B-007). Add backward-compat shim for one release.

### 🟢 R-05 — Audible confirmation read-back
After every successful command, speak a structured confirmation through `xp.speakString` (or pyttsx3) — e.g. "Heading two seven zero, set." Helps catch mis-recognitions.

### 🟢 R-06 — Confidence thresholding & clarification
If `confidence < 0.7`, the plugin should refuse and ask "Say again?" instead of executing. Plumb `confidence` from `predict()` into the handler dispatch.

---

## Medium term

### 🟡 R-07 — Add more intents
- Radio: NAV1/NAV2/ADF tuning, transponder squawk codes
- Lights: landing/taxi/strobe/beacon/nav
- Comm: VHF channel select, audio panel routing
- Trim: pitch trim up/down/neutral
- Flight controls: speed brake, autobrake setting
- Aircraft systems: APU start, fuel pump, anti-ice

Each requires (a) schema entries in `config/schema_config.py`, (b) template additions, (c) full data-pipeline re-run, (d) plugin handler.

### 🟡 R-08 — Per-aircraft profiles
Datarefs differ between Cessna 172 vs. Airbus A320 vs. Boeing 737. Add a `profiles/` directory mapping intents to datarefs per aircraft. Auto-detect via `sim/aircraft/view/acf_ICAO`.

### 🟡 R-09 — Offline STT via Whisper
Port the existing `plugin/legacy/PI_Vimaan_Whisper.py` capture pipeline into the main plugin as a configurable backend so the system works without internet.

### 🟡 R-10 — Slot value RAG / clarification
When the user says "tune com to one two one" with no decimal, ask for the decimal portion rather than guessing.

### 🟡 R-11 — Plugin packaging script
A `scripts/install_plugin.sh` that copies the right files into `<X-Plane>/Resources/plugins/PythonPlugins/` and verifies the model exists.

---

## Long term

### 🔵 R-12 — Quantized / smaller model
Export the joint model to ONNX or TFLite, quantize to int8. Goal: < 50 MB on disk and < 100 ms inference on CPU so the plugin doesn't need a GPU.

### 🔵 R-13 — Multi-turn dialog
Track context across utterances: "Set heading 270." → "Now altitude 12000." Stateful slot carryover for ATC-style read-backs.

### 🔵 R-14 — Voice biometrics / wake word
Replace push-to-talk with "Hey Vimaan" wake-word detection (e.g. `openWakeWord`).

### 🔵 R-15 — ATC chatter integration
Generate two-way ATC exchanges: read clearances back, request altitude, declare emergencies.

### 🔵 R-16 — Reinforcement learning from pilot corrections
Log every misclassification + pilot's manual override. Periodically fine-tune on the correction stream.

---

## Engineering / tooling

### 🟢 R-17 — Tests + CI
Add `pytest` with at least:
- unit tests for `normalization.py`, `postprocessor.py`
- a smoke test that loads the latest model and runs `command_tester.py`
- a GitHub Actions workflow on push/PR

### 🟢 R-18 — Dataset manifest
A small `datasets/MANIFEST.json` recording, for each stage, the row count, source script, and SHA-256 of the output file. Catches silent regressions.

### 🟡 R-19 — Type hints + `ruff` + `mypy`
Run `ruff check .` and `mypy ML/` clean.

### 🟡 R-20 — Documentation site
Render `docs/` via MkDocs or similar; publish to GitHub Pages.

---

## Out of scope (for now)

- Real avionics certification / DO-178C. **Project Vimaan is for simulation only.**
- Speech synthesis of full ATC voice (requires licensed voice models).
