# Features

A snapshot of what Project Vimaan currently does. For what's *planned*, see [docs/ROADMAP.md](docs/ROADMAP.md).

## Current

### Voice → action
- Push-to-talk capture (`Z` hotkey) inside X-Plane.
- Google Web Speech transcription via the `SpeechRecognition` library.
- Phonetic and word-number normalization (`"flight level two five zero"` → `flight_level=250`).

### Joint NLU model
- DistilBERT-based **joint intent + slot** classifier (`JointIntentAndSlotModel`).
- Versioned training pipeline that never overwrites prior runs.
- 13 intents covering autopilot, gear, flaps, engines, parking brake, flight director, and COM tuning.
- Numeric slot guards: altitude, heading, flight level, frequency all range-checked.

### Data pipeline
- Schema-driven synthetic dataset generation.
- Two-way paraphrase augmentation (Pegasus + FLAN-T5).
- Automatic cleaning to drop off-intent paraphrases.
- Final merge + dedup + shuffle stage.
- Word-form augmentation ("2" ↔ "two").

### Evaluation
- Per-version metrics: intent accuracy, slot F1, confusion matrices.
- Batch evaluator across all model versions.
- Matplotlib visualizations (per-intent F1, intent confusion, etc.).
- Canned regression command tester (`ML/command_tester.py`).

### Plugin
- X-Plane 12 + XPPython3 integration.
- Auto-loads the latest model version on plugin start.
- Per-session log file under `~/Desktop/Vimaan_Logs/`.
- Spoken acknowledgement via `xp.speakString`.

### Project hygiene
- Comprehensive `.gitignore` (models, datasets, secrets, OS noise).
- Documentation in `docs/`: architecture, bugs, roadmap, structure, version control, production checklist.
- Upstream/origin remote topology documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## Out of scope (today)

- Real avionics / DO-178C certification — **simulation only.**
- Offline STT (Whisper exists in legacy form but is not wired into the active plugin yet).
- Multi-turn dialog.
- Per-aircraft profile switching.
