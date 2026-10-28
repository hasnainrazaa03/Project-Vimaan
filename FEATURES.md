# Features

A snapshot of what Project Vimaan currently does. For what's *planned*, see [docs/ROADMAP.md](docs/ROADMAP.md).

## Current

### Voice → action
- Push-to-talk capture (`Z` hotkey) inside X-Plane, with speech capture and
  transcription run **off the simulator's main thread** so the sim never blocks.
- **Selectable speech backend** via `VIMAAN_STT_BACKEND`: `google` (online, default),
  or `vosk` / `whisper` / `sphinx` for fully **offline** operation — see
  [docs/OFFLINE_STT.md](docs/OFFLINE_STT.md).
- **Microphone health check** — a dead or muted mic is detected from the captured
  audio's RMS level and reported, instead of failing with a confusing "didn't catch that".
- Aviation number normalization: phonetic digits (`"two seven zero"` → `270`),
  ATC group form (`"one five thousand"` → `15000`), spoken decimals
  (`"one one eight decimal seven five"` → `118.75`), and compound cardinals
  (`"seven thousand five hundred"` → `7500`).
- **Phonetic read-back** of every executed command through `xp.speakString`
  ("Heading two seven zero", using `niner` for 9) for closed-loop confirmation.

### Joint NLU model
- DistilBERT-based **joint intent + slot** model (`JointIntentAndSlotModel`): a shared
  encoder with an intent-classification head and a BIO token-tagging head, trained on a
  summed cross-entropy loss.
- **17 intent classes** — 13 actionable commands (autopilot, altitude, flight level, COM
  tuning, gear, flaps, flight director, parking brake, engines), 3 conversational intents,
  and an out-of-scope `None` reject class that improves precision on real commands.
- **6 slot types** (`altitude`, `degrees`, `flight_level`, `frequency`, `com_port`,
  `state`) over 13 BIO tags.
- **98.9% intent accuracy, 0.967 macro-F1, and 0.99 slot F1 on a held-out test set**
  that ships with the checkpoint (`holdout_test.jsonl`) so results are reproducible.
- **~5.9 ms** inference (p95 6.1 ms) on CPU; **~3.1 ms** via the ONNX backend.
- Versioned training pipeline that never overwrites prior runs.

### Inference backends
- **torch** (default), optionally **dynamic int8-quantized** — ~48% smaller weights with
  no measured macro-F1 change.
- **ONNX Runtime** backend (`VIMAAN_BACKEND=onnx`) — ~2× faster than torch fp32 with
  verified numerical parity. It duck-types the torch model, so every handler is unchanged.

### Safety & validation
- **Declarative slot validators** (range / enum / wrap / reject) as a single source of
  truth, enforced at dispatch — heading wraps into 0–360°, out-of-range altitudes and
  frequencies are rejected before any dataref is written.
- **Flight-context safety interlocks** — gear up below 1000 ft AGL, engine shutdown in
  flight, and parking brake above 5 kt require the pilot to repeat the command to confirm.
  Rules **fail safe**: if a relevant dataref is unavailable, the situation is treated as
  risky rather than silently skipped.
- **Confidence floor** — predictions below 0.55 are never dispatched; the copilot asks the
  pilot to repeat instead of acting on a low-certainty guess.
- **Idempotent state reconciliation** — binary systems read their current dataref and act
  only when it differs from the request, so "autopilot off" can never turn it on.

### Data pipeline
- Schema-driven synthetic dataset generation from a single source-of-truth config.
- Two-strategy LLM augmentation: **Pegasus** paraphrasing and **FLAN-T5** generation.
- Automated cleaning that drops paraphrases which lost a slot value or a numbered
  instance (e.g. the `2` in `toggle_engine_2`).
- Offset-mapping BIO label alignment with word-boundary anchoring.
- Final merge + dedup + shuffle + normalization stage.

### Training & evaluation
- CLI-flagged trainer (`--dataset --epochs --lr --batch-size --max-length --device` …)
  with **CUDA > MPS (Apple GPU) > CPU** auto-selection and early stopping.
- **Stratified 3-way split** (70/15/15) with the held-out test rows persisted inside the
  checkpoint, so evaluation measures true generalization.
- **Provenance manifests** on every checkpoint: dataset SHA-256, row count, per-intent
  distribution, git SHA, framework versions, and all hyperparameters.
- Per-version metrics: intent accuracy, macro-F1, slot pair-F1, confusion matrices,
  Cohen's κ, confidence and latency distributions.
- **Live training telemetry** — a JSONL metrics stream plus a dependency-free real-time
  monitor (`ML/training_monitor.py`) with live loss and validation charts.
- Error-analysis script, batch evaluator across all versions, matplotlib visualizations,
  and a canned regression command tester.

### Plugin & deployment
- X-Plane 12 + XPPython3 integration; auto-loads the latest model version on start.
- **Self-test hotkey (Ctrl-Z)** — runs canned utterances through the full NLU → dispatch
  path with speech bypassed, to verify the model and handlers without a microphone.
- **Installer script** (`scripts/install_plugin.py`) that copies only the runtime pieces
  into `<X-Plane>/Resources/plugins/PythonPlugins/`, with `--dry-run`.
- Model checkpoints distributed as **GitHub Release assets**, not committed to the repo
  (`scripts/fetch_model.sh` / `publish_model.sh`).
- Per-session log file under `~/Vimaan_Logs/`.

### Web dashboard
- FastAPI + vanilla-JS single-page app (no build step): checkpoint browser with parsed
  manifests, live inference, row-validated dataset upload, and start/stop training with
  live loss curves and run metrics.
- Binds to `127.0.0.1` by design — single-user, no auth.

### Project hygiene
- **347 automated tests** (25 modules) running in under 10 s with no simulator,
  microphone, GPU, or network required.
- GitHub Actions CI on Python 3.11 + 3.12: ruff lint + format, **mypy**, compileall, and
  pytest with coverage. Conventional Commits enforced by commitlint.
- Pre-commit hooks including `detect-secrets`, large-file and private-key guards.
- Pinned dependencies split across runtime / dev / dashboard / legacy requirement files.
- Documentation in `docs/`: architecture, data sources, hardening log, model distribution,
  offline STT, bugs, roadmap, structure, version control, production checklist.

## Out of scope (today)

- Real avionics / DO-178C certification — **simulation only.**
- Multi-turn dialog and per-aircraft profile switching.
- Wake-word activation (push-to-talk only).
- **Real-speech evaluation** — all reported metrics are measured on held-out *synthetic*
  data. Real-ASR performance is not yet measured; see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).
