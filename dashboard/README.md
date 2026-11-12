# Vimaan Dashboard

A lightweight web UI for monitoring training runs, picking a trained
checkpoint, sending live predictions, uploading datasets, kicking off
new training jobs, and reviewing flights flown with Vimaan — all in one place.

## Install

```bash
python3 -m pip install -r requirements-dashboard.txt
# For prediction + training you also need the ML stack:
python3 -m pip install -r requirements.txt   # torch, transformers, ...
```

## Run

```bash
python3 -m dashboard               # http://127.0.0.1:8765
python3 -m dashboard --reload      # auto-reload during dev
python3 -m dashboard --host 0.0.0.0 --port 9000
```

## What it does

Four tabs. `?tab=training` (or `inference`, `evals`, `flights`) links straight
to one; `?run=v18&compare=v17` pins a run and the run to compare it with;
`?flight=20261107_203741` opens one flight; `?theme=light` picks the theme.

### Training: a run, live

Every run is read from its metrics stream,
`ML/training_runs/<version>/metrics.jsonl`, so a run started in a terminal
shows up exactly like one started from the form. The page follows the newest
run by default and switches when a new one starts.

| Panel | Shows |
| --- | --- |
| Run header | status (live / finished / failed / stopped / interrupted), phase (training, validating, saving), progress through all epochs and the current one |
| Tiles | epoch, elapsed, time left (if every epoch runs), speed in steps/s and examples/s, best validation loss and its epoch, epochs since the best (against patience), latest intent accuracy and slot F1 |
| Training loss | every step (faint), smoothed (EMA over ~100 steps), and the comparison run's smoothed curve, on a shared epoch axis |
| Loss per epoch | train and validation loss, with the best epoch marked, plus the comparison run's validation loss |
| Validation accuracy | intent accuracy and slot F1 per epoch, against the comparison run |
| Gradient norm | sampled every 25 steps |
| By intent / by slot | the latest epoch's accuracy per intent and F1 per slot. The bar is the error rate (worst first), with the change since the previous epoch |
| Epochs | every epoch as a table: losses, scores, time, validation time |
| Run data | split, hyperparameters, intent distribution |
| Training log | the dashboard job's output, or the run's `train.log` |

"Interrupted" means no new line for 3 minutes and no end record: the process
was killed or the machine slept. A crash is recorded as "failed" with its
message, and Ctrl-C as "stopped".

### Evals: every model version's record

| Panel | Shows | From |
| --- | --- | --- |
| Gate results by version | intent accuracy and slot F1 on new phrasings across versions, and a table with all six gates per version | `<version>/gate.json` (`scripts/model_gate.py`) |
| Gates | each gate's pass or fail and its numbers, plus backend parity | `gate.json`, `onnx/parity.json` |
| New phrasings by intent | the version's accuracy per intent, with the baseline as a tick | `gate.json` |
| Real voice by flight | right / rejected / wrong per recorded flight, against the baseline | `gate.json` |
| Where it differs | the examples the gate lists: baseline right but candidate wrong or rejected, gold failures, negatives acted on | `gate.json` |

### Flights: what happened in the sim

Every X-Plane session the plugin logged, from `~/Vimaan_Logs` (set
`VIMAAN_LOG_DIR` to read another folder). The page follows the newest flight,
and a flight in progress updates live (server-sent events, once a second
while its log grows).

| Panel | Shows | From |
| --- | --- | --- |
| Flights | date, length, aircraft, model, commands, done / not done / not understood, autopilot drops not made by Vimaan, problems, live or ended | `vimaan_<session>.events.jsonl` |
| Tiles | the same for one flight, plus speech-to-action time (median, p95) | |
| Timeline | each push-to-talk command: what was heard, the intent and values understood, Vimaan's reply, the result, speech-to-action time, and ▶ the recording. Click a row for what was sent to X-Plane, what was read back afterwards, and notes (safety holds, overrides, refused values, the aircraft 10 s and 30 s later) | the command's events, `audio/<session>_<cmd>.wav` |
| | autopilot changes, marked Vimaan / not Vimaan / the probe, and disconnects with the plugin's diagnosis (cause, stick input, null zone, what the aircraft was doing) | `autopilot_state_change`, `autopilot_disconnect` |
| | startup, model, aircraft, joystick null zone, self test, probe, freezes and errors | |

Filters: everything, only what needs a look, only commands that worked, only
the autopilot; a word search; probe steps and other details are hidden until
asked for.

### Inference

Model selector, live prediction (intent, confidence, slots), and the model's provenance.

## What the trainer records

`ML/train_nlu_model.py` streams one JSON object per line:

| Record | When | Fields |
| --- | --- | --- |
| `meta` | start | version, device, dataset, split, hyperparameters, steps per epoch, intent distribution |
| `step` | every step | step, epoch, loss, learning rate, elapsed; `grad_norm` every 25th step |
| `phase` | validation / saving starts | phase, epoch |
| `epoch` | every epoch | train and validation loss, intent accuracy, slot F1, `per_intent` {acc, n}, `per_slot` {f1, n}, validation seconds, epochs without improvement, patience |
| `done` / `error` / `stopped` | the end | best validation loss and epoch, whether it stopped early; or the error |

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/runs` | every run with its summary (status, progress, speed, ETA, best epoch) |
| GET | `/api/runs/{name}` | one run: summary, downsampled steps, gradient norms, epochs, log tail |
| GET | `/api/runs/{name}/stream` | server-sent events: `snapshot`, then `records` as lines arrive, `summary` every 5 s |
| GET | `/api/evals` | every model version's gate summary, parity and run summary |
| GET | `/api/evals/{version}` | one version's full `gate.json`, parity and manifest |
| GET | `/api/flights` | every logged session with its summary, newest first |
| GET | `/api/flights/{session}` | one session: summary, timeline items, problems |
| GET | `/api/flights/{session}/stream` | server-sent events: `flight` now and each time the log grows, until the session ends |
| GET | `/api/flights/{session}/audio/{cmd}` | one command's recording (WAV) |
| GET | `/api/models` | list checkpoints + parsed manifests |
| GET | `/api/datasets` | list `.jsonl` datasets (final + uploads) |
| POST | `/api/datasets/upload` | upload a `.jsonl` (validated row-by-row) |
| POST | `/api/predict` | `{model_path, text}` → `{intent, confidence, slots}` |
| POST | `/api/train/start` | start a training run (409 if one is in flight) |
| POST | `/api/train/stop` | terminate the running job |
| GET | `/api/train/status` | the dashboard job's state and log |

## Security note

The dashboard binds to `127.0.0.1` by default and assumes a trusted user
on the same machine. It runs subprocesses (`python3 ML/train_nlu_model.py`)
and reads/writes files under `ML/`, so don't expose it to the public
internet without adding auth + a reverse proxy.

## Tests

```bash
python3 -m pytest tests/test_dashboard.py tests/test_run_metrics.py tests/test_eval_metrics.py tests/test_flights.py
```
