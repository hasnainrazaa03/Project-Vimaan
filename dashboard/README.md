# Vimaan Dashboard

A lightweight web UI for monitoring training runs, picking a trained
checkpoint, sending live predictions, uploading datasets, and kicking off
new training jobs — all in one place.

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

| Panel | What it shows | Backed by |
| --- | --- | --- |
| Inference | Model selector + live prediction (intent, confidence, slots) | `dashboard.predictor.predict` ← `vimaan_nlu.predict` |
| Train a model | Dataset selector + upload (.jsonl) + hyperparameters + start/stop | spawns `ML/train_nlu_model.py` with the chosen flags |
| Loss curve | Per-epoch train/val loss, updated live during training | parses `Epoch N - Average ... Loss:` lines from stdout |
| Training log | Last 200 stdout/stderr lines | `dashboard.training.TrainingManager` |
| Status pill | `idle` / `running` / `finished` / `failed` | `/api/train/status` |

The dashboard polls `/api/train/status` every 1.5s while training is
running and stops polling once the job finishes (no extra battery drain
when idle).

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/models` | list checkpoints + parsed manifests |
| GET | `/api/datasets` | list `.jsonl` datasets (final + uploads) |
| POST | `/api/datasets/upload` | upload a `.jsonl` (validated row-by-row) |
| POST | `/api/predict` | `{model_path, text}` → `{intent, confidence, slots}` |
| POST | `/api/train/start` | start a training run (409 if one is in flight) |
| POST | `/api/train/stop` | terminate the running job |
| GET | `/api/train/status` | full state snapshot (logs, metrics, checkpoints) |

## Security note

The dashboard binds to `127.0.0.1` by default and assumes a trusted user
on the same machine. It runs subprocesses (`python3 ML/train_nlu_model.py`)
and reads/writes files under `ML/`, so don't expose it to the public
internet without adding auth + a reverse proxy.

## Tests

```bash
python3 -m pytest tests/test_dashboard.py
```
