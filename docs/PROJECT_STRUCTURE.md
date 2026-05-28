# Project Structure

> Updated after the production-readiness reorg. Every directory has one job; cross-directory deps are documented inline.

## Top-Level

```text
ProjectVimaan/
├── README.md                ← entry point
├── CHANGELOG.md
├── CONTRIBUTING.md
├── FEATURES.md
├── requirements.txt
├── .gitignore
│
├── ML/                      ← NLU package: data, training, eval, inference
├── plugin/                  ← X-Plane plugin (active + legacy)
├── docs/                    ← all engineering documentation
└── scratch/                 ← personal notes (gitignored)
```

## `ML/`

NLU package. Importable as a package when `ML/` is on `sys.path` (the plugin inserts `<repo>/ML` at load time).

```
ML/
├── vimaan_nlu/
│   ├── __init__.py
│   ├── model.py             ← JointIntentAndSlotModel (DistilBERT + 2 heads)
│   ├── model_loader.py      ← ModelLoader.load_all(): tokenizer + body + heads + maps
│   ├── inference.py         ← predict(text, ...) → {intent, slots, confidence, ...}
│   ├── normalization.py     ← phonetic numbers, word→digit, decimals
│   └── postprocessor.py     ← slot range clamps, implicit on/off, value cleanup
│
├── utils/
│   ├── __init__.py          ← re-exports file_utils helpers
│   └── file_utils.py        ← versioned path helpers (vN/, get_latest_model_path)
│
├── config/
│   └── schema_config.py     ← INTENTS / SLOTS / SYNONYMS / RANGES (source of truth)
│
├── data/                    ← runnable scripts (cwd-independent via script_dir)
│   ├── generate_slot_dataset.py
│   ├── generate_data_pegasus.py
│   ├── generate_data_flan_t5.py
│   ├── clean_pegasus_dataset.py
│   ├── clean_flan_t5_dataset.py
│   ├── dataset_summary.py
│   └── verify_dataset.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── batch_evaluator.py
│   ├── visualizer.py
│   ├── run_single.py
│   ├── run_all.py
│   └── results/             ← (gitignored) per-model metrics + plots
│
├── train_nlu_model.py       ← entry point: trains joint model → models/vN/
├── merge_datasets.py
├── augment_with_word_forms.py
├── predict.py               ← interactive REPL
├── command_tester.py        ← canned regression check
│
├── datasets/                ← (gitignored) all stages 01_* … 06_*
└── models/                  ← (gitignored) vimaan_nlu_model_best/vN/
```

## `plugin/`

X-Plane integration.

```
plugin/
├── PI_VimaanCoPilot.py      ← active plugin (DistilBERT NLU + SpeechRecognition)
└── legacy/
    ├── AI_CoPilot.py        ← v1: joblib + MiniLM embeddings (BUGS B-001)
    ├── PI_Vimaan_Whisper.py ← Whisper-based prototype
    └── Plugin_Ref.py        ← reference snippets
```

The active plugin imports `from vimaan_nlu.model_loader import ModelLoader` after inserting `<repo>/ML` (resolved as `../ML` relative to the plugin file) on `sys.path`. When installing into X-Plane, copy both `PI_VimaanCoPilot.py` **and** the `ML/` tree into `PythonPlugins/`.

## `docs/`

```
docs/
├── ARCHITECTURE.md          ← system design + dataflow
├── BUGS.md                  ← known issues with severity & triage table
├── ROADMAP.md               ← planned features
├── PROJECT_STRUCTURE.md     ← this file
├── VERSION_CONTROL.md       ← branch/remote workflow + quick git command reference
└── PRODUCTION_CHECKLIST.md  ← pre-release gates
```

## `scratch/`

Personal sandbox. **Always gitignored.** Use it for TODO scratchpads, experiment outputs, locally rotated secrets (then delete). If something matures into a real artifact, move it to the appropriate dir and reference it from `docs/`.

## Cross-cutting conventions

- **Versioning:** never overwrite. All artifact-producing scripts use `get_next_version_path()` from `ML/utils/file_utils.py`.
- **Imports:** scripts under `ML/` resolve sibling paths via `script_dir`; do not break this without updating the call sites.
- **Secrets:** never commit. See `.gitignore` and [README → Security](../README.md#security).
- **Large artifacts:** `ML/datasets/`, `ML/models/`, `ML/evaluation/results/`, `Vimaan_Logs/` are all gitignored.
