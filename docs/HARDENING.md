# Hardening Log

A running record of security/robustness hardening passes. Newest first.

---

## 2026-06-29 — Hardening sweep (branch `chore/hardening-sweep`)

Full-codebase audit + remediation. 145 tests pass (11 new), ruff + format clean.

### Shipped (4 commits)

| Area | Change | Commit |
| --- | --- | --- |
| **Dependencies** | Pinned `requirements.txt` with lower bounds + next-major caps (so a breaking major like transformers 4→5 can't land silently). `torch>=2.6` chosen as the floor because that's where `torch.load` defaults to `weights_only=True`. Split legacy-only heavy deps (`openai-whisper`, `joblib`, `sentence-transformers`) into `requirements-legacy.txt`. | `chore(deps)` |
| **Dependencies** | Removed the runtime `pip install num2words` fallback from `ML/data/clean_*` + `verify_dataset.py` (supply-chain / repro footgun); now a clean `ImportError`. | `chore(deps)` |
| **Security** | `dashboard /api/predict` accepted an arbitrary `model_path` → `from_pretrained` + `torch.load` (traversal / arbitrary-load, reachable on localhost via DNS-rebinding). Added `safe_model_path()` to confine it under `MODELS_ROOT`, mirroring how `/api/train/start` validates its dataset. | `fix(security)` |
| **Security** | `model_loader.py` now passes `torch.load(..., weights_only=True)` explicitly (defence-in-depth for the whole `torch>=2.6` floor). | `fix(security)` |
| **Correctness/UX** | Plugin now handles non-actionable intents (`None`, `ask_status_generic`, `ask_time`, `chit_chat_greeting`) gracefully instead of "Command not found". `None` = silent reject; others get a short ack. No retrain needed. | `refactor(nlu)` |
| **Hygiene** | Trimmed dead `IMPLICIT_STATE_INTENTS` (`toggle_autopilot_3`/`engine_3`/`engine_4`) + added a schema-consistency test. | `refactor(nlu)` |
| **Repo** | `git rm --cached` the v1–v9 weights + generated datasets (93 files). They were tracked despite `.gitignore`/docs claiming otherwise — the ignore rules were added *after* the files were committed. Working-tree copies preserved. | `chore(repo)` |

Note: the pre-commit `check-added-large-files` (2 MB) + `detect-private-key` guards already existed — the weights pre-date them. No pre-commit change was needed.

### Outstanding — owner actions

1. **🔴 Revoke the HuggingFace token.** `scratch/token_HF.txt` holds a live `hf_…` token. It was never committed (verified) and `scratch/` is gitignored, but it's a usable credential on disk. Revoke at <https://huggingface.co/settings/tokens>, then delete the file. (Master-plan B-002, still open.)
2. **🟠 Decide on a git history rewrite.** `git rm --cached` stops *future* tracking but the ~2.2 GB of weight/dataset blobs remain in history. Reclaiming the space needs `git filter-repo` (or BFG), which **rewrites every commit SHA** — all collaborators must re-clone. Defer until the team agrees; this is the only way to (a) shrink clones and (b) remove already-published weights from history.

### Not done (proposed follow-ups)

- End-to-end inference regression test (CI skips torch, so `predict()`/model-load is untested).
- `ask_time` could answer real sim Zulu time instead of a placeholder ack.
- Tighten the globally-ignored bare-`except` (E722) in normalization/postprocessor.
- Remove dead `ML/utils/debug.py`.
