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
| **Repo** | Untracked the v1–v9 weights + generated datasets (they were tracked despite `.gitignore`/docs claiming otherwise — the ignore rules were added *after* the files were committed). Working-tree copies preserved. Initially done via `git rm --cached`; **subsumed by the history rewrite below**, which removes them from all history. | `chore(repo)` (pruned by rewrite) |
| **Repo / LFS** | **History rewrite** (`git filter-repo`) purged `ML/models/**` + generated datasets from all commits; **Git LFS retired** in favour of GitHub Releases for weight distribution (`scripts/publish_model.sh` / `fetch_model.sh`, `docs/MODEL_DISTRIBUTION.md`). `git lfs prune` reclaimed the local LFS cache. `main` + the PR branch were force-pushed. | rewrite + `chore(lfs)` |

Notes:
- The weights are in **Git LFS** (`.gitattributes`); the 2.2 GB lived in `.git/lfs`, not pack history (which was ~12 MB). So the rewrite reclaims local space and cleans history, but GitHub's **remote** LFS store is not auto-GC'd — see `docs/MODEL_DISTRIBUTION.md`.
- The pre-commit `check-added-large-files` (2 MB) + `detect-private-key` guards already existed — the weights pre-date them. No pre-commit change was needed.

### Outstanding — owner actions

1. **🔴 Revoke the HuggingFace token.** `scratch/token_HF.txt` holds a live `hf_…` token. It was never committed (verified) and `scratch/` is gitignored, but it's a usable credential on disk. Revoke at <https://huggingface.co/settings/tokens>, then delete the file. (Master-plan B-002, still open.)
2. **🟠 Reclaim GitHub's remote LFS storage.** The history rewrite removed the LFS pointers, but GitHub keeps the ~2.2 GB of LFS objects until the repo is deleted/recreated or GitHub Support purges them. Only needed if the LFS quota matters.
3. **🟠 Collaborators must re-clone.** The force-push rewrote every SHA; Aryan and Vyom should re-clone rather than pull.

### Not done (proposed follow-ups)

- End-to-end inference regression test (CI skips torch, so `predict()`/model-load is untested).
- `ask_time` could answer real sim Zulu time instead of a placeholder ack.
- Tighten the globally-ignored bare-`except` (E722) in normalization/postprocessor.
- Remove dead `ML/utils/debug.py`.
