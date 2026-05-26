# Changelog

All notable changes to this project are documented in this file.

The format is inspired by Keep a Changelog and this project aims to follow Semantic Versioning for tagged releases.

## [Unreleased]

### Added
- Comprehensive `README.md` rewrite with architecture diagram, command table, install + usage instructions, and links to all docs.
- `requirements.txt` declaring every runtime / training dependency.
- `.gitignore` blocking models, datasets, eval outputs, secrets, OS noise, and bytecode.
- `docs/ARCHITECTURE.md` — full system design and dataflow walkthrough.
- `docs/BUGS.md` — audited list of known issues with severity and triage table.
- `docs/ROADMAP.md` — near-term, medium-term, long-term feature plans.
- Refreshed `docs/PROJECT_STRUCTURE.md` to match the new layout.
- Refreshed root `FEATURES.md`.

### Changed
- **Folder reorg for production readiness:**
  - `PLUGIN/` → `plugin/` (lowercase) with active plugin promoted to `plugin/PI_VimaanCoPilot.py` and old code under `plugin/legacy/`.
  - Root-level `xplane_vimaan_copilot.py` → `plugin/PI_VimaanCoPilot.py`; path resolution updated to `../ML`.
  - Root-level `AI_CoPilot.py` → `plugin/legacy/AI_CoPilot.py`.
  - `MISC/git_cmds.txt` → `docs/git_workflow.md`; `MISC/` removed.
  - Personal notes (`first run.txt`, `potential improvements.txt`, `token HF.txt`) moved to `scratch/` (gitignored).
- Repository history reconciled and standardized under a single source-of-truth remote (`origin`).

### Removed
- `ML/core/normalization_backup.py` (duplicate).
- `ML/utils/debug.py`, `ML/utils/debug_new.py` (dead code).
- Tracked `.pyc` files under `ML/utils/__pycache__/`.
- Tracked large dataset files under `ML/datasets/01_base/` (regenerable; now gitignored).

### Security
- A local token file was moved to `scratch/` and excluded from version control.

## [2026-05-25]

### Changed
- Safe migration completed:
  - Previous local folder snapshot backed up.
  - Repository initialized for continued independent development under a single remote.
