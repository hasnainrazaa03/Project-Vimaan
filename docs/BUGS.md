# Known Bugs & Footguns

> Audit performed against commit head as of this snapshot. Severity:
> 🔴 critical — breaks main flow · 🟠 high — easy to hit · 🟡 medium — surprising · ⚪ minor.

---

## 🔴 B-001 — `plugin/legacy/AI_CoPilot.py` references a model that does not exist
- **File:** `plugin/legacy/AI_CoPilot.py`
- **Symptom:** Plugin loads `ai_copilot.pkl` from `ML/`, but no such file is in the repo.
- **Impact:** Legacy plugin cannot run as-is.
- **Fix:** Either ship the pickle, regenerate it from a documented script, or mark file as deprecated and remove. Recommend the last option — the active plugin is `plugin/PI_VimaanCoPilot.py`.

## 🔴 B-002 — HuggingFace token committed to the working tree
- **File:** `scratch/token_HF.txt` (was at repo root pre-reorg).
- **Impact:** Token must be considered **compromised**. Even though `scratch/` is now `.gitignore`d, the file may exist in working trees on other machines.
- **Fix:**
  1. **Revoke the token immediately** at <https://huggingface.co/settings/tokens>.
  2. Replace with an env var: `export HUGGINGFACE_HUB_TOKEN=...`.
  3. If the token was ever committed historically, scrub with `git filter-repo` and force-push.

## 🟠 B-003 — `ML/core/normalization.py` auto-pip-installs `word2number` at import time
- **File:** `ML/core/normalization.py`
- **Symptom:** On import, the module shells out to `pip install word2number` if missing.
- **Impact:** Fails inside the X-Plane plugin (no shell, no pip), and silently masks dependency declarations.
- **Fix:** Remove the auto-install block; declare `word2number` in `requirements.txt` (done) and raise a clean `ImportError` with install instructions.

## 🟠 B-004 — `train_nlu_model.py` may also auto-pip-install dependencies
- **File:** `ML/train_nlu_model.py`
- **Impact:** Same as B-003 — masks the dependency contract.
- **Fix:** Remove auto-install, document in README.

## 🟠 B-005 — Toggle "off" intents map to a toggle command, not an explicit off command
- **File:** `plugin/PI_VimaanCoPilot.py` (intent handlers)
- **Symptom:** `toggle_autopilot_1_off`, `toggle_autopilot_2_off`, similar "off" branches invoke `sim/autopilot/servos_toggle` (or analogous toggle commands) rather than an explicit "off" command/dataref write.
- **Impact:** If the autopilot is already off, "autopilot 1 off" turns it **on**. Same risk for FD, engines, parking brake.
- **Fix:** Read current state from the dataref first; only fire the toggle if state ≠ desired. Or use the dedicated "_off" command where X-Plane exposes one.

## 🟠 B-006 — Microphone callback can block indefinitely
- **File:** `plugin/PI_VimaanCoPilot.py` — `OnPressCallback`
- **Symptom:** `recognize_google()` is called synchronously after Z release with no timeout argument; if Google's STT hangs, the X-Plane main loop is blocked.
- **Fix:** Run STT on a worker thread; surface result via a flight-loop callback. Set `timeout=` and `phrase_time_limit=` on the recognizer.

## 🟠 B-007 — `sys.path` hack couples plugin to ML internals
- **File:** `plugin/PI_VimaanCoPilot.py`
- **Symptom:** Plugin does `sys.path.insert(0, "<repo>/ML")` and then `from core.model_loader import ModelLoader`. This pollutes the global module namespace and conflicts with any other plugin that also vendors a `core` package.
- **Fix:** Rename the package to `vimaan_nlu` (or similar) so it can be imported as `from vimaan_nlu.model_loader import ...`.

## 🟡 B-008 — `command_tester.py` expects intents that aren't in `intent_map.json`
- **File:** `ML/command_tester.py`
- **Symptom:** Tests reference `chit_chat_greeting`, `ask_time`, etc., which were never trained.
- **Impact:** Permanent "FAIL" rows in the report; obscures real regressions.
- **Fix:** Either remove those tests or actually add chit-chat intents to the dataset and model.

## 🟡 B-009 — `ML/core/__init__.py` imports submodules at package load
- **File:** `ML/core/__init__.py`
- **Symptom:** `__init__.py` does `from core.model_loader import ModelLoader` and `from core.inference import predict`. Importing the package triggers loading torch + transformers immediately, even if you only need normalization helpers.
- **Fix:** Make these imports lazy or expose only lightweight symbols at the top level.

## 🟡 B-010 — Duplicate normalization module
- **File:** `ML/core/normalization_backup.py` (removed in this reorg).
- **Action:** Already deleted. If you regenerate it, please don't.

## 🟡 B-011 — Tracked `.pyc` files in `ML/utils/__pycache__/`
- **Action:** Already untracked in this reorg via `git rm --cached`. `.gitignore` now blocks them.

## 🟡 B-012 — `models/` and `datasets/` directories are huge but were tracked
- **Action:** Both are now `.gitignore`d. The local working tree still has them (~2.3 GB) but they no longer pollute commits.

## ⚪ B-013 — `MISC/` directory contained ad-hoc notes
- **Action:** Renamed `MISC/git_cmds.txt` → `docs/git_workflow.md`; folder removed.

## ⚪ B-014 — Hard-coded macOS log path
- **File:** `plugin/PI_VimaanCoPilot.py`
- **Symptom:** Writes logs to `~/Desktop/Vimaan_Logs/`. Works everywhere `~` resolves, but `Desktop/` may not exist on headless or non-English locales.
- **Fix:** Use `Path.home() / "Vimaan_Logs"` or X-Plane's plugin directory.

## ⚪ B-015 — No tests
- No unit tests, no CI. `command_tester.py` is the closest thing to a regression suite.
- **Fix:** Add `pytest`, GitHub Actions workflow, lint + import smoke tests.

---

## Quick triage table

| ID | Severity | Area | Owner | Status |
| --- | --- | --- | --- | --- |
| B-001 | 🔴 | Legacy plugin | — | **done** (deprecation banner added) |
| B-002 | 🔴 | Security | — | **needs revoke** (user action) |
| B-003 | 🟠 | ML core | — | **done** |
| B-004 | 🟠 | Training | — | **done** (unused import removed) |
| B-005 | 🟠 | Plugin handlers | — | **done** (state-aware toggles) |
| B-006 | 🟠 | Plugin audio | — | **done** (threaded worker + queue) |
| B-007 | 🟠 | Packaging | — | deferred (rename to `vimaan_nlu` next pass) |
| B-008 | 🟡 | Tests | — | **done** |
| B-009 | 🟡 | ML core | — | **done** |
| B-010 | 🟡 | Hygiene | — | **done** |
| B-011 | 🟡 | Hygiene | — | **done** |
| B-012 | 🟡 | Repo size | — | **done** |
| B-013 | ⚪ | Hygiene | — | **done** |
| B-014 | ⚪ | Plugin | — | **done** |
| B-015 | ⚪ | Tooling | — | deferred (separate CI pass) |
