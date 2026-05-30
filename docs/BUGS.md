# Known Bugs — Historical Changelog

> This file is the **historical record** of the Phase 0 audit. As of
> Phase 3, all open work and new findings live on the **GitHub Issues**
> tracker:
>
> https://github.com/hasnainrazaa03/Project-Vimaan/issues
>
> Do **not** add new bugs here. Open an issue with the appropriate
> labels instead (`area:plugin` / `area:nlu` / `area:training-data`;
> `type:bug` / `type:enhancement`; `priority:p0..p3`).

Severity legend: 🔴 critical · 🟠 high · 🟡 medium · ⚪ minor.

---

## Phase 0 audit — final status

| ID | Severity | Area | Status | Commit / Notes |
| --- | --- | --- | --- | --- |
| B-001 | 🔴 | Legacy plugin | **done** | `444fd60` — deprecation banner; active plugin is `plugin/PI_VimaanCoPilot.py` |
| B-002 | 🔴 | Security | **needs revoke** (user action) | scratch/ now gitignored; token must be revoked at huggingface.co |
| B-003 | 🟠 | ML core | **done** | `444fd60` — auto-pip block removed; clean ImportError |
| B-004 | 🟠 | Training | **done** | `444fd60` — unused num2words import removed |
| B-005 | 🟠 | Plugin handlers | **done** | `444fd60` — state-aware toggles; explicit on/off |
| B-006 | 🟠 | Plugin audio | **done** | `444fd60` — STT on worker thread + queue; bounded timeout |
| B-007 | 🟠 | Packaging | **done** | `bc89b28` — `core/` → `vimaan_nlu/`; all call sites updated |
| B-008 | 🟡 | Tests | **done** | `444fd60` — out-of-scope chit-chat tests pruned |
| B-009 | 🟡 | ML core | **done** | `dca83e7` — `__init__.py` now lazy via PEP 562 `__getattr__` |
| B-010 | 🟡 | Hygiene | **done** | `bc89b28` — `normalization_backup.py` removed |
| B-011 | 🟡 | Hygiene | **done** | `bc89b28` — all `__pycache__/` untracked |
| B-012 | 🟡 | Repo size | **done** | `bc89b28` — `models/` and `datasets/` gitignored |
| B-013 | ⚪ | Hygiene | **done** | `bc89b28` — `MISC/` folder dissolved into `docs/` |
| B-014 | ⚪ | Plugin | **done** | `444fd60` — log path uses `Path.home() / Vimaan_Logs` |
| B-015 | ⚪ | Tooling | **done** | `dca83e7` — pytest + ruff + pre-commit + GitHub Actions CI |

**Outstanding from Phase 0:** only B-002 (user action: revoke the
compromised HuggingFace token). Everything else is shipped.

---

## Detailed write-ups

The original per-bug write-ups (symptom, impact, fix) are preserved in
git history. To read them at any time:

```bash
git show 444fd60:docs/BUGS.md | less
```

Or browse on GitHub:
https://github.com/hasnainrazaa03/Project-Vimaan/blob/444fd60/docs/BUGS.md
