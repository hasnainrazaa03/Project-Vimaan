# Version Control Guide

## Remote Topology

Use this setup:

- `origin`: your repository (single source of truth)

Current intended values:

- `origin`: `https://github.com/hasnainrazaa03/Project-Vimaan.git`

## One-Time Setup

If `origin` is missing or incorrect:

```bash
git remote add origin https://github.com/hasnainrazaa03/Project-Vimaan.git
```

If old `origin` must be removed first:

```bash
git remote remove origin
git remote add origin https://github.com/hasnainrazaa03/Project-Vimaan.git
```

## Verify Remotes

```bash
git remote -v
```

Expected pattern:

- origin fetch/push -> source-of-truth repository

## Sync Main

```bash
git checkout main
git pull --ff-only origin main
git push origin main
```

## Safe Reset Procedure (When Replacing Local State)

Use only when intentionally discarding local working tree history/state.

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

Before this operation, always create backup snapshot of local files and ensure no uncommitted work is needed.

---

## Quick command reference

> The everyday workflow distilled into copy-pasteable blocks. For the
> rationale and branch strategy, see [CONTRIBUTING.md](../CONTRIBUTING.md).

### Refresh main

```bash
git checkout main
git pull origin main
```

### Start a feature branch

```bash
git checkout -b feature/branch-name
```

### Commit and push the feature branch

```bash
git add .
git commit -m "describe your changes"
git push -u origin feature/branch-name
```

### Merge feature into main

```bash
git checkout main
git pull origin main
git merge feature/branch-name
git push origin main
```

### Clean up the feature branch

```bash
git branch -d feature/branch-name           # local
git push origin --delete feature/branch-name # remote
git fetch -p                                 # prune stale refs
```

### Inspect history

```bash
git log --oneline --graph --decorate -n 20
git var GIT_AUTHOR_IDENT     # confirm commit identity
```

---

## Release tagging & model artifacts

Trained model directories live under `ML/models/` and are **gitignored**
(typically ~150 MB each). They are released as **GitHub Release assets**
rather than committed.

### Naming convention

- Code releases: `v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v0.2.0`).
- Model artifacts: `model-v<MAJOR>.<MINOR>.<PATCH>` (e.g.
  `model-v0.1.0`), independent of code version. A model release MUST
  ship its `train_manifest.json` alongside the weights so the consumer
  can reproduce or audit the run.

### Cutting a model release

```bash
# 1. (Optional) Backfill the manifest if it was trained pre-Phase-3.
python3 ML/data/generate_manifest.py \
  --model-dir ML/models/vimaan_nlu_model_best/v9 \
  --dataset   ML/datasets/05_final_merged/aviation_cmds_final_training_set.jsonl

# 2. Bundle the model directory.
tar -C ML/models/vimaan_nlu_model_best -czf model-v0.1.0.tar.gz v9

# 3. Tag the code commit the model was trained on.
git tag -a model-v0.1.0 -m "DistilBERT baseline (v9 checkpoint)"
git push origin model-v0.1.0

# 4. Publish as a GitHub Release with the tarball attached.
gh release create model-v0.1.0 model-v0.1.0.tar.gz \
  --title "Model v0.1.0 — DistilBERT baseline" \
  --notes-file <(cat <<'NOTES'
First tagged model artifact.
Base: distilbert-base-uncased, max_length=64, 10 epochs, AdamW lr=5e-5.
Dataset: see attached train_manifest.json (committed inside the tarball).
NOTES
)
```

### Consuming a model release

```bash
gh release download model-v0.1.0 --pattern '*.tar.gz'
mkdir -p ML/models/vimaan_nlu_model_best
tar -C ML/models/vimaan_nlu_model_best -xzf model-v0.1.0.tar.gz
```

The plugin / `ML/predict.py` will pick up the directory automatically via
`find_latest_version_path`.

---

## Backup & disaster recovery

The repo's source of truth is **GitHub** (`origin/main`). Everything
else is reconstructible.

### What is backed up where

| Asset | Backup location | Recreate by |
| --- | --- | --- |
| Source code | `origin/main` on GitHub | `git clone` |
| Trained models (`ML/models/`) | GitHub Releases (one per `model-v*` tag) | `gh release download` (see above) |
| Datasets (`ML/datasets/`) | Regeneratable from `ML/data/generate_*.py` | re-run generators with the same `git_sha` listed in the model's `train_manifest.json` |
| Secrets (HF token, etc.) | Local `.env`, never in git | re-issue via HuggingFace settings |
| `.secrets.baseline` | git-tracked | regenerated with `detect-secrets scan` |

### Recovery scenarios

1. **Laptop wiped, only GitHub left.**
   ```bash
   git clone https://github.com/hasnainrazaa03/Project-Vimaan.git
   cd Project-Vimaan
   python3 -m pip install -r requirements-dev.txt
   gh release download model-v0.1.0 --pattern '*.tar.gz'   # if a release exists
   tar -C ML/models/vimaan_nlu_model_best -xzf model-v0.1.0.tar.gz
   ```
   Datasets are not on GitHub by design (too large); regenerate them
   from the scripts under `ML/data/` if you need to retrain.

2. **GitHub repo deleted or compromised.**
   - Local clones (your laptop, any collaborator's) still contain full
     history — `git push` to a new remote restores `origin`.
   - Restore release assets from any laptop that pulled them.

3. **Local working tree corrupted (mid-merge, bad rebase, etc.).**
   - `git reflog` recovers any commit that was HEAD in the last 90 days
     (default reflog expiry).
   - For shared branches, the remote copy is authoritative — re-fetch
     and reset.

### Hygiene checklist (run quarterly)

```bash
# 1. Verify GitHub still has every model release you expect.
gh release list

# 2. Re-run secrets scan with the latest detect-secrets ruleset.
python3 -m detect_secrets scan --baseline .secrets.baseline

# 3. Confirm no large files snuck in.
git ls-files | xargs -I{} stat -f "%z %N" "{}" 2>/dev/null \
  | sort -n -r | head -20
```
