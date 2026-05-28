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
