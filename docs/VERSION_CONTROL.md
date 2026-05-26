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
