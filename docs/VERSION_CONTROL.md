# Version Control Guide

## Remote Topology

Use this setup:

- `origin`: your repository (push target for day-to-day work)
- `upstream`: source repository (read/sync source)

Current intended values:

- `origin`: `https://github.com/hasnainrazaa03/Project-Vimaan.git`
- `upstream`: `https://github.com/The-Aryan/PROJECT-VIMAAN.git`

## One-Time Setup

If repository is cloned from upstream directly:

```bash
git remote rename origin upstream
git remote add origin https://github.com/hasnainrazaa03/Project-Vimaan.git
```

If old `origin` must be removed first:

```bash
git remote remove origin
git remote add origin https://github.com/hasnainrazaa03/Project-Vimaan.git
git remote add upstream https://github.com/The-Aryan/PROJECT-VIMAAN.git
```

## Verify Remotes

```bash
git remote -v
```

Expected pattern:

- origin fetch/push -> personal repository
- upstream fetch/push -> source repository

## Sync from Upstream

For repositories whose base branch is `dev`:

```bash
git checkout dev
git fetch upstream
git merge --ff-only upstream/dev
git push origin dev
```

For repositories using `main`:

```bash
git checkout main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

For repositories using `master`:

```bash
git checkout master
git fetch upstream
git merge --ff-only upstream/master
git push origin master
```

## Safe Reset Procedure (When Replacing Local State)

Use only when intentionally discarding local working tree history/state.

```bash
git fetch upstream
git checkout <base-branch>
git reset --hard upstream/<base-branch>
```

Before this operation, always create backup snapshot of local files and ensure no uncommitted work is needed.
