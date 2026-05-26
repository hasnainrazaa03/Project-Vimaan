# Contributing

## Branch Strategy

Current upstream default branch is `dev`.

Use this branch model:

1. Sync base branch (`dev`) from `upstream`.
2. Create feature branch from updated `dev`.
3. Commit logically grouped changes.
4. Push to `origin` and open PR.
5. Merge after review and validation.

## Remote Roles

- `origin`: personal/fork repository (`hasnainrazaa03/Project-Vimaan`).
- `upstream`: source repository (`The-Aryan/PROJECT-VIMAAN`).

## Local Sync Workflow

```bash
git checkout dev
git fetch upstream
git merge --ff-only upstream/dev
git push origin dev
```

## Feature Workflow

```bash
git checkout dev
git pull --ff-only origin dev
git checkout -b feature/short-description

# work
git add .
git commit -m "feat: short description"
git push -u origin feature/short-description
```

## Commit Style

Use clear, imperative commit messages. Suggested prefixes:

- `feat:` new functionality
- `fix:` bug fix
- `docs:` documentation updates
- `refactor:` internal improvements without behavior changes
- `chore:` maintenance changes
- `test:` test additions/updates

## Pull Requests

PRs should include:

- summary of changes
- reason/problem statement
- test/validation notes
- rollback considerations (if impactful)

## Data and Model Artifacts

- Avoid committing large generated artifacts unless intentionally versioning outputs.
- Keep generated dataset versioning consistent (`_vN` pattern).
- Document dataset generation parameters in PR description.

## Documentation Requirements

For any user-visible or workflow-impacting change, update at least one of:

- `README.md`
- `CHANGELOG.md`
- `FEATURES.md`
- relevant file in `docs/`
