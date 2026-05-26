# Production Readiness Checklist

Use this checklist before release or handoff.

## Version Control and Release Hygiene

- [ ] Remote setup verified (`origin` is source of truth).
- [ ] Base branch (`main`) synced from origin.
- [ ] No unreviewed local-only commits on release branch.
- [ ] `CHANGELOG.md` updated with release notes.
- [ ] Release tag planned (`vMAJOR.MINOR.PATCH`).

## Documentation

- [ ] `README.md` reflects current architecture and setup.
- [ ] `FEATURES.md` reflects shipped capabilities.
- [ ] `CONTRIBUTING.md` reflects active workflow.
- [ ] `docs/PROJECT_STRUCTURE.md` reflects actual tree.
- [ ] Operational caveats and limitations documented.

## ML Pipeline Quality

- [ ] Schema validates successfully.
- [ ] Dataset generation run completes without errors.
- [ ] Validation report indicates zero invalid records.
- [ ] Generated outputs are versioned and reproducible.
- [ ] Metadata snapshot generated and archived.

## Plugin Quality

- [ ] Plugin loads in target simulator environment.
- [ ] Microphone detection works on target hardware.
- [ ] Speech transcription tested with expected phrases.
- [ ] Command execution mapped and validated safely.
- [ ] Failure paths provide clear logs/feedback.

## Security and Operations

- [ ] No credentials or secrets are committed.
- [ ] Dependency versions reviewed and pinned where practical.
- [ ] Large artifacts policy defined (.gitignore or release assets).
- [ ] Rollback strategy documented for releases.

## CI/CD (Recommended)

- [ ] Add lint and static checks.
- [ ] Add unit/integration tests.
- [ ] Add docs validation.
- [ ] Add release automation for tags and changelog enforcement.
