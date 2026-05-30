#!/usr/bin/env bash
# bootstrap_github_issues.sh
# ---------------------------------------------------------------------------
# Phase 3 (B-015 / workflow hardening): one-shot bootstrap for the GitHub
# Issues workflow.
#
# What this script does:
#   1. Creates the standard label set (area:*, type:*, priority:*).
#   2. Opens an issue for every still-outstanding item from the Phase 0
#      audit (currently just B-002).
#   3. Prints the list so the user can spot-check.
#
# Prereqs:
#   - `brew install gh`  (or equivalent)
#   - `gh auth login`     (web flow, scope: repo)
#   - run from repo root.
#
# Safe to re-run: label creation uses `--force`, and issue creation is
# guarded by a search to avoid duplicates.
# ---------------------------------------------------------------------------
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) not found. Install with: brew install gh" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh not authenticated. Run: gh auth login" >&2
  exit 1
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "Bootstrapping issues on: $REPO"
echo

# ---------- labels ----------
declare -a LABELS=(
  # area
  "area:plugin|#0e8a16|Code under plugin/"
  "area:nlu|#1d76db|ML model + vimaan_nlu package"
  "area:training-data|#5319e7|Dataset generation & augmentation"
  "area:docs|#c5def5|Documentation & planning"
  "area:infra|#fbca04|CI, pre-commit, packaging"
  # type
  "type:bug|#d73a4a|Something is broken"
  "type:enhancement|#a2eeef|New feature or improvement"
  "type:tech-debt|#cfd3d7|Refactor / cleanup, no behaviour change"
  "type:security|#b60205|Credential, dependency, or attack-surface issue"
  # priority
  "priority:p0|#b60205|Drop everything — fix now"
  "priority:p1|#d93f0b|Next sprint"
  "priority:p2|#fbca04|When convenient"
  "priority:p3|#0e8a16|Nice to have"
)

echo "==> Creating / updating labels"
for entry in "${LABELS[@]}"; do
  IFS='|' read -r name color desc <<<"$entry"
  if gh label create "$name" --color "${color#\#}" --description "$desc" --force >/dev/null 2>&1; then
    echo "  ok:  $name"
  else
    echo "  WARN: could not upsert label $name" >&2
  fi
done
echo

# ---------- issues ----------
# Anything still outstanding from the Phase 0 audit.
# Format: TITLE|LABELS (comma-separated)|BODY (multiline ok via \n)
create_issue () {
  local title="$1" labels="$2" body="$3"

  # Skip if an issue with the same title already exists (open OR closed).
  local existing
  existing="$(gh issue list --search "in:title \"$title\"" --state all --limit 1 --json number -q '.[0].number' || true)"
  if [[ -n "$existing" ]]; then
    echo "  skip (exists as #$existing): $title"
    return 0
  fi

  local num
  num="$(printf '%b' "$body" | gh issue create --title "$title" --label "$labels" --body-file - --json number -q .number)"
  echo "  created #$num: $title"
}

echo "==> Opening outstanding Phase 0 issues"

create_issue \
  "[B-002] Revoke compromised HuggingFace token" \
  "area:infra,type:security,priority:p0" \
  "A HuggingFace token was committed pre-reorg under \`scratch/token_HF.txt\`. The file is now gitignored, but **the token must be treated as compromised**.\n\n**User action required:**\n1. Open https://huggingface.co/settings/tokens\n2. Revoke the leaked token.\n3. (Optional) Issue a new token and export as env var: \`export HUGGINGFACE_HUB_TOKEN=...\`.\n4. If the token ever landed in a commit, scrub history with \`git filter-repo\` and force-push.\n\nHistorical context: docs/BUGS.md (commit 444fd60)."

echo
echo "Done. Visit: https://github.com/$REPO/issues"
