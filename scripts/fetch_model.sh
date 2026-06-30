#!/usr/bin/env bash
# fetch_model.sh
# ---------------------------------------------------------------------------
# Download a model checkpoint from GitHub Releases into
# ML/models/vimaan_nlu_model_best/. Weights are NOT stored in the git repo;
# see docs/MODEL_DISTRIBUTION.md.
#
# Usage:  scripts/fetch_model.sh [version]    default: latest model-* release
#         e.g. scripts/fetch_model.sh v10
# Requires: gh (authenticated), tar.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/ML/models/vimaan_nlu_model_best"
mkdir -p "$DEST"

VERSION="${1:-}"
if [ -n "$VERSION" ]; then
  TAG="model-$VERSION"
else
  TAG="$(gh release list --limit 100 | awk '$0 ~ /model-/ {for (i=1;i<=NF;i++) if ($i ~ /^model-/) {print $i; exit}}')"
  [ -n "$TAG" ] || { echo "error: no model-* release found" >&2; exit 1; }
  echo "Latest model release: $TAG"
fi

TMP="$(mktemp -d)"
echo "Downloading $TAG ..."
gh release download "$TAG" --dir "$TMP" --pattern '*.tar.gz'
ARCHIVE="$(ls "$TMP"/*.tar.gz | head -1)"
echo "Extracting $(basename "$ARCHIVE") -> $DEST"
tar -xzf "$ARCHIVE" -C "$DEST"
echo "Done. Installed $TAG into $DEST"
