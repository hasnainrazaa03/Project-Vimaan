#!/usr/bin/env bash
# publish_model.sh
# ---------------------------------------------------------------------------
# Publish a trained model checkpoint as a GitHub Release asset.
#
# Weights live ONLY locally (ML/models/** is gitignored) and are distributed
# via GitHub Releases instead of Git LFS — see docs/MODEL_DISTRIBUTION.md.
#
# Usage:  scripts/publish_model.sh <version>     e.g. scripts/publish_model.sh v10
# Requires: gh (authenticated), tar.
set -euo pipefail

VERSION="${1:?usage: publish_model.sh <version, e.g. v10>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_PARENT="$REPO_ROOT/ML/models/vimaan_nlu_model_best"
MODEL_DIR="$MODELS_PARENT/$VERSION"

[ -d "$MODEL_DIR" ] || { echo "error: model dir not found: $MODEL_DIR" >&2; exit 1; }

TAG="model-$VERSION"
TMP="$(mktemp -d)"
ARCHIVE="$TMP/vimaan-nlu-$VERSION.tar.gz"

echo "Packing $MODEL_DIR"
tar -czf "$ARCHIVE" -C "$MODELS_PARENT" "$VERSION"
echo "Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "Release $TAG exists — uploading asset (clobber)."
  gh release upload "$TAG" "$ARCHIVE" --clobber
else
  echo "Creating release $TAG."
  gh release create "$TAG" "$ARCHIVE" \
    --title "Vimaan NLU model $VERSION" \
    --notes "Joint intent+slot DistilBERT checkpoint ($VERSION). Install with: scripts/fetch_model.sh $VERSION"
fi
echo "Done: $TAG"
