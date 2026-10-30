"""Download the offline speech model (Vosk small English, ~40 MB) into
ML/models/vosk/, verifying its SHA-256. scripts/install_plugin.py then ships
it to X-Plane.

Usage::

    python scripts/fetch_vosk_model.py
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
SHA256 = (
    "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"  # pragma: allowlist secret
)
DEST = os.path.join(ROOT, "ML", "models", "vosk")


def main() -> int:
    if os.path.isdir(os.path.join(DEST, NAME)):
        print(f"already present: {os.path.join(DEST, NAME)}")
        return 0
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"ERROR: checksum mismatch ({digest}), not installing", file=sys.stderr)
        return 1
    os.makedirs(DEST, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():  # refuse paths that escape DEST
            target = os.path.realpath(os.path.join(DEST, member))
            if not target.startswith(os.path.realpath(DEST) + os.sep):
                print(f"ERROR: unsafe path in archive: {member}", file=sys.stderr)
                return 1
        zf.extractall(DEST)
    print(f"installed {os.path.join(DEST, NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
