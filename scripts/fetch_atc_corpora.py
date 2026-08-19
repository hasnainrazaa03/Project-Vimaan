"""Fetch transcript text from open ATC speech corpora into a phrasing bank.

These corpora are audio+transcript parquet on the HF Hub; to Vimaan they are
useful only for ASR-realism (how numbers/frequencies/headings are actually
spoken), NOT for intents/slots — see docs/DATA_SOURCES.md. This script streams
each row and keeps ONLY the transcript text (audio is never written to disk),
dedups, and writes a plain-text phrasing bank you can mine when upgrading data
generation.

Requires ``datasets`` (``pip install datasets``). Respect licenses: UWB-ATCC and
the jacktol/Tabys/jlvdoorn merges are CC BY-NC-SA (non-commercial); ATCO2 sets
are EULA-gated. Use these to *inform* generation, not as shipped training labels.

Examples::

    python scripts/fetch_atc_corpora.py --dataset Jzuluaga/atcosim_corpus --limit 5000
    python scripts/fetch_atc_corpora.py --dataset Jzuluaga/uwb_atcc --split test --limit 3000
"""

from __future__ import annotations

import argparse
import os
import re
import sys

TEXT_COLS = ["text", "transcription", "transcript", "sentence", "segment_text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. Jzuluaga/atcosim_corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=5000, help="max unique transcripts to keep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("This script needs `datasets`. Install it with:  pip install datasets")

    print(f"Streaming {args.dataset} [{args.split}] — transcripts only, audio discarded...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # Keep only a text column so the audio bytes aren't materialised.
    cols = list(getattr(ds, "column_names", None) or [])
    text_col = next((c for c in TEXT_COLS if c in cols), None)
    if text_col:
        try:
            ds = ds.select_columns([text_col])
        except Exception:
            pass

    seen: set[str] = set()
    lines: list[str] = []
    for row in ds:
        if len(lines) >= args.limit:
            break
        raw = (
            row.get(text_col)
            if text_col
            else next((row.get(c) for c in TEXT_COLS if row.get(c)), None)
        )
        if not raw:
            continue
        t = re.sub(r"\s+", " ", str(raw)).strip().lower()
        if len(t) < 3 or t in seen:
            continue
        seen.add(t)
        lines.append(t)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "ML",
        "datasets",
        "external",
        args.dataset.replace("/", "__") + ".txt",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} unique transcripts -> {out}")
    if lines:
        print("Sample:")
        for line in lines[:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
