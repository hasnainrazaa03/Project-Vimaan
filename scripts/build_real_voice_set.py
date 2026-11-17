"""Freeze the real-voice test set: tests/fixtures/real_voice.jsonl.

For every hand-labelled recording in ``~/Vimaan_Logs/audio/labels.jsonl``
(what the pilot actually said), store what the offline recogniser hears with
the current grammar. The expected command (``intent``, ``slots``) for each
clip is reviewed by hand and kept when the file is rebuilt. ``scripts/model_gate.py`` and ``tests/test_real_voice.py``
then judge each model on real speech without needing the audio.

Re-run after labelling new flights or changing the grammar:

    python scripts/build_real_voice_set.py
    python scripts/build_real_voice_set.py --set heldout

``--set heldout``: the held-out set (``labels_heldout.jsonl`` ->
``tests/fixtures/real_voice_heldout.jsonl``), the owner's 2026-11-16 voice
check. The gate measures it and decides nothing with it, and nothing is
trained or tuned on it: it stays a fresh test of the owner's voice.

It also writes ``real_voice.meta.json``: the grammar's fingerprint and the Vosk
model. A test and the gate refuse a set heard with another grammar (it went
stale once: frozen at E1, then Tier 1 and 2 grew the grammar and 12 of 118
clips came out differently, mostly worse, while the gate kept grading the old
transcripts).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ML"))

AUDIO = Path.home() / "Vimaan_Logs" / "audio"
FIXTURES = ROOT / "tests" / "fixtures"
# set -> (labels in AUDIO, fixture name)
SETS = {
    "gate": ("labels.jsonl", "real_voice"),
    "heldout": ("labels_heldout.jsonl", "real_voice_heldout"),
}


def main(argv: list[str] | None = None) -> int:
    from vimaan_nlu.stt import DEFAULT_VOSK_MODEL, UnknownValueError, VoskStream
    from vimaan_nlu.stt_vocabulary import grammar, grammar_sha256

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", choices=sorted(SETS), default="gate")
    args = parser.parse_args(argv)
    labels_name, name = SETS[args.set]
    out = FIXTURES / f"{name}.jsonl"
    meta_path = FIXTURES / f"{name}.meta.json"

    stream = VoskStream(DEFAULT_VOSK_MODEL, words=grammar())
    labels = [json.loads(line) for line in open(AUDIO / labels_name, encoding="utf-8")]
    expected = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            expected[r["clip"]] = {"intent": r["intent"], "slots": r["slots"]}
    rows = []
    for row in labels:
        path = AUDIO / row["file"]
        if not path.exists():
            continue
        with wave.open(str(path)) as w:
            pcm = w.readframes(w.getnframes())
        stream.begin()
        for i in range(0, len(pcm), 640):
            stream.feed(pcm[i : i + 640])
        try:
            heard = stream.finish()
        except UnknownValueError:
            heard = ""
        if row["file"] not in expected:
            print(f"needs an expected command (add by hand): {row['file']} {row['text']!r}")
            continue
        rows.append(
            {"clip": row["file"], "said": row["text"], "heard": heard, **expected[row["file"]]}
        )
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = {
        "grammar_sha256": grammar_sha256(),
        "vosk_model": os.path.basename(DEFAULT_VOSK_MODEL),
        "clips": len(rows),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} clips to {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
