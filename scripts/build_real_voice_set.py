"""Freeze the real-voice test set: tests/fixtures/real_voice.jsonl.

For every hand-labelled recording in ``~/Vimaan_Logs/audio/labels.jsonl``
(what the pilot actually said), store what the offline recogniser hears with
the current grammar. The expected command (``intent``, ``slots``) for each
clip is reviewed by hand and kept when the file is rebuilt. ``scripts/model_gate.py`` and ``tests/test_real_voice.py``
then judge each model on real speech without needing the audio.

Re-run after labelling new flights or changing the grammar:

    python scripts/build_real_voice_set.py
"""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ML"))

AUDIO = Path.home() / "Vimaan_Logs" / "audio"
OUT = ROOT / "tests" / "fixtures" / "real_voice.jsonl"


def main() -> int:
    from vimaan_nlu.stt import DEFAULT_VOSK_MODEL, UnknownValueError, VoskStream
    from vimaan_nlu.stt_vocabulary import grammar

    stream = VoskStream(DEFAULT_VOSK_MODEL, words=grammar())
    labels = [json.loads(line) for line in open(AUDIO / "labels.jsonl", encoding="utf-8")]
    expected = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
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
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} clips to {OUT}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
