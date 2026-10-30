"""Compare speech recognisers on real recorded commands, end to end.

The plugin saves each command's audio to ``~/Vimaan_Logs/audio/``. A labels
file says what was actually said (one JSON object per line)::

    {"file": "20261030_002528_c0007.wav", "text": "heading two seventy"}

Each backend transcribes every labelled clip, the transcript goes through the
same normalizer + model as in the sim, and the result counts as correct when
its intent and slots match those predicted from the label text. That is the
number that matters: a transcript can be "wrong" ("the com one") and still
produce the right command, or nearly right ("heading do seventy") and the
wrong one.

Usage::

    python scripts/stt_benchmark.py                       # vosk vs google
    python scripts/stt_benchmark.py --backends vosk,vosk-open
    python scripts/stt_benchmark.py --seed-labels         # add Google's transcripts
                                                          # for unlabelled clips (review them!)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ML"))

AUDIO_DIR = Path.home() / "Vimaan_Logs" / "audio"
BACKENDS = ("vosk", "vosk-open", "google")
MIN_CONFIDENCE = 0.7  # the plugin's execution threshold


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["file"]] = row["text"]
    return labels


def make_transcriber(name: str):
    """Return ``transcribe(pcm) -> str`` for a backend (raises on no speech)."""
    if name in ("vosk", "vosk-open"):
        from vimaan_nlu.stt import DEFAULT_VOSK_MODEL, VoskStream
        from vimaan_nlu.stt_vocabulary import grammar

        stream = VoskStream(DEFAULT_VOSK_MODEL, words=grammar() if name == "vosk" else None)

        def vosk(pcm: bytes) -> str:
            stream.begin()
            for i in range(0, len(pcm), 640):  # 20 ms chunks, like the live mic
                stream.feed(pcm[i : i + 640])
            return stream.finish()

        return vosk
    if name == "google":
        import speech_recognition as sr
        from vimaan_nlu.stt import transcribe

        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 8

        def google(pcm: bytes) -> str:
            return transcribe(recognizer, sr.AudioData(pcm, 16000, 2), "google")

        return google
    raise SystemExit(f"unknown backend {name!r}; choose from {BACKENDS}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument(
        "--labels", type=Path, default=None, help="default: <audio-dir>/labels.jsonl"
    )
    parser.add_argument("--backends", default="vosk,google")
    parser.add_argument("--seed-labels", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="show every clip")
    args = parser.parse_args(argv)
    labels_path = args.labels or args.audio_dir / "labels.jsonl"
    labels = load_labels(labels_path)
    clips = sorted(args.audio_dir.glob("*.wav"))

    if args.seed_labels:
        google = make_transcriber("google")
        added = 0
        with open(labels_path, "a", encoding="utf-8") as fh:
            for clip in clips:
                if clip.name in labels:
                    continue
                try:
                    text = google(read_pcm(clip))
                except Exception:
                    continue  # nothing intelligible: label by hand if it matters
                fh.write(json.dumps({"file": clip.name, "text": text}) + "\n")
                added += 1
        print(f"added {added} labels from Google to {labels_path} — review them before trusting")
        return 0

    labelled = [c for c in clips if c.name in labels]
    if not labelled:
        print(f"no labelled clips in {args.audio_dir} (see --seed-labels)", file=sys.stderr)
        return 1

    import torch
    from vimaan_nlu.inference import predict
    from vimaan_nlu.model_loader import ModelLoader

    loader = ModelLoader(torch.device("cpu"))
    loader.load_all(quantize=True)

    def nlu(text: str) -> tuple[str, dict]:
        p = predict(
            text,
            loader.model,
            loader.tokenizer,
            loader.device,
            loader.intent_map_rev,
            loader.slot_map_rev,
        )
        if p["confidence"] < MIN_CONFIDENCE:
            return "low_confidence", {}
        return p["intent"], p["slots"]

    names = [b.strip() for b in args.backends.split(",") if b.strip()]
    transcribers = {name: make_transcriber(name) for name in names}
    correct = dict.fromkeys(names, 0)
    times: dict[str, list[float]] = {n: [] for n in names}
    misses: dict[str, list[str]] = {n: [] for n in names}
    for clip in labelled:
        expected = nlu(labels[clip.name])
        pcm = read_pcm(clip)
        for name, transcribe in transcribers.items():
            t = time.perf_counter()
            try:
                text = transcribe(pcm)
            except Exception as exc:
                text = f"<{type(exc).__name__}>"
            times[name].append(time.perf_counter() - t)
            got = nlu(text) if not text.startswith("<") else ("-", {})
            ok = got == expected
            correct[name] += ok
            line = f"{clip.name}: said {labels[clip.name]!r} -> heard {text!r} -> {got[0]} {got[1]}"
            if not ok:
                misses[name].append(line)
            elif args.verbose:
                print(f"  ok  [{name}] {line}")

    n = len(labelled)
    print(f"\n{n} labelled clips (correct = same intent + slots as the label text)\n")
    print(f"{'backend':<12}{'correct':>12}{'p50 ms':>10}")
    for name in names:
        ts = sorted(times[name])
        print(f"{name:<12}{correct[name]:>6}/{n:<5}{1000 * ts[len(ts) // 2]:>10.0f}")
    for name in names:
        if misses[name]:
            print(f"\n--- {name} misses")
            print("\n".join(misses[name]))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.exit(main())
