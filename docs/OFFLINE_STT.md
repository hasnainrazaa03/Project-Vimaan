# Speech recognition

Vimaan recognises speech **offline by default**, with
[Vosk](https://alphacephei.com/vosk/) (Kaldi). No audio leaves the machine and
no network is needed. Google Web Speech (online) is still available.

```bash
export VIMAAN_STT_BACKEND=vosk      # default
export VIMAAN_STT_BACKEND=google    # online
```

Set the variable before launching X-Plane. The worker reports which backend it
actually uses in the `model_load` event; if Vosk can't load (package or model
missing) it falls back to Google and logs a `stt_fallback` warning with the
reason.

## How it works

- **Streaming.** Audio is fed to the recogniser while you hold the key, so on
  release only the final bit is decoded: 2–5 ms, versus ~460 ms (p50) for a
  Google round trip in-sim.
- **Grammar.** Vosk may only output the ~145 words in
  [`ML/vimaan_nlu/command_words.txt`](../ML/vimaan_nlu/command_words.txt)
  plus `[unk]`. That file is generated from the training schema by
  `python scripts/build_stt_vocabulary.py`; a test fails if it is stale.
  Out-of-grammar speech becomes `[unk]` (dropped), so chatter doesn't turn
  into commands.

Measured on 108 synthetic clips (6 macOS voices × 18 commands), right intent
and slots end to end, through the v11 model:

| Recogniser | Correct | Notes |
| --- | --- | --- |
| Vosk, full vocabulary | 62 / 108 | "gear" → "you're", "set" → "that" |
| **Vosk, command grammar + `[unk]`** | **94 / 108** | default |
| Vosk, command grammar, no `[unk]` | 99 / 108 | but chatter becomes commands ("let me check the chart" → "retract start" = start engine 1) |
| Google | 49 / 108 | returns nothing on many synthetic clips; real voices are the fair test (see below) |

## Setup

```bash
python scripts/fetch_vosk_model.py      # ~40 MB, SHA-256 checked -> ML/models/vosk/
python scripts/install_plugin.py --xplane "<X-Plane 12>"   # ships the model too
```

and `vosk` in XPPython3's Python (see [IN_SIM_TESTING.md](IN_SIM_TESTING.md)
§2 for installing packages there on Apple Silicon).

## Saved audio

Each command's audio is saved to `~/Vimaan_Logs/audio/<session>_<cmd>.wav`
(16 kHz mono, ~32 KB per second, stops at 500 MB). The file name joins it to
that session's `events.jsonl`, which holds the transcript and outcome. These
recordings are how recognisers are compared on real voices (Track S3 in
[PHASE_PLAN.md](PHASE_PLAN.md)). Turn off with `VIMAAN_SAVE_AUDIO=0`.

## Other backends

| Backend | Offline | Install | Notes |
| --- | --- | --- | --- |
| `vosk` | ✅ | `vosk` + model | Default. Streaming, grammar-restricted. |
| `google` | ❌ | (included) | Free tier, rate-limited, needs internet. |
| `whisper` | ✅ | `pip install openai-whisper` | Starts after key-up; 0.3–1 s on CPU; can invent text on silence. `VIMAAN_WHISPER_MODEL` (default `base.en`). |
| `sphinx` | ✅ | `pip install pocketsphinx` | Lowest accuracy. |

Words the small Vosk model doesn't know (warned in the worker log): compounds
such as *autothrottle*, *speedbrake*, *autobrake*. New commands must use the
split forms ("auto throttle") in the grammar and have the normalizer rejoin
them.
