# Offline Speech-to-Text

By default the plugin transcribes with **Google Web Speech** (online). You can
switch to a fully **offline** backend so the copilot works with no network and
no audio leaving the machine. Selection is by environment variable — no code
change:

```bash
export VIMAAN_STT_BACKEND=vosk       # or: whisper | sphinx | google (default)
```

The dispatch lives in [`ML/vimaan_nlu/stt.py`](../ML/vimaan_nlu/stt.py)
(`transcribe()`), so all backends share the same capture path and the plugin's
intent pipeline is unchanged.

## Backends

| Backend | Offline | Install | Notes |
| --- | --- | --- | --- |
| `google` | ❌ | (default) | Free tier, rate-limited, needs internet. |
| `vosk` | ✅ | `pip install vosk` + a model | Lightweight, real-time friendly. **Recommended offline choice.** |
| `whisper` | ✅ | `pip install openai-whisper` | Most accurate; heavier, slower on CPU. Model via `VIMAAN_WHISPER_MODEL` (default `base.en`). |
| `sphinx` | ✅ | `pip install pocketsphinx` | Lowest resource, lowest accuracy. |

## Vosk setup (recommended)

```bash
pip install vosk
# Download a small English model and point SpeechRecognition at it:
#   https://alphacephei.com/vosk/models  (e.g. vosk-model-small-en-us-0.15)
```

`SpeechRecognition`'s `recognize_vosk` looks for a `model/` directory in the
working directory by default; place or symlink the unpacked model there, or set
it up per the SpeechRecognition docs. `transcribe()` parses vosk's JSON output
back to a plain string automatically.

## Whisper setup

```bash
pip install openai-whisper
export VIMAAN_STT_BACKEND=whisper
export VIMAAN_WHISPER_MODEL=base.en   # tiny.en / base.en / small.en ...
```

## Notes

- Offline packages are **not** in `requirements.txt` (they're heavy and
  model-dependent) — install only the backend you use.
- On plugin start the log records the active backend:
  `[Vimaan] STT backend: vosk`.
- End-to-end accuracy of each backend in-sim is still to be measured
  (tracked with the gold-audio set, Phase 6E).
