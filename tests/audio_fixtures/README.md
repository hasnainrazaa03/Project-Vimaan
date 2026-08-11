# Gold audio fixtures (Phase 6E)

Recorded voice samples for the end-to-end audio regression test
([`tests/test_gold_audio.py`](../test_gold_audio.py)). The wavs and the real
manifest are **gitignored** (only this README and `manifest.example.jsonl` are
committed) — audio is bulky and personal.

## How it works

`test_gold_audio.py` reads `manifest.jsonl`, and for each entry: transcribes the
wav with **offline Whisper** (deterministic) → runs the model → asserts the
predicted `intent` (and any listed `slots`). It skips cleanly when there's no
manifest, no wavs, or whisper/torch aren't installed.

## Recording

1. Record clear, ~1–3 s mono wavs, 16 kHz, one command each. Aim for ~5 samples
   per intent (~70 total) across the commands in the README's *Supported
   Commands* table. Vary phrasing and pace.
2. Save them here, e.g. `gear_up_01.wav`, `heading_270_02.wav`.
3. Create `manifest.jsonl` (one JSON object per line) — see
   [`manifest.example.jsonl`](manifest.example.jsonl):

   ```json
   {"wav": "gear_up_01.wav", "intent": "toggle_landing_gear", "slots": {"state": "up"}}
   {"wav": "heading_270_01.wav", "intent": "set_autopilot_heading", "slots": {"degrees": "270"}}
   ```

4. Install the offline STT: `pip install openai-whisper` (see
   [docs/OFFLINE_STT.md](../../docs/OFFLINE_STT.md)).
5. Run: `pytest tests/test_gold_audio.py -v`

`slots` is optional per entry; omit it to assert intent only. Keep known-hard
samples in a separate manifest if you want a tolerance tier.
