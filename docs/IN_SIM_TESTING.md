# In-sim testing runbook (X-Plane 12, macOS)

How to get Vimaan running inside X-Plane for the first time, and how to read
what happened afterwards. Every step leaves a trace in `~/Vimaan_Logs/`, so a
failure at any point can be diagnosed from the logs alone.

Verified environment: X-Plane **12.4.3** (Steam, Apple Silicon/Metal) on an
M-series Mac. X-Plane's `Info.plist` declares microphone access for plugins,
so macOS will ask for permission once on first use.

---

## 1. Install XPPython3 (once)

1. Download **XPPython3 4.7.x for X-Plane 12** from
   <https://xppython3.readthedocs.io/en/latest/usage/installation_plugin.html>.
2. Unzip it into `<X-Plane 12>/Resources/plugins/` so you have
   `Resources/plugins/XPPython3/`.
3. Remove the macOS quarantine flag, or XPPython3 will not load:
   ```bash
   xattr -dr com.apple.quarantine "<X-Plane 12>/Resources/plugins/XPPython3"
   ```
4. Start X-Plane once. The **Plugins** menu should now show **XPPython3**, and
   `Resources/plugins/PythonPlugins/` will exist.

XPPython3 bundles its own Python (3.12). Vimaan's packages must be installed
into **that** Python, not the project venv.

## 2. Install Vimaan's Python packages into XPPython3 (once)

In X-Plane: **Plugins → XPPython3 → Pip Package Installer**, then install:

```
torch transformers tokenizers safetensors numpy SpeechRecognition sounddevice soundfile vosk word2number
```

or, equivalently, from a terminal with XPPython3's own interpreter:

```bash
PY="<X-Plane 12>/Resources/plugins/XPPython3/mac_x64/python3.12/bin/python3.12"
"$PY" -m pip install torch transformers tokenizers safetensors numpy \
    SpeechRecognition sounddevice soundfile vosk word2number
```

> **Apple Silicon gotcha (XPPython3 4.7.x):** the standalone
> `python3.12` executable XPPython3 ships is **not code-signed for arm64**, so
> macOS kills it on launch (exit 137) and the command above fails. XPPython3
> itself is fine — inside X-Plane it runs Python from the signed
> `libpython3.12.dylib`. Install with any other arm64 Python 3.12 instead
> (e.g. `uv python install 3.12`), targeting XPPython3's `site-packages`:
>
> ```bash
> SP="<X-Plane 12>/Resources/plugins/XPPython3/mac_x64/python3.12/lib/python3.12/site-packages"
> python3.12 -m pip install --target /tmp/xpp3_site torch "transformers>=5.9,<6" \
>     "tokenizers>=0.21,<0.23" safetensors numpy SpeechRecognition sounddevice soundfile word2number==1.1
> rsync -a /tmp/xpp3_site/ "$SP/"      # do NOT exclude nested bin/ — torch/bin is required
> ```
>
> The bundled `charset_normalizer` 3.3.2 is also x86_64-only (breaks
> `import requests` on arm64); replace it with
> `python3.12 -m pip install --no-deps --target ... "charset_normalizer>=2,<4"`.

No PyAudio, Homebrew or compiler is needed: microphone capture uses
`sounddevice`, whose wheel bundles PortAudio.

If anything is missing, the plugin still loads, says *"Vimaan could not load
its Python packages"*, and logs a `dependency_missing` event naming exactly
which import failed. A missing `sounddevice` only disables the microphone
(`microphone_unavailable`); the self-test still works.

## 3. Install the plugin

```bash
X="$HOME/Library/Application Support/Steam/steamapps/common/X-Plane 12"
python scripts/check_xplane_refs.py --xplane "$X"      # every dataref/command exists
python scripts/fetch_vosk_model.py                     # offline speech model (once)
python scripts/install_plugin.py --xplane "$X" --dry-run
python scripts/install_plugin.py --xplane "$X"
```

This copies the plugin, the runtime `ML/` package, the latest model (v11) and
the offline speech model into `Resources/plugins/PythonPlugins/`. The model loads from disk; no network
is needed at startup.

## 4. First flight: test sequence

Use the **Beechcraft Baron 58** (Laminar default): two engines, retractable
gear, flaps, autopilot and flight director, so every command can be exercised.

| # | Do | Expect to hear | If not |
| --- | --- | --- | --- |
| 1 | Load the Baron on a runway, engines running | "Vimaan AI CoPilot ready" | `model_load` / `dependency_missing` in the log |
| 2 | **Ctrl-Z** (self-test, no mic) | "Self test complete. 10 of 10 passed." | `self_test_probe` events with `ok: false` |
| 3 | **Hold Z**, say "set heading two seven zero", release | "Heading two seven zero"; HDG bug moves | see the command trace (§5) |
| 4 | "set altitude seven thousand five hundred" | "Altitude 7500 feet" | |
| 5 | "tune com one to one one eight decimal seven five" | "COM 1 set to …"; COM1 shows 118.750 | |
| 6 | "parking brake on" then "parking brake off" | each confirmed | |
| 7 | "gear up" **on the ground** | "Confirm gear up, low altitude. Repeat to confirm." — do **not** repeat | `safety_hold` event |
| 8 | Take off, climb above 1,000 ft AGL: "gear up" | "Gear up" | |
| 9 | "flaps up" | "Flaps up" | |
| 10 | "flight director on", "autopilot on" | each confirmed; AP engages | `verify` event with `ok: false` |
| 11 | "climb to flight level one two zero" | "Flight level one two zero" | |
| 12 | "autopilot off" | "Autopilot 1 off" | |
| 13 | "shut down engine two" (airborne) | asks to confirm — say it again only if you mean it | |

On the ground with engines off you can also try "start engine one": the
plugin holds the starter until the engine runs (up to 20 s) and says
"Engine 1 running", or "did not start" with mixture/ignition/battery values in
the log.

**Voice worker process:** the microphone, speech recognition and the model run in a
separate process started with XPPython3's `xp.pythonExecutable` (logged as
`worker_spawn`; its own output goes to `~/Vimaan_Logs/vimaan_<ts>.worker.log`).
Inside X-Plane, Python background threads only run when the sim's main thread
releases the GIL, so doing this work in-process cost a ~118 ms frame hitch per
command (measured in-sim). Now the plugin only sends start/stop messages and
reads replies without blocking: measured ≤0.4 ms per frame in every phase,
~27 MB in X-Plane instead of ~900 MB, and X-Plane keeps rendering while the
model loads (~2-3 s; Vimaan says "ready" when done). If the worker crashes it
is logged (`worker_exit` with the tail of its log) and restarted up to 3 times.

To replay a recording instead of using the mic (debugging):
`VIMAAN_WORKER_AUDIO_FILE=/path/to/16k-mono.wav` before launching X-Plane.

**Talking:** push-to-talk is *hold*-to-talk — recording runs while Z (or the
bound button) is held and stops on release (12 s cap; taps under 0.3 s are
ignored). Start speaking a beat after pressing — the mic takes ~0.2 s to open. The copilot stays quiet while you talk so its voice isn't recorded.

**Overlay:** a bar at the bottom of the screen shows what Vimaan hears
*while you talk* (Vosk), then what it heard, what it understood (command,
values, confidence) and what happened, colour-coded: green done, amber
confirm / repeat / rejected (with the reason), red errors or "did not
respond". It appears when you press push-to-talk and fades ~6 s after the
result. It's an ordinary X-Plane window: drag it, resize it, or pop it out to
another monitor. Bind **Vimaan: overlay auto / pinned / off** (command
`vimaan/toggle_overlay`) to cycle modes, or set `VIMAAN_OVERLAY=pinned|off`
before launch. Drawn with ImGui (bundled with XPPython3), falling back to
X-Plane's own text if ImGui can't load; the log report shows which, and the
worst draw time per frame.

**"To" and "two":** they sound the same, so avoid "to" before a value. Say "altitude one thousand" and "heading zero one zero". When both readings of a heading, altitude or flight level are valid and differ, Vimaan changes nothing and asks which you meant (overlay: WHICH?). Heading and flight level given as all three digits ("two seven zero") are taken as digits, which means "heading to seven zero" can still set 270.

**Target vs aircraft:** commands set targets and don't engage autopilot modes. `aircraft_state` events record modes, actual heading and altitude, and COM values when a target or radio is set and 10 s and 30 s later; `autopilot_state_change` records changes Vimaan didn't make. The log report shows both.

**Push-to-talk on a yoke button:** Settings → Joystick → search `vimaan` →
bind **Vimaan: push to talk** (and optionally **Vimaan: run self test**).

## 5. Reading what happened

```bash
python scripts/vimaan_log_report.py          # newest session
python scripts/vimaan_log_report.py --list
```

The report shows how the session ended (clean stop vs **crash**), startup and
per-stage latency (p50/p95/max), every command with its transcript, intent,
slots and outcome, and every problem: errors with tracebacks, slow stages,
sim stalls, failed dataref writes, and post-command verifications that didn't
match.

Raw files in `~/Vimaan_Logs/` per session:

| File | Contents |
| --- | --- |
| `vimaan_<ts>.log` | human-readable log (also mirrored into XPPython3's log) |
| `vimaan_<ts>.events.jsonl` | one JSON event per line; `grep c0007` follows one command end to end |
| `vimaan_<ts>.faults.log` | native-crash stack traces and hang dumps (empty is good) |

X-Plane's own logs, for crashes outside Python: `<X-Plane 12>/Log.txt` and
`<X-Plane 12>/XPPython3Log.txt`.

### What the telemetry catches

| Symptom | Event | Notes |
| --- | --- | --- |
| Plugin crashed X-Plane | no `plugin_stop`; stack in `faults.log` | `faulthandler` writes every thread's stack on SIGSEGV/SIGABRT |
| X-Plane froze | `faults.log` hang dump | the main thread hasn't called back for 30 s |
| Sim hitched / stalled | `sim_stall`, `main_thread_overrun` | >2 s between frames we saw; >20 ms spent in our callback |
| Slow command | `stage` with `slow: true` | budgets: STT 3 s, NLU 150 ms, dispatch 30 ms, total 4 s |
| Wrong action | `stage nlu` (transcript, normalized text, intent, slots, confidence) → `sim_write` / `sim_command` → `verify` | |
| Mic dead/muted | `command_done outcome=silent_mic`; `rms` on `audio_level` | `microphone_ready` shows which input device is used |
| No mic permission | `microphone_unavailable`, or capture with `rms` ≈ 0 | System Settings → Privacy & Security → Microphone → enable X-Plane (and Steam, if launched from Steam) |
| Memory | `peak_rss_mb` on `heartbeat` / `session_end` | X-Plane's whole process (scenery included, ~5 GB in flight); torch + model live in the worker process, not here |

## 6. Speech recognition

Default is `vosk` (offline, streaming, restricted to the command words);
`google` is the online alternative. See [OFFLINE_STT.md](OFFLINE_STT.md); set
`VIMAAN_STT_BACKEND` before launching X-Plane. Every command's audio is saved
to `~/Vimaan_Logs/audio/` for comparing recognisers on real voices
(`VIMAAN_SAVE_AUDIO=0` turns it off).
