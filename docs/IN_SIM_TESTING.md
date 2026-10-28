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
torch transformers tokenizers safetensors numpy SpeechRecognition sounddevice word2number num2words
```

or, equivalently, from a terminal with XPPython3's own interpreter:

```bash
PY="<X-Plane 12>/Resources/plugins/XPPython3/mac_x64/python3.12/bin/python3.12"
"$PY" -m pip install torch transformers tokenizers safetensors numpy \
    SpeechRecognition sounddevice word2number num2words
```

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
python scripts/install_plugin.py --xplane "$X" --dry-run
python scripts/install_plugin.py --xplane "$X"
```

This copies the plugin, the runtime `ML/` package and the latest model (v11)
into `Resources/plugins/PythonPlugins/`. The model loads from disk; no network
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

**Talking:** push-to-talk is *hold*-to-talk — recording runs while Z (or the
bound button) is held and stops on release (12 s cap; taps under 0.3 s are
ignored). The copilot stays quiet while you talk so its voice isn't recorded.

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
| Memory | `peak_rss_mb` on `heartbeat` / `session_end` | expect ~1 GB for torch + model |

## 6. Switching speech backends

Default is `google` (online, nothing to install). For offline recognition see
[OFFLINE_STT.md](OFFLINE_STT.md); set `VIMAAN_STT_BACKEND` before launching
X-Plane. The `stt` stage latency in the report is the number to compare.
