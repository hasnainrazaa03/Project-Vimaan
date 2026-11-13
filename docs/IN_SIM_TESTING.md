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
| 7 | "gear up" **on the ground** | "Confirm gear up, low altitude. Say override, or repeat to confirm." — do **not** repeat (then say "cancel") | `safety_hold` event |
| 8 | Take off and, once climbing, "gear up" | "Gear up" (no confirmation while climbing at 300 fpm or more) | |
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

**"To" and "two":** they sound the same, so avoid "to" before a value. Say "altitude one thousand" and "heading zero one zero". When both readings of a heading, altitude or flight level are valid and differ, Vimaan changes nothing and asks which you meant (overlay: WHICH?). You can answer with the number alone ("one thousand", "zero two zero"). It is used only if it is one of the two values offered and you reply within 12 seconds. Any other number on its own is answered with "Say the full command" or "Say it with the word heading", and nothing changes. Heading and flight level given as all three digits ("two seven zero") are taken as digits, which means "heading to seven zero" can still set 270.

**Baron autopilot:** HDG follows the heading bug, so a voice heading turns the aircraft once HDG is on. ALT holds the altitude you are at, and pressing it replaces the selected altitude with the current one. A voice altitude is only a number on the display until there is a mode that climbs or descends to it.

**Climbs and descents (Tier 1):** "descend at one thousand five hundred" engages V/S, sets the rate, and has the autopilot level off at the selected altitude ("Descend 1500 feet per minute, to 3000"). Where the aircraft can arm altitude capture, Vimaan arms it (if the aircraft hasn't already). Where it can't, Vimaan can level off for you: bind **Vimaan: altitude capture by Vimaan on/off** (`vimaan/toggle_capture`), or set `VIMAAN_CAPTURE=1` before launch. It presses altitude hold just before the target and stands down if you change mode, move the target, or say "cancel". Otherwise the read-back says to say "altitude hold" at the target.

**Target vs aircraft:** commands other than modes and climbs set targets and don't engage autopilot modes. `aircraft_state` events record modes, actual heading and altitude, and COM values when a target or radio is set and 10 s and 30 s later; `autopilot_state_change` records changes Vimaan didn't make. The log report shows both.

**Autopilot disconnects:** when the autopilot drops and Vimaan didn't do it, Vimaan says so with the likely cause: "Autopilot disconnected. Stick moved." (pilot stick input in the last 4 s), "…by a switch or button." (an autopilot-off command from a key, joystick button or cockpit click), "…Overspeed." (above the aircraft's Vne / Vmo or Mmo: Airbus high-speed protection takes the autopilot), "…The autopilot has no power.", "…Autopilot failure.", or just "Autopilot disconnected." It stays quiet when you asked Vimaan for "autopilot off", during the probe, and for 10 s after a flight or aircraft loads. Each one is logged as `autopilot_disconnect` with the stick peak, the commands seen, power and fail flags and the joystick null zone; the null zone is also logged per aircraft as `joystick`. A null zone of 0% lets a resting stick fly the aircraft (and, on airliners, take the autopilot off): 10–15% is usual.

**Push-to-talk on a yoke button:** Settings → Joystick → search `vimaan` →
bind **Vimaan: push to talk** (and optionally **Vimaan: run self test**).

## 4b. Probing a new aircraft

To find out which controls an aircraft honours (Tier 1), run the probe once
on the ground and once in the air. It is pilot-paced, one key press per step,
and puts everything back after each step. Key bindings, steps and what to
expect are in [AIRCRAFT_PROFILES.md](AIRCRAFT_PROFILES.md#running-the-probe).
Results land in `~/Vimaan_Logs/aircraft/` and are used from the next load of
that aircraft.

## 4c. Automated tests: the test bridge (Phase A)

With X-Plane running and Vimaan loaded, a script on this Mac can say
commands to Vimaan and read or set the sim, with no one at the controls. The
bridge is **off** unless you switch it on, and Vimaan says "Test bridge on" /
"Test bridge off" when it changes. Don't fly while it's on.

```bash
python -m simtest launch           # X-Plane through Steam (pick "X-Plane" if Steam asks)
python -m simtest fly baron        # in the air over Seattle at 6,000 ft, running (or kingair, 737, a330)
python -m simtest bridge on        # creates ~/Vimaan_Logs/test/bridge.on
python -m simtest status           # Vimaan ready? aircraft, autopilot, altitude, speed
python -m simtest say "autopilot on"           # generated voice (macOS 'say'), through recognition
python -m simtest say --text "autopilot on"    # straight to the model: no recognition
python -m simtest say --wav ~/Vimaan_Logs/audio/20261107_203741_c0014.wav   # a real recording
python -m simtest read sim/cockpit2/autopilot/servos_on "sim/flightmodel/engine/ENGN_running[0]"
python -m simtest write sim/cockpit2/autopilot/altitude_dial_ft 5000
python -m simtest command sim/autopilot/heading            # counts as the pilot's, not Vimaan's
python -m simtest place --lat 47.45 --lon -122.31 --alt-ft 6000 --hdg 180 --kt 150
python -m simtest quit             # bridge off, then X-Plane quits
```

**The Tier 1 checklist, unattended (A2):**

```bash
python -m simtest launch
python -m simtest run tier1 baron        # or kingair, 737, a330, all; --text skips recognition
```

Each aircraft starts in the air west of Portland (every scenery tile around
it is installed; over Seattle X-Plane stops with "Missing scenery"). About 30
steps are said in a generated voice and each is checked against the sim:
values, modes, the turn to the new heading, the level-off after a V/S descent,
the autothrottle answer, altimeter, COM, the flap-speed hold, the autopilot
off without a callout, and a pilot disconnect with one. A voice that's
misunderstood is scored, then the step is redone as text so the check is about
the sim. Reports: `~/Vimaan_Logs/test/reports/<time>_<aircraft>.json`.
`python -m simtest screenshot` shows what has the sim paused, if it stops.

**The Tier 2 checklist (radios and transponder):**

```bash
python -m simtest run tier2 baron        # or kingair, 737, a330, all
```

The same start, with the autopilot flying (and the jets' thrust managed),
then: COM 1 standby and flip, COM 2 standby, NAV 1 110.50 and course 090,
NAV 2 standby, flip and course 180, squawk 4217, 0120 and 1200, an emergency
squawk held and cancelled, then the transponder: alt, standby, ident refused
in standby, on, ident, off (refused on the 737 and A330, which have none),
and alt again. The transponder is checked the way each aircraft shows it:
X-Plane's mode on the Baron and King Air, the selector position on the 737,
TA or TA/RA on the A330. It needs a model trained on Tier 2 (v20 or later)
and refuses to start on an older one.

Two clicks remain until they're switched off once: Steam's launch-option
question (set a default for X-Plane 12 in Steam), and X-Plane's "Quit the
simulator?" (tick "Don't ask again").

A said command answers when it's done and its read-backs are checked: what
was heard and understood, the outcome, Vimaan's replies, what was sent to
X-Plane, and the checks. It shows in the overlay and the Flights tab like a
spoken one (push-to-talk source `test`).

Generated voices are recognised like a real one, mistakes included. To see
which voices Vosk hears correctly, without X-Plane:

```bash
python -m simtest voices "descend at one thousand five hundred feet per minute" "mach point seven eight"
```

On ten Tier 1 phrases, Daniel (the default) was heard right 9 times out of 10,
Samantha, Karen and Moira 7, Fred 4 and Ralph 3. Use `--text` when a check is
about what the sim does, not what was heard.

How it works: requests are files in `~/Vimaan_Logs/test/inbox/`, results in
`test/outbox/` (`ML/vimaan_nlu/sim_bridge.py` documents the format). The
plugin looks four times a second, only for the switch file while the bridge
is off.

## 5. Reading what happened

**In the browser:** `python -m dashboard`, open http://127.0.0.1:8765/?tab=flights.
Every session is listed, the newest is followed live, and each command can be
replayed from its recording. See [dashboard/README.md](../dashboard/README.md#flights-what-happened-in-the-sim).

**In the terminal:**

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
