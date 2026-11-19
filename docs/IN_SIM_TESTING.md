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
ignored). You can start speaking as you press: the mic stays open and each recording starts with the last 300 ms before the key went down (`VIMAAN_MIC_PREROLL_MS`, #57; checked in flight on 2026-11-16). The copilot stays quiet while you talk so its voice isn't recorded.

**No joystick?** Then X-Plane flies the aircraft with the mouse while its button is held in the 3-D view. To give X-Plane the keyboard for Z, click once and let go. On 2026-11-16 a drag pulled the yoke fully back: the autopilot dropped out ("Autopilot disconnected. Stick moved.") and the 737 climbed and slowed to 90 kt.

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

**Questions (Tier 4).** Hold **Shift+Z** (or bind **Vimaan: ask a question (hold)** to a button) to ask: whatever is said on it is only answered, never acted on. A command said there gets "I can't answer that". Questions are decoded with their own word list (`question_words.txt`); the talk key keeps the command word list, which never changes for questions (their words in it made commands worse: "flight level" heard as "light level").

- On the talk key a question works too, as long as Vosk hears how it starts (is, are, what, how, where, do, does, or "r the" / "our the" for "are the"): it is decoded again with the question words and answered, or refused with "I can't answer that". It is never acted on. A question whose first word went unheard can't be told from the command ("is the autopilot off" heard as "the autopilot off" switches it off), so for yes/no questions use Shift+Z, or ask without on/off: "autopilot status", "landing lights status".
- A reading named on its own is answered, never acted on: "heading", "what the heading", "say altitude", "what are squawk". Switch names on their own ("flaps", "gear") still ask which way.
- The questions are read by a question model of their own (`q1`, installed with `install_plugin.py --question-model q1`) beside the command model (v26), so they never change how commands are read. On the talk key it answers only what the command model would not do ("landing lights status" is not "landing lights on"); a system named alone ("flaps") still asks which way. Without a question model only those bare readings are answered.
- What you can ask, and how it answers (each answer names what it answers):

| Ask | Answer |
| --- | --- |
| what's our altitude / heading / airspeed / ground speed / vertical speed / mach | "Altitude 7400 feet", "Heading two seven zero", "Descending 1200 feet per minute", "Mach point seven eight" |
| how high above the ground are we | "1250 feet above the ground" (every aircraft, the Baron too) |
| what's the wind / outside temperature | "Wind two seven zero at 35 knots", "Outside air temperature minus 5 degrees Celsius" |
| what's the altimeter | "Altimeter two niner niner two", "Q N H one zero one three" (737/A330 set to hPa), "Altimeter standard" |
| how much fuel do we have / how long will our fuel last | "Fuel 23000 pounds" (Baron gallons, King Air and 737 pounds, A330 kg); "About 2 hours 10 minutes of fuel at this fuel flow" |
| what's the N1 / RPM / torque / ITT / EGT / oil pressure / fuel flow, engine status, is engine two running | "N1 85 percent on both engines"; only the readings this kind of engine shows |
| what's com one / nav two standby / the course on nav two / our squawk | "COM 1 one two one decimal niner, standby one three two decimal four five", "Squawk four two one seven, transponder altitude" |
| is the autopilot on, autopilot status, what altitude is set | "Autopilot on, heading two seven zero, altitude hold, altitude set 8000 feet", "Altitude set 8000 feet" |
| are the landing lights on, is the gear down, where are the flaps, is the parking brake set, what's the auto brake set to | "Landing lights on", "Gear down", "Flaps 15", "This aircraft has no auto brake" |
| how far to the next waypoint | "Next waypoint 12 miles, 4 minutes", or "No waypoint set" |
| what time is it / the local time, status report | "Time one four three five Zulu"; "Altitude 8400 feet, heading two seven zero, airspeed 250 knots, autopilot on" |

**"To" and "two":** they sound the same, so avoid "to" before a value. Say "altitude one thousand" and "heading zero one zero". When both readings of a heading, altitude or flight level are valid and differ, Vimaan changes nothing and asks which you meant (overlay: WHICH?). You can answer with the number alone ("one thousand", "zero two zero"). It is used only if it is one of the two values offered and you reply within 12 seconds. Any other number on its own is answered with "Say the full command" or "Say it with the word heading", and nothing changes. Heading and flight level given as all three digits ("two seven zero") are taken as digits, which means "heading to seven zero" can still set 270.

A frequency, auto brake setting or squawk that is impossible as heard, but valid with one "two" read as "to", is asked about instead of refused. For example, "set com one standby to one two two decimal eight" heard as "... standby two one two two decimal eight" (212.28) gets "Did you mean COM 1 standby one two two decimal eight? Say confirm, or cancel." "Confirm" (or "override") sets it, and so does saying the command again; "cancel" or any other command drops it. The aircraft check still applies (no auto brake on the Baron), and so do the safety holds: an emergency squawk asked about this way is then held for its own confirmation.

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
TA or TA/RA on the A330. It needs a model that knows Tier 2 (v20 or later; v25 passed the gate)
and refuses to start on an older one.

**The Tier 3 checklist (switches):**

```bash
python -m simtest run tier3 baron        # or kingair, 737, a330, all
```

The same start, then lights (beacon, landing, strobes, nav, taxi off, and
logo and strobes auto where the aircraft has them), cabin signs, yaw damper,
anti-ice (pitot / probe and window heat, wing and engine anti-ice, prop heat),
avionics, fuel pumps on, and on the 737 and A330 speed brake armed and stowed
and the autobrake set and off. Every switch is checked the way its aircraft
shows it: the 737's own switch positions, the A330's buttons, the King Air's
selectors, X-Plane's values on the Baron. A switch the aircraft lacks must be
refused ("This aircraft has no auto brake"), and generator, fuel pumps or
probe heat off in the air must be held for confirmation, then cancelled:
nothing is switched off for real (the A330's probe heat has no off, so there
the repeat confirms it and Vimaan asks "on or auto?"). 28 steps on the Baron,
31 on the King Air, 33 on the 737, 35 on the A330. It needs a model that knows Tier 3 (v26 or later).

**The Tier 4 checklist (questions):**

```bash
python -m simtest run tier4 baron --text   # or kingair, 737, a330, all
```

The same start, landing lights on, then every question family asked on the
question key (as Shift+Z): altitude, airspeed, heading, vertical speed,
ground speed, wind, outside temperature, altimeter, height above ground, Mach
on the jets, fuel, endurance, oil pressure and what the engine kind shows
(manifold pressure; torque and N1; E G T and N1), COM 1 and its standby, NAV
1, the squawk, autopilot status, altitude set and heading bug, landing
lights, gear, flaps, next waypoint, time and status. Each answer is checked
against the sim read just before the question and just after the answer (it
passes on either: the 737's EGT moved 9 °C between the answer and the read
after it): the numbers it says, within what moves in that time (60 ft, 5 kt, 3°, 1 °C, ...), in the cockpit's units from the
2026-11-16 probe (fuel in gallons on the Baron, pounds on the King Air and
737, kilograms on the A330; the altimeter in inches or hPa as the 737's and
A330's selectors show). No question may write to or command the sim. Then
"gear down" and "autopilot off" on the question key must be refused ("I
can't answer that") with nothing changed, and on the talk key "landing lights
status" and "what's our altitude" must be answered. 34 steps on the Baron, 35
on the King Air, 37 on the 737, 38 on the A330. It needs a question model
(q1) installed beside the command model. `--text` sends the questions as
text; without it a generated voice says them through the question grammar.

**Flights: takeoff, landing, go-around (Phase F.3):**

```bash
python -m simtest run takeoff baron --text     # or kingair, 737, a330, all
python -m simtest run landing baron --text
python -m simtest run go-around baron --text
```

Whole flights at Portland, runway 10R, so the phase of flight and the speed
margins are recorded on real takeoffs and landings (the tier checklists only
fly in the air). Every step checks the phase Vimaan's tracker shows.

- **Takeoff:** on the runway, engines running, parked. "gear up" is held on
  the ground (cancelled); altitude 4,000 and heading 103 set; jets set
  takeoff flaps; "parking brake off"; takeoff thrust. "parking brake on" at
  speed is held (cancelled). At Vr the runner pulls the stick until the nose
  is up and lets go; "autopilot on" and a 1,500 fpm climb straight after;
  "gear up" from 200 ft; then climb thrust at once: jets the autothrottle
  and speed 210, flaps up from 190 (737) or 200 kt (A330); the Baron and
  King Air hold 120 and 140 kt on the runner's throttle. Phases: parked,
  taxi (the first 40 kt), takeoff roll, initial climb, climb, cruise at
  4,000 ft.
- **Landing:** from an 8 nm final (X-Plane starts it about 1,500 ft up,
  gear and flaps up, autopilot off). Autopilot on, altitude hold, heading
  103, the 10R ILS (110.50, course 103), approach mode; on the glideslope,
  gear down, landing flaps and the final speed. Jets: autothrottle off and
  thrust idle at 50 ft. On the ground: thrust idle, brakes, autopilot off.
  Phases: approach, landing (the last 100 ft), rollout, taxi, parked.
- **Go-around:** the same approach to 500 ft, then full thrust and a
  1,500 fpm climb to 3,000 ft, gear up, flaps one step up (the pistons then
  hold their climb speed). Phases: approach, go-around, climb, cruise.

What the runner does as the pilot: the thrust levers, the stick for the
rotation, the throttle on the Baron and King Air (held on an airspeed, as
they have no autothrottle) and the brakes. At the end of a flight, or if it
stops, the throttle hold stops and the stick is centred.

**The A330 takes off but doesn't land or go around yet.** On its approach
LOC and G/S stayed armed and it flew on past the airport: its autopilot
follows NAV 1, which on the A330 gets no localizer (the deviations read
exactly 0), while its own ILS receiver (NAV 3) did receive IPDX. Flying its
ILS needs its radio page, LS and APPR, which Vimaan's A330 profile doesn't
do yet (a follow-up). `run landing a330` says so and skips it.

A flight that misses the glideslope turns back west and holds its altitude,
and the rest of it isn't run: flying on east, it leaves the installed
scenery (east of about 122° W), where X-Plane pauses under a "Missing
scenery" dialog that only a click closes and the bridge stops answering.

Recorded flights from these runs replay through the tracker as test
fixtures (`tests/fixtures/flights/`, `tests/test_flight_replays.py`).

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
match. Two performance sections answer "does Vimaan slow the game?":

- **Game performance**: X-Plane's frame rate and frame times per minute
  (median, p95, worst, frames over 50 and 100 ms), and the same for the frames
  from each key press to done, next to the whole flight's.
- **Mac usage**, sampled every 15 s by the voice worker (outside X-Plane):
  CPU (% of one core) and memory of X-Plane and of the worker, GPU
  utilisation and memory, macOS's free-memory level, swap and load.
  `--usage-every 0` on the worker turns it off.
- **Flight phases** (Phase F.1, observe only): the phase of flight as it
  changed (parked, taxi, takeoff roll, initial climb, climb, cruise,
  descent, approach, landing, go-around, rollout), with height, speeds, gear
  and flaps at each change, the time spent in each, and whether the
  recorded samples replay to the same phases through today's tracker
  (`vimaan_nlu/flight_phase.py`; where they don't, when and how: "t=193s
  approach -> descent"). The aircraft is sampled once a second and
  recorded in the events file (`flight`), not the log, from the sixth second
  after a load on (an air start's airspeed reads 0 for a couple of seconds). With it, the speed
  margins (Phase F.2, `vimaan_nlu/envelope.py`): the lowest margin to the
  stall, flap, gear, Vne and Mmo limits in each phase, and every stretch
  beyond one; and with callouts switched on (Phase F.4,
  `vimaan/toggle_callouts`), each callout said and when it was first said.
  The dashboard's Flights tab shows the phase changes and callouts in the
  timeline ("Phases of flight" hides them) and the route of phases.

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
| Game slowed | `frames` (once a minute), `frames` on `command_done` | X-Plane's own frame times, recorded by a callback that runs every frame and does nothing else |
| Mac busy | `usage` (every 15 s) | from `ps`, `ioreg` and `sysctl`, run by the worker; no admin rights needed |
| Memory | `peak_rss_mb` on `heartbeat` / `session_end` | X-Plane's whole process (scenery included, ~5 GB in flight); torch + model live in the worker process, not here |

## 6. Speech recognition

Default is `vosk` (offline, streaming, restricted to the command words);
`google` is the online alternative. See [OFFLINE_STT.md](OFFLINE_STT.md); set
`VIMAAN_STT_BACKEND` before launching X-Plane. Every command's audio is saved
to `~/Vimaan_Logs/audio/` for comparing recognisers on real voices
(`VIMAAN_SAVE_AUDIO=0` turns it off).

The mic stays open while X-Plane runs, so macOS shows its mic indicator the
whole time. Each command starts with the last 300 ms of sound from before the
key went down, so the first word isn't cut off by the mic opening (~85 ms) or by
speaking as you press. Between presses only the newest 300 ms is kept, in
memory; nothing else is saved or recognised. `VIMAAN_MIC_PREROLL_MS=0` opens the
mic on each press instead, as before; another number changes the length.
