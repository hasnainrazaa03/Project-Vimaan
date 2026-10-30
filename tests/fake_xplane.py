"""A fake X-Plane (XPPython3 ``xp``) for exercising the real plugin in CI.

Knows only datarefs/commands that exist in X-Plane 12.4 (verified against its
DataRefs.txt / Commands.txt), so a handler that reaches for a misspelled or
removed name gets ``None`` back exactly as it would in the sim. Records every
write, command and spoken string for assertions.
"""

from __future__ import annotations

import types
from typing import Any

# name -> initial value (list for array datarefs). A plausible cold aircraft
# parked on the ground: twin, retractable gear, everything off.
DEFAULT_DATAREFS: dict[str, Any] = {
    "sim/cockpit2/autopilot/heading_dial_deg_mag_pilot": 0.0,
    "sim/cockpit2/autopilot/altitude_dial_ft": 0.0,
    "sim/cockpit2/radios/actuators/com1_frequency_hz_833": 122800,
    "sim/cockpit2/radios/actuators/com2_frequency_hz_833": 122800,
    "sim/cockpit2/controls/gear_handle_down": 1,
    "sim/aircraft/gear/acf_gear_retract": 1,
    "sim/cockpit2/controls/flap_handle_request_ratio": 0.0,
    "sim/flightmodel/controls/parkbrake": 0.0,
    "sim/cockpit2/autopilot/servos_on": 0,
    "sim/cockpit2/autopilot/servos2_on": 0,
    "sim/cockpit2/autopilot/flight_director_mode": 0,
    "sim/cockpit2/autopilot/flight_director2_mode": 0,
    "sim/flightmodel/engine/ENGN_running": [0] * 16,
    "sim/flightmodel/engine/ENGN_mixt": [1.0] * 16,
    "sim/cockpit2/engine/actuators/ignition_key": [3] * 16,
    "sim/cockpit2/electrical/battery_on": [1] * 8,
    "sim/aircraft/engine/acf_num_engines": 2,
    "sim/time/paused": 0,
    "sim/flightmodel/position/y_agl": 0.0,
    "sim/flightmodel/position/groundspeed": 0.0,
    "sim/flightmodel/failures/onground_any": 1,
    "sim/cockpit2/gauges/indicators/airspeed_kts_pilot": 0.0,
}

KNOWN_COMMANDS = {
    "sim/autopilot/servos_on",
    "sim/autopilot/servos_toggle",
    "sim/autopilot/servos2_on",
    "sim/autopilot/servos2_toggle",
    "sim/autopilot/fdir_on",
    "sim/autopilot/fdir_toggle",
    "sim/autopilot/fdir2_on",
    "sim/autopilot/fdir2_toggle",
    "sim/flight_controls/landing_gear_up",
    "sim/flight_controls/landing_gear_down",
    "sim/flight_controls/flaps_up",
    "sim/flight_controls/flaps_up_full",
    "sim/flight_controls/flaps_down",
    "sim/flight_controls/park_brake_set",
    "sim/flight_controls/park_brake_release",
    "sim/starters/engage_starter_1",
    "sim/starters/engage_starter_2",
    "sim/starters/shut_down_1",
    "sim/starters/shut_down_2",
}


def make_xp() -> types.ModuleType:
    xp = types.ModuleType("xp")
    xp.datarefs = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_DATAREFS.items()}
    xp.writes = []  # (name, value)
    xp.commands_fired = []  # command names via commandOnce
    xp.held = set()  # commands between commandBegin/commandEnd
    xp.spoken = []
    xp.custom_commands = {}  # name -> handler
    xp.hotkeys = []
    xp.flight_loops = []

    xp.VK_Z, xp.DownFlag, xp.UpFlag, xp.ControlFlag = 90, 1, 2, 4
    xp.CommandBegin, xp.CommandContinue, xp.CommandEnd = 0, 1, 2

    def findDataRef(name):
        return name if name in xp.datarefs else None

    def findCommand(name):
        if name in KNOWN_COMMANDS or name in xp.custom_commands:
            return name
        return None

    def getDataf(ref):
        return float(xp.datarefs[ref])

    def getDatai(ref):
        return int(xp.datarefs[ref])

    def setDataf(ref, value):
        xp.datarefs[ref] = float(value)
        xp.writes.append((ref, float(value)))

    def setDatai(ref, value):
        xp.datarefs[ref] = int(value)
        xp.writes.append((ref, int(value)))

    def getDatav(ref, values=None, offset=0, count=-1):
        arr = xp.datarefs[ref]
        if values is None:
            return len(arr)
        chunk = arr[offset:] if count < 0 else arr[offset : offset + count]
        values.extend(chunk)
        return len(chunk)

    def commandOnce(ref):
        xp.commands_fired.append(ref)

    def commandBegin(ref):
        xp.held.add(ref)
        xp.commands_fired.append(ref + ":begin")

    def commandEnd(ref):
        xp.held.discard(ref)
        xp.commands_fired.append(ref + ":end")

    def createCommand(name, description):
        xp.custom_commands[name] = None
        return name

    def registerCommandHandler(ref, callback, before=1, refCon=None):
        xp.custom_commands[ref] = callback

    def unregisterCommandHandler(ref, callback, before=1, refCon=None):
        xp.custom_commands.pop(ref, None)

    def registerHotKey(key, flags, description, callback, refCon=None):
        xp.hotkeys.append((key, flags, description, callback))
        return len(xp.hotkeys)

    def createFlightLoop(callback, phase=0, refCon=None):
        xp.flight_loops.append(callback)
        return len(xp.flight_loops)

    for fn in (
        findDataRef,
        findCommand,
        getDataf,
        getDatai,
        setDataf,
        setDatai,
        commandOnce,
        commandBegin,
        commandEnd,
        createCommand,
        registerCommandHandler,
        unregisterCommandHandler,
        registerHotKey,
        createFlightLoop,
    ):
        setattr(xp, fn.__name__, fn)
    xp.getDatavi = getDatav
    xp.getDatavf = getDatav
    xp.unregisterHotKey = lambda hid: None
    xp.scheduleFlightLoop = lambda lid, interval, rel=1: None
    xp.destroyFlightLoop = lambda lid: None
    xp.speakString = xp.spoken.append
    xp.log = lambda msg: None
    xp.getVersions = lambda: (124300, 420, 1)
    return xp


def make_speech_recognition() -> types.ModuleType:
    sr = types.ModuleType("speech_recognition")

    class UnknownValueError(Exception):
        pass

    class RequestError(Exception):
        pass

    class AudioData:
        def __init__(self, frame_data, sample_rate, sample_width):
            self.frame_data = frame_data
            self.sample_rate = sample_rate
            self.sample_width = sample_width

    class Recognizer:
        # Tests set this to the transcript (or an exception instance to raise).
        next_result: Any = "set heading two seven zero"

        def recognize_google(self, audio, language="en-US"):
            if isinstance(self.next_result, Exception):
                raise self.next_result
            return self.next_result

    sr.UnknownValueError = UnknownValueError
    sr.RequestError = RequestError
    sr.AudioData = AudioData
    sr.Recognizer = Recognizer
    sr.__version__ = "fake"
    return sr


def make_sounddevice() -> types.ModuleType:
    """Fake ``sounddevice`` in blocking mode: tests push PCM with
    ``sd.streams[-1].feed()``; the recorder drains it with ``read()``."""
    sd = types.ModuleType("sounddevice")
    sd.streams = []
    devices = [
        {"name": "Fake Mic", "max_input_channels": 1},
        {"name": "Fake Speakers", "max_input_channels": 0},
    ]

    class RawInputStream:
        def __init__(self, samplerate, channels, dtype, device=None, latency=None, callback=None):
            self.samplerate, self.latency, self.callback = samplerate, latency, callback
            self.active = self.closed = False
            self.overflow_next = False
            self._buf = bytearray()
            sd.streams.append(self)

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.closed = True

        def feed(self, raw: bytes) -> None:
            self._buf.extend(raw)

        @property
        def read_available(self) -> int:
            return len(self._buf) // 2

        def read(self, frames):
            data = bytes(self._buf[: frames * 2])
            del self._buf[: frames * 2]
            overflowed, self.overflow_next = self.overflow_next, False
            return data, overflowed

    def query_devices(device=None, kind=None):
        return devices[0] if kind == "input" else devices

    sd.RawInputStream = RawInputStream
    sd.query_devices = query_devices
    sd.__version__ = "fake"
    return sd
