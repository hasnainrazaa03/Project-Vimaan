"""Tests for vimaan_nlu.overlay (drawing the overlay in X-Plane)."""

import types

import pytest
from fake_xplane import make_xp
from vimaan_nlu.overlay import (
    BODY_PX,
    FONT_OVERSAMPLE,
    IMGUI_FONT_SCALE,
    LEFT_MARGIN,
    MAX_HEIGHT,
    MIN_HEIGHT,
    SMALL_PX,
    TOP_MARGIN,
    WIDTH,
    Overlay,
    fit_text,
    wrap_text,
)
from vimaan_nlu.overlay_state import OverlayState


def press(state, cmd="c1"):
    state.on_event({"kind": "ptt_press", "cmd": cmd})


def finish(state, cmd="c1", confidence=0.98):
    for ev in (
        {"kind": "stage", "stage": "stt", "cmd": cmd, "text": "heading two seven zero"},
        {
            "kind": "stage",
            "stage": "nlu",
            "cmd": cmd,
            "intent": "set_autopilot_heading",
            "slots": {"degrees": "270"},
            "confidence": confidence,
        },
        {"kind": "command_done", "cmd": cmd, "outcome": "executed", "speech_to_action_ms": 21.0},
    ):
        state.on_event(ev)


def test_native_window_at_the_top_left_hidden_then_shown_on_activity():
    xp = make_xp()
    state = OverlayState()
    overlay = Overlay(xp, state, prefer_imgui=False)
    assert overlay.start() == "native"
    (wid,) = xp.windows
    w = xp.windows[wid]
    assert w["visible"] == 0 and w["title"] == "Vimaan"
    left, top, right, bottom = w["geom"]  # top left of a 1920x1080 screen
    assert (left, top) == (LEFT_MARGIN, 1080 - TOP_MARGIN)
    assert (right - left, top - bottom) == (WIDTH, MIN_HEIGHT)
    overlay.update()
    assert w["visible"] == 0
    press(state)
    state.on_partial("c1", "gear up")
    overlay.update()
    assert w["visible"] == 1
    assert xp.draw_windows() == ["LISTEN", "HEARD", "gear up_", "ACTION", "----", "CONF", "--"]
    assert overlay.draw_ms_max >= 0


def test_draw_never_raises():
    xp = make_xp()
    state = OverlayState(mode="pinned")
    logs = []
    overlay = Overlay(xp, state, log=logs.append, prefer_imgui=False)
    overlay.start()
    overlay.update()

    def boom():
        raise RuntimeError("bad line")

    state.lines = boom
    xp.draw_windows()  # must not propagate into X-Plane
    assert overlay.draw_errors == 1 and "draw failed" in logs[-1]


def test_falls_back_to_native_without_imgui():
    xp = make_xp()
    logs = []
    overlay = Overlay(xp, OverlayState(), log=logs.append)  # no XPPython3 / imgui here
    assert overlay.start() == "native"
    assert "ImGui unavailable" in logs[0]


def test_stop_destroys_window():
    xp = make_xp()
    overlay = Overlay(xp, OverlayState(), prefer_imgui=False)
    overlay.start()
    overlay.stop()
    assert xp.windows == {} and overlay.window_id is None


# ---------------------------------------------------------------------------
# The ImGui renderer (design D), with a fake ImGui
# ---------------------------------------------------------------------------
class FakeDrawList:
    def __init__(self, texts):
        self.texts, self.shapes = texts, []

    def add_rect_filled(self, *a):
        self.shapes.append(("fill", a))

    def add_rect(self, *a):
        self.shapes.append(("rect", a))

    def add_line(self, *a):
        self.shapes.append(("line", a))

    def add_circle_filled(self, *a):
        self.shapes.append(("circle", a))

    def add_text(self, x, y, col, text):
        self.texts.append(text)
        self.shapes.append(("text", (x, y, col, text)))


class FakeFonts:
    def __init__(self):
        self.added = []

    def add_font_from_file_ttf(self, path, size):
        self.added.append((path, size))
        return f"font{len(self.added)}"


class FakeImgui(types.SimpleNamespace):
    COLOR_WINDOW_BACKGROUND = 2

    def __init__(self):
        super().__init__(texts=[], io=types.SimpleNamespace(font_global_scale=1.0))
        self.io.fonts = FakeFonts()
        self.style = types.SimpleNamespace(window_padding=None, window_border_size=1, colors={})
        self.draw_list = FakeDrawList(self.texts)
        self.fonts_pushed = []
        self.clicked = False
        self.mouse = (0.0, 0.0)

    def get_window_draw_list(self):
        return self.draw_list

    def get_color_u32_rgba(self, r, g, b, a):
        return (r, g, b, a)

    def get_window_position(self):
        return (0.0, 0.0)

    def get_window_height(self):
        return 118.0

    def get_window_width(self):
        return float(WIDTH)

    def get_text_line_height(self):
        return 16.0

    def get_time(self):
        return 1.0

    def set_current_context(self, ctx):
        pass

    def get_io(self):
        return self.io

    def get_style(self):
        return self.style

    def calc_text_size(self, text):
        return types.SimpleNamespace(x=8.0 * len(text))

    def push_font(self, font):
        self.fonts_pushed.append(font)

    def pop_font(self):
        pass

    def is_mouse_clicked(self, button=0):
        return self.clicked

    def get_mouse_pos(self):
        return self.mouse


def imgui_overlay(state, tmp_path=None, font=True, **kw):
    xp = make_xp()
    imgui = FakeImgui()
    refreshed = []

    class Window:
        def __init__(self, **kwargs):
            self.kw = kwargs
            self.imgui_context = object()
            self.renderer = types.SimpleNamespace(refresh_font_texture=lambda: refreshed.append(1))
            self.windowID = xp.createWindowEx(
                kwargs["left"], kwargs["top"], kwargs["right"], kwargs["bottom"], 0, None
            )

        def delete(self):
            refreshed.append("deleted")

    font_dir = None
    if tmp_path is not None:
        font_dir = str(tmp_path)
        if font:
            (tmp_path / "B612Mono-Regular.ttf").write_bytes(b"ttf")
    overlay = Overlay(
        xp,
        state,
        imgui_modules=(types.SimpleNamespace(Window=Window), imgui),
        font_dir=font_dir,
        **kw,
    )
    assert overlay.start() == "imgui"

    def draw():
        imgui.texts.clear()
        imgui.draw_list.shapes.clear()
        overlay.window.kw["draw"](overlay.window_id, None)

    return xp, imgui, overlay, draw, refreshed


def test_imgui_loads_b612_mono_from_x_planes_fonts(tmp_path):
    state = OverlayState(mode="pinned")
    _, imgui, overlay, draw, refreshed = imgui_overlay(state, tmp_path)
    assert overlay.font == "B612Mono-Regular.ttf"
    sizes = [size for _, size in imgui.io.fonts.added]
    assert sizes == [BODY_PX * FONT_OVERSAMPLE, SMALL_PX * FONT_OVERSAMPLE]
    assert refreshed == [1] and imgui.io.font_global_scale == pytest.approx(1 / FONT_OVERSAMPLE)
    assert imgui.style.window_border_size == 0.0 and imgui.style.colors
    draw()
    assert set(imgui.fonts_pushed) == {"font1", "font2"}
    overlay.stop()
    assert refreshed[-1] == "deleted"


def test_imgui_without_a_ttf_scales_its_own_font(tmp_path):
    _, imgui, overlay, draw, _ = imgui_overlay(OverlayState(mode="pinned"), tmp_path, font=False)
    assert overlay.font is None and imgui.io.font_global_scale == IMGUI_FONT_SCALE
    draw()
    assert imgui.fonts_pushed == [] and "PIN" in imgui.texts


def test_imgui_listening():
    state = OverlayState(mode="pinned")
    _, imgui, _, draw, _ = imgui_overlay(state)
    press(state)
    state.on_partial("c1", "flaps up")
    draw()
    assert imgui.texts[:4] == ["LISTEN", "KEY", "OFF", "PIN"]
    assert "flaps up_" in imgui.texts and "----" in imgui.texts


def text_at(imgui, text):
    return [s[1] for s in imgui.draw_list.shapes if s[0] == "text" and s[1][3] == text][0]


def test_imgui_confidence_bar_with_the_floor_marked():
    state = OverlayState(mode="pinned", floor=0.55)
    _, imgui, _, draw, _ = imgui_overlay(state)
    press(state)
    finish(state, confidence=0.5)
    draw()
    assert "50%" in imgui.texts and "HDG 270" in imgui.texts
    fills = [s[1] for s in imgui.draw_list.shapes if s[0] == "fill"]
    _, conf_y, *_ = text_at(imgui, "CONF")
    bars = [f for f in fills if conf_y <= f[1] <= conf_y + 16]
    track, fill, tick = bars[:3]
    width = track[2] - track[0]
    assert fill[2] - fill[0] == pytest.approx(width * 0.5)
    assert (tick[0] + tick[2]) / 2 == pytest.approx(track[0] + width * 0.55)
    assert fill[4][:3] == pytest.approx((0xF5 / 255, 0xB8 / 255, 0x41 / 255))  # amber: below it


def test_imgui_history_fades_down_the_list():
    from vimaan_nlu.overlay_state import HISTORY_SEC

    clock = types.SimpleNamespace(now=100.0)
    state = OverlayState(lambda: clock.now)
    _, imgui, _, draw, _ = imgui_overlay(state)
    for cmd in ("c1", "c2"):
        press(state, cmd)
        finish(state, cmd)
        clock.now += 30
    draw()
    assert [t for t in imgui.texts if t.startswith("-")] == ["-0:30", "-1:00"]
    newest, older = (
        text_at(imgui, "HDG 270")[2],
        [s[1][2] for s in imgui.draw_list.shapes if s[0] == "text" and s[1][3] == "HDG 270"][1],
    )
    assert newest[:3] > older[:3]  # brighter ink for the newest
    clock.now = 100.0 + HISTORY_SEC - 1  # c1 nearly gone
    draw()
    alphas = [s[1][2][3] for s in imgui.draw_list.shapes if s[0] == "text" and s[1][3] == "HDG 270"]
    assert alphas[-1] < 0.1


def click(imgui, overlay, draw, name):
    x0, y0, x1, y1 = overlay._buttons[name]
    imgui.mouse, imgui.clicked = ((x0 + x1) / 2, (y0 + y1) / 2), True
    draw()
    imgui.clicked = False


def test_pin_and_off_buttons():
    state = OverlayState(mode="auto")
    modes = []

    def on_mode(mode):
        modes.append(mode)
        state.mode = mode

    _, imgui, overlay, draw, _ = imgui_overlay(state, on_mode=on_mode)
    press(state)
    draw()
    click(imgui, overlay, draw, "PIN")
    assert modes == ["pinned"]
    click(imgui, overlay, draw, "PIN")
    click(imgui, overlay, draw, "OFF")
    assert modes == ["pinned", "auto", "off"]
    imgui.mouse, imgui.clicked = (5.0, 60.0), True  # a click anywhere else
    draw()
    assert modes == ["pinned", "auto", "off"]


def test_pin_is_lit_when_pinned():
    state = OverlayState(mode="pinned")
    _, imgui, _, draw, _ = imgui_overlay(state)
    draw()
    assert text_at(imgui, "PIN")[2][:3] == pytest.approx((0x5E / 255, 0xD3 / 255, 0xF3 / 255))
    assert text_at(imgui, "OFF")[2][:3] == pytest.approx((0x8A / 255, 0x92 / 255, 0x9B / 255))


def test_imgui_checklist_leaders_and_the_offer_box():
    clock = types.SimpleNamespace(now=100.0)
    state = OverlayState(lambda: clock.now, mode="pinned")
    _, imgui, _, draw, _ = imgui_overlay(state)
    state.on_event(
        {
            "kind": "checklist_start",
            "cmd": "c1",
            "checklist": "before landing",
            "items": ["gear", "speed brakes"],
        }
    )
    state.on_event({"kind": "checklist_item", "item": "gear", "state": "down", "result": "checked"})
    state.on_event(
        {
            "kind": "checklist_item",
            "item": "speed brakes",
            "state": "down",
            "result": "offered",
            "ask": "Set speed brakes armed",
        }
    )
    draw()
    assert "CHKL" in imgui.texts and "BEFORE LANDING" in imgui.texts
    assert any(set(t) == {"."} and len(t) > 5 for t in imgui.texts)  # the leaders
    assert "SET SPEED BRAKES ARMED? SAY CONFIRM / CANCEL" in imgui.texts
    boxes = [s for s in imgui.draw_list.shapes if s[0] == "rect"]
    assert len(boxes) == 2  # the panel's border, the offer's box


def test_imgui_panel_grows_downward_from_the_top(tmp_path):
    state = OverlayState(mode="pinned")
    xp, _, overlay, draw, _ = imgui_overlay(state)
    overlay.update()
    (wid,) = xp.windows
    left, top, right, _ = xp.windows[wid]["geom"]
    state.notice("I heard " + "a very long reply that goes on " * 8)
    draw()
    overlay.update()
    new_left, new_top, new_right, new_bottom = xp.windows[wid]["geom"]
    assert (new_left, new_top, new_right) == (left, top, right)  # anchored top-left
    assert MIN_HEIGHT < new_top - new_bottom <= MAX_HEIGHT


def test_short_content_keeps_a_small_panel():
    state = OverlayState(mode="pinned")
    xp, _, overlay, draw, _ = imgui_overlay(state)
    overlay.update()
    draw()
    overlay.update()
    (wid,) = xp.windows
    _, top, _, bottom = xp.windows[wid]["geom"]
    assert top - bottom < 60


# ---------------------------------------------------------------------------
# Wrapping and fitting text
# ---------------------------------------------------------------------------
PX = 8.0  # the fake font: 8 px per character


def measure(text):
    return PX * len(text)


def test_wrap_breaks_between_words_and_never_loses_text():
    text = "Probe stopped. Heading mode did not restore. Check the autopilot."
    parts = wrap_text(text, 20 * PX, measure)
    assert all(measure(p) <= 20 * PX for p in parts)
    assert " ".join(parts) == text
    assert wrap_text("short", 100, measure) == ["short"]
    assert wrap_text("", 100, measure) == [""]


def test_a_word_wider_than_the_line_is_split_not_cut():
    parts = wrap_text("x" * 25, 10 * PX, measure)
    assert parts == ["x" * 10, "x" * 10, "x" * 5]


def test_fit_text_cuts_with_dots():
    assert fit_text("HDG 270", 100, measure) == "HDG 270"
    assert fit_text("BEFORE LANDING, 1 OPEN", 10 * PX, measure) == "BEFORE L.."
    assert fit_text("ABC", 1, measure) == ""
