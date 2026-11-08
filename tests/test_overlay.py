"""Tests for vimaan_nlu.overlay (drawing the overlay in X-Plane)."""

import types

from fake_xplane import make_xp
from vimaan_nlu.overlay import BOTTOM_MARGIN, HEIGHT, LEFT_MARGIN, WIDTH, Overlay
from vimaan_nlu.overlay_state import OverlayState


def press(state, cmd="c1"):
    state.on_event({"kind": "ptt_press", "cmd": cmd})


def test_native_window_hidden_then_shown_on_activity():
    xp = make_xp()
    state = OverlayState()
    overlay = Overlay(xp, state, prefer_imgui=False)
    assert overlay.start() == "native"
    (wid,) = xp.windows
    w = xp.windows[wid]
    assert w["visible"] == 0 and w["title"] == "Vimaan"
    left, top, right, bottom = w["geom"]  # bottom left of a 1920x1080 screen
    assert (right - left, top - bottom) == (WIDTH, HEIGHT)
    assert (left, bottom) == (LEFT_MARGIN, BOTTOM_MARGIN)
    overlay.update()
    assert w["visible"] == 0
    press(state)
    state.on_partial("c1", "gear up")
    overlay.update()
    assert w["visible"] == 1
    assert xp.draw_windows() == ["REC", "gear up"]
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


class FakeDrawList:
    def __init__(self, texts):
        self.texts, self.shapes = texts, []

    def add_rect_filled(self, *a):
        self.shapes.append(("rect", a))

    def add_circle_filled(self, *a):
        self.shapes.append(("circle", a))

    def add_text(self, x, y, col, text):
        self.texts.append(text)


class FakeImgui(types.SimpleNamespace):
    COLOR_WINDOW_BACKGROUND = 2

    def __init__(self):
        super().__init__(texts=[], io=types.SimpleNamespace(font_global_scale=1.0))
        self.style = types.SimpleNamespace(window_padding=None, colors={})
        self.draw_list = FakeDrawList(self.texts)
        self.cursor = (16.0, 12.0)

    def get_window_draw_list(self):
        return self.draw_list

    def get_color_u32_rgba(self, r, g, b, a):
        return (r, g, b, a)

    def get_window_position(self):
        return (0.0, 0.0)

    def get_window_height(self):
        return 118.0

    def get_text_line_height(self):
        return 19.0

    def get_cursor_screen_pos(self):
        return self.cursor

    def set_cursor_screen_pos(self, pos):
        self.cursor = pos

    def get_time(self):
        return 1.0

    def set_current_context(self, ctx):
        pass

    def get_io(self):
        return self.io

    def get_style(self):
        return self.style

    def text_colored(self, text, r, g, b, a):
        self.texts.append(text)

    def same_line(self, position=0.0):
        pass

    def get_window_width(self):
        return 720.0

    def calc_text_size(self, text):
        return types.SimpleNamespace(x=7.0 * len(text))


def test_imgui_backend():
    xp = make_xp()
    imgui = FakeImgui()
    created = {}

    class Window:
        def __init__(self, **kw):
            created.update(kw)
            self.imgui_context = object()
            self.windowID = xp.createWindowEx(
                kw["left"], kw["top"], kw["right"], kw["bottom"], 0, None
            )

        def delete(self):
            created["deleted"] = True

    state = OverlayState(mode="pinned")
    overlay = Overlay(xp, state, imgui_modules=(types.SimpleNamespace(Window=Window), imgui))
    assert overlay.start() == "imgui"
    assert imgui.io.font_global_scale > 1 and imgui.style.colors
    state.on_event({"kind": "ptt_press", "cmd": "c1"})
    state.on_partial("c1", "flaps up")
    created["draw"](overlay.window_id, None)
    assert imgui.texts == ["REC", "flaps up"]
    kinds = [k for k, _ in imgui.draw_list.shapes]
    assert kinds == ["rect", "rect", "circle"]  # outcome stripe, badge pill, record dot
    overlay.stop()
    assert created["deleted"]


# ---------------------------------------------------------------------------
# Wrapping long text, and growing the panel to fit
# ---------------------------------------------------------------------------
from vimaan_nlu.overlay import MAX_HEIGHT, wrap_text  # noqa: E402

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


def test_long_line_wraps_and_the_panel_grows_upward():
    xp = make_xp()
    state = OverlayState(mode="pinned")
    overlay = Overlay(xp, state, prefer_imgui=False)
    overlay.start()
    overlay.update()
    (wid,) = xp.windows
    left, _, right, bottom = xp.windows[wid]["geom"]
    long = "I heard " + "a very long transcript that goes on " * 6
    state.notice(long)
    drawn = xp.draw_windows()
    parts = [d for d in drawn if d != "VIMAAN"]
    assert len(parts) > 1 and " ".join(parts) == long.strip()
    overlay.update()  # applies the measured height
    new_left, new_top, new_right, new_bottom = xp.windows[wid]["geom"]
    assert (new_left, new_right, new_bottom) == (left, right, bottom)  # anchored bottom-left
    assert HEIGHT < new_top - new_bottom <= MAX_HEIGHT


def test_short_content_keeps_the_normal_height():
    xp = make_xp()
    state = OverlayState(mode="pinned")
    overlay = Overlay(xp, state, prefer_imgui=False)
    overlay.start()
    overlay.update()
    xp.draw_windows()
    overlay.update()
    (wid,) = xp.windows
    left, top, right, bottom = xp.windows[wid]["geom"]
    assert top - bottom == HEIGHT


def test_imgui_wraps_and_draws_the_probe_progress_bar():
    xp = make_xp()
    imgui = FakeImgui()

    class Window:
        def __init__(self, **kw):
            self.kw = kw
            self.imgui_context = object()
            self.windowID = xp.createWindowEx(
                kw["left"], kw["top"], kw["right"], kw["bottom"], 0, None
            )

        def delete(self):
            pass

    state = OverlayState(mode="pinned")
    overlay = Overlay(xp, state, imgui_modules=(types.SimpleNamespace(Window=Window), imgui))
    overlay.start()
    state.on_event({"kind": "probe_start", "where": "air", "steps": 4, "name": "BE58", "next": "x"})
    state.on_event({"kind": "probe_step", "label": "heading mode", "verdict": "works", "index": 0})
    overlay.window.kw["draw"](overlay.window_id, None)
    rects = [s for s in imgui.draw_list.shapes if s[0] == "rect"]
    assert len(rects) >= 5  # stripe, pills, progress track and fill
    imgui.texts.clear()
    state.on_event({"kind": "probe_end", "reason": "stopped", "message": "word " * 60})
    overlay.window.kw["draw"](overlay.window_id, None)
    wrapped = [t for t in imgui.texts if t.startswith("word")]
    assert len(wrapped) > 1
    assert overlay._wanted_height > HEIGHT
