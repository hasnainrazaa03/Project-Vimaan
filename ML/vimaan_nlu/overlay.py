"""In-sim overlay: a bar at the bottom of the screen showing what Vimaan
heard (live while you talk), what it understood, and what happened.

Draws ``OverlayState.lines()`` in an X-Plane window. Prefers Dear ImGui
(bundled with XPPython3 as ``xp_imgui`` + ``imgui``: colours, scaling) and
falls back to X-Plane's own text drawing if ImGui can't load. The window is a
normal X-Plane floating window, so it can be dragged, resized, or popped out
to another monitor; it is hidden when there's nothing to show (auto mode).

Everything here runs on X-Plane's main thread inside draw callbacks, so the
draw functions never raise (an exception would kill the ImGui window for the
session) and their CPU time is measured (``draw_ms_max``).
"""

from __future__ import annotations

import importlib
import time
from typing import Any

from vimaan_nlu.overlay_state import Line, OverlayState

WIDTH, HEIGHT = 720, 112
BOTTOM_MARGIN = 70  # px above the bottom of the main screen
BADGE_COLUMN = 104  # px reserved for the status badge (native renderer)
LINE_HEIGHT = 26
IMGUI_FONT_SCALE = 1.5

TONES: dict[str, tuple[float, float, float]] = {
    "live": (1.00, 0.36, 0.33),  # recording red
    "text": (0.93, 0.95, 0.97),
    "muted": (0.62, 0.66, 0.72),
    "accent": (0.45, 0.75, 1.00),
    "ok": (0.40, 0.86, 0.50),
    "warn": (1.00, 0.76, 0.28),
    "error": (1.00, 0.40, 0.38),
}


def text_colour(tone: str) -> tuple[float, float, float]:
    """The line's text: plain white for good/neutral news, the tone colour
    when it needs attention (recording, warnings, errors)."""
    if tone in ("ok", "accent", "text"):
        return TONES["text"]
    return TONES[tone]


class Overlay:
    def __init__(
        self,
        xp: Any,
        state: OverlayState,
        log: Any = print,
        *,
        prefer_imgui: bool = True,
        imgui_modules: tuple[Any, Any] | None = None,
    ) -> None:
        self.xp = xp
        self.state = state
        self.log = log
        self.prefer_imgui = prefer_imgui
        self._imgui_modules = imgui_modules  # (xp_imgui, imgui) — injectable for tests
        self.backend: str | None = None  # "imgui" | "native"
        self.window: Any = None  # xp_imgui.Window (imgui backend)
        self.window_id: Any = None
        self._shown = False
        self.draw_ms_max = 0.0
        self.draw_errors = 0

    # ------------------------------------------------------------------
    def start(self) -> str:
        """Create the (hidden) window; returns the backend used."""
        left, top, right, bottom = self._initial_geometry()
        if self.prefer_imgui:
            try:
                self._start_imgui(left, top, right, bottom)
            except Exception as exc:
                self.log(f"overlay: ImGui unavailable ({type(exc).__name__}: {exc}); using native")
        if self.backend is None:
            self.window_id = self.xp.createWindowEx(
                left,
                top,
                right,
                bottom,
                0,
                self._draw_native,
                None,
                None,
                None,
                None,
                None,
                self.xp.WindowDecorationRoundRectangle,
                self.xp.WindowLayerFloatingWindows,
                None,
            )
            self.backend = "native"
        xp = self.xp
        xp.setWindowTitle(self.window_id, "Vimaan")
        xp.setWindowPositioningMode(self.window_id, xp.WindowPositionFree, -1)
        xp.setWindowResizingLimits(self.window_id, 380, 70, 1600, 260)
        return self.backend

    def _initial_geometry(self) -> tuple[int, int, int, int]:
        try:
            left, _top, right, bottom = self.xp.getScreenBoundsGlobal()
        except Exception:
            left, right, bottom = 0, 1024, 0
        cx = (left + right) // 2
        b = bottom + BOTTOM_MARGIN
        return cx - WIDTH // 2, b + HEIGHT, cx + WIDTH // 2, b

    def _start_imgui(self, left: int, top: int, right: int, bottom: int) -> None:
        if self._imgui_modules is not None:
            xp_imgui, imgui = self._imgui_modules
        else:  # both ship with XPPython3; absent anywhere else
            imgui = importlib.import_module("imgui")
            xp_imgui = importlib.import_module("XPPython3.xp_imgui")
        self.window = xp_imgui.Window(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            visible=0,
            draw=self._draw_imgui,
            decoration=self.xp.WindowDecorationRoundRectangle,
            layer=self.xp.WindowLayerFloatingWindows,
        )
        self._imgui = imgui
        self.window_id = self.window.windowID
        try:
            imgui.set_current_context(self.window.imgui_context)
            imgui.get_io().font_global_scale = IMGUI_FONT_SCALE
            style = imgui.get_style()
            style.window_padding = (14, 10)
            style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.05, 0.07, 0.10, 0.90)
        except Exception as exc:  # cosmetic only
            self.log(f"overlay: style not applied ({exc})")
        self.backend = "imgui"

    def update(self) -> None:
        """Show/hide to match the state. Call from the flight loop."""
        if self.window_id is None:
            return
        want = self.state.visible
        if want != self._shown:
            self.xp.setWindowIsVisible(self.window_id, 1 if want else 0)
            self._shown = want

    def stop(self) -> None:
        try:
            if self.window is not None:
                self.window.delete()
            elif self.window_id is not None:
                self.xp.destroyWindow(self.window_id)
        except Exception as exc:
            self.log(f"overlay: stop failed ({exc})")
        self.window = self.window_id = None

    # ------------------------------------------------------------------
    # Draw callbacks (main thread, every frame while visible)
    # ------------------------------------------------------------------
    def _timed(self, draw: Any) -> None:
        t = time.perf_counter()
        try:
            draw(self.state.lines())
        except Exception as exc:
            self.draw_errors += 1
            if self.draw_errors <= 3:
                self.log(f"overlay: draw failed ({type(exc).__name__}: {exc})")
        self.draw_ms_max = max(self.draw_ms_max, (time.perf_counter() - t) * 1000)

    def _draw_imgui(self, window_id: Any, refcon: Any) -> None:
        self._timed(self._imgui_lines)

    def _imgui_lines(self, lines: list[Line]) -> None:
        imgui = self._imgui
        width = imgui.get_window_width()
        badge_col = 96 * IMGUI_FONT_SCALE
        for line in lines:
            r, g, b = TONES[line.tone]
            if line.badge:
                imgui.text_colored(line.badge, r, g, b, 1.0)
                imgui.same_line(position=badge_col)
            imgui.text_colored(line.text, *text_colour(line.tone), 1.0)
            if line.detail:
                w = imgui.calc_text_size(line.detail).x
                imgui.same_line(position=max(badge_col, width - w - 16))
                mr, mg, mb = TONES["muted"]
                imgui.text_colored(line.detail, mr, mg, mb, 1.0)

    def _draw_native(self, window_id: Any, refcon: Any) -> None:
        self._timed(lambda lines: self._native_lines(window_id, lines))

    def _native_lines(self, window_id: Any, lines: list[Line]) -> None:
        xp = self.xp
        geom = xp.getWindowGeometry(window_id)
        if not geom:
            return
        left, top, right, bottom = geom
        xp.drawTranslucentDarkBox(left, top, right, bottom)
        font = xp.Font_Proportional
        y = top - LINE_HEIGHT
        for line in lines:
            colour = list(TONES[line.tone])
            if line.badge:
                xp.drawString(colour, left + 12, y, line.badge, None, font)
            xp.drawString(
                list(text_colour(line.tone)), left + BADGE_COLUMN, y, line.text, None, font
            )
            if line.detail:
                w = int(xp.measureString(font, line.detail))
                xp.drawString(list(TONES["muted"]), right - 12 - w, y, line.detail, None, font)
            y -= LINE_HEIGHT
