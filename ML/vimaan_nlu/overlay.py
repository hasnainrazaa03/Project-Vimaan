"""In-sim overlay: a panel at the bottom left of the screen showing what Vimaan
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
import math
import time
from collections.abc import Callable
from typing import Any

from vimaan_nlu.overlay_state import Line, OverlayState

WIDTH, HEIGHT = 660, 118
LEFT_MARGIN = 30  # px from the left of the main screen
BOTTOM_MARGIN = 60  # px above the bottom of the main screen
BADGE_COLUMN = 104  # px reserved for the status badge (native renderer)
LINE_HEIGHT = 26
IMGUI_FONT_SCALE = 1.45
WIDEST_BADGE = "NO RESPONSE"  # every pill is this wide, so text columns line up
PILL_ROUNDING = 5.0
LINE_GAP = 8  # px between rows
WRAP_GAP = 2  # px between the wrapped lines of one row
STRIPE_WIDTH = 4  # outcome-coloured stripe down the left edge
RIGHT_PAD = 16  # px kept clear at the right edge
PROGRESS_HEIGHT = 5  # px, the probe's progress bar
PAD_Y = 12  # ImGui window padding, top and bottom
MAX_HEIGHT = 600  # the panel grows to fit wrapped text, up to this


def wrap_text(text: str, width: float, measure: Callable[[str], float]) -> list[str]:
    """Greedy word wrap to ``width`` as ``measure`` (px) sees it. A single word
    wider than the line is split by characters, so nothing is ever cut off."""
    out: list[str] = []
    current = ""
    for word in text.strip().split(" "):  # inner double spaces kept: they separate fields
        candidate = f"{current} {word}" if current else word
        if measure(candidate) <= width:
            current = candidate
            continue
        if current:
            out.append(current)
            current = ""
        while measure(word) > width and len(word) > 1:
            cut = len(word) - 1
            while cut > 1 and measure(word[:cut]) > width:
                cut -= 1
            out.append(word[:cut])
            word = word[cut:]
        current = word
    if current or not out:
        out.append(current)
    return out


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
        self._wanted_height: int | None = None  # measured by the last draw
        self._resize_errors = 0

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
        xp.setWindowResizingLimits(self.window_id, 380, 70, 1600, MAX_HEIGHT)
        return self.backend

    def _initial_geometry(self) -> tuple[int, int, int, int]:
        try:
            left, _top, right, bottom = self.xp.getScreenBoundsGlobal()
        except Exception:
            left, right, bottom = 0, 1024, 0
        x, b = left + LEFT_MARGIN, bottom + BOTTOM_MARGIN
        return x, b + HEIGHT, x + min(WIDTH, right - left - 2 * LEFT_MARGIN), b

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
            style.window_padding = (16, 12)
            style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.04, 0.06, 0.09, 0.92)
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
        if want:
            self._fit_height()

    def _fit_height(self) -> None:
        """Grow (or shrink back) to the height the last frame needed, keeping
        the bottom edge where it is: the panel sits at the bottom of the
        screen. Left alone when popped out to its own OS window."""
        wanted = self._wanted_height
        if wanted is None:
            return
        xp = self.xp
        try:
            popped = getattr(xp, "windowIsPoppedOut", None)
            if popped is not None and popped(self.window_id):
                return
            left, top, right, bottom = xp.getWindowGeometry(self.window_id)
            height = max(HEIGHT, min(MAX_HEIGHT, wanted))
            if abs((top - bottom) - height) > 2:
                xp.setWindowGeometry(self.window_id, left, bottom + height, right, bottom)
        except Exception as exc:
            self._resize_errors += 1
            if self._resize_errors <= 3:
                self.log(f"overlay: resize failed ({type(exc).__name__}: {exc})")

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
        """Rows of [pill badge] text .......... detail, with a stripe down the
        left edge in the colour of the latest row (the outcome so far). Long
        text wraps under itself; the probe's title row has a progress bar."""
        imgui = self._imgui
        draw = imgui.get_window_draw_list()
        rgba = imgui.get_color_u32_rgba
        win_x, win_y = imgui.get_window_position()
        win_w, win_h = imgui.get_window_width(), imgui.get_window_height()
        if lines:
            draw.add_rect_filled(
                win_x,
                win_y,
                win_x + STRIPE_WIDTH,
                win_y + win_h,
                rgba(*TONES[lines[-1].tone], 0.95),
            )
        line_h = imgui.get_text_line_height()
        pill_w = imgui.calc_text_size(WIDEST_BADGE).x + 20
        muted = rgba(*TONES["muted"], 1.0)

        def measure(text: str) -> float:
            return float(imgui.calc_text_size(text).x)

        start_y = imgui.get_cursor_screen_pos()[1]
        bottom = start_y
        for line in lines:
            x, y = imgui.get_cursor_screen_pos()
            r, g, b = TONES[line.tone]
            if line.badge:
                draw.add_rect_filled(
                    x, y - 3, x + pill_w, y + line_h + 3, rgba(r, g, b, 0.18), PILL_ROUNDING
                )
                badge_x = x + (pill_w - imgui.calc_text_size(line.badge).x) / 2
                if line.tone == "live":  # pulsing record dot
                    pulse = 0.55 + 0.45 * math.sin(imgui.get_time() * 6.0)
                    draw.add_circle_filled(
                        x + 11, y + line_h / 2, line_h * 0.26, rgba(r, g, b, pulse)
                    )
                    badge_x += 7
                draw.add_text(badge_x, y, rgba(r, g, b, 1.0), line.badge)
            text_x = x + pill_w + 14
            detail_w = measure(line.detail) + 18 if line.detail else 0.0
            avail = max(80.0, win_x + win_w - RIGHT_PAD - detail_w - text_x)
            parts = wrap_text(line.text, avail, measure)
            for i, part in enumerate(parts):
                imgui.set_cursor_screen_pos((text_x, y + i * (line_h + WRAP_GAP)))
                imgui.text_colored(part, *text_colour(line.tone), 1.0)
            if line.detail:
                draw.add_text(
                    win_x + win_w - measure(line.detail) - RIGHT_PAD, y, muted, line.detail
                )
            bottom = y + len(parts) * line_h + (len(parts) - 1) * WRAP_GAP
            if line.progress is not None:
                bar_y, bar_end = bottom + 5, win_x + win_w - RIGHT_PAD
                track = rgba(*TONES["muted"], 0.22)
                draw.add_rect_filled(text_x, bar_y, bar_end, bar_y + PROGRESS_HEIGHT, track, 2.5)
                fill = text_x + (bar_end - text_x) * min(1.0, max(0.0, line.progress))
                if fill > text_x:
                    draw.add_rect_filled(
                        text_x, bar_y, fill, bar_y + PROGRESS_HEIGHT, rgba(r, g, b, 0.95), 2.5
                    )
                bottom = bar_y + PROGRESS_HEIGHT
            imgui.set_cursor_screen_pos((x, bottom + LINE_GAP))
        self._wanted_height = int(math.ceil(bottom - start_y + 2 * PAD_Y + 2))

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

        def measure(text: str) -> float:
            return float(xp.measureString(font, text))

        y = top - LINE_HEIGHT
        for line in lines:
            colour = list(TONES[line.tone])
            if line.badge:
                xp.drawString(colour, left + 12, y, line.badge, None, font)
            detail_w = measure(line.detail) + 18 if line.detail else 0.0
            avail = max(80.0, right - 12 - detail_w - (left + BADGE_COLUMN))
            first_y = y
            for part in wrap_text(line.text, avail, measure):
                xp.drawString(
                    list(text_colour(line.tone)), left + BADGE_COLUMN, y, part, None, font
                )
                y -= LINE_HEIGHT
            if line.detail:
                w = int(measure(line.detail))
                xp.drawString(
                    list(TONES["muted"]), right - 12 - w, first_y, line.detail, None, font
                )
        self._wanted_height = int(top - y - LINE_HEIGHT + 2 * PAD_Y)
