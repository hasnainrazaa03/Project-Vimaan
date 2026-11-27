"""In-sim overlay: a panel at the top left of the screen showing what Vimaan
heard (live while you talk), what it understood and how sure it was, what
happened, and the last few commands, fading as they age (phase O.5, design
D "avionics, top left").

Draws ``OverlayState.panel()`` in an X-Plane window with Dear ImGui (bundled
with XPPython3 as ``xp_imgui`` + ``imgui``): shapes, colours, and B612 Mono,
the cockpit-display font X-Plane ships in Resources/fonts (nothing bundled
here). Falls back to X-Plane's own text drawing of ``OverlayState.lines()``
if ImGui can't load. The window is a normal X-Plane floating window, so it
can be dragged or popped out; it is hidden when there's nothing to show
(auto mode), and grows downward to fit.

Everything here runs on X-Plane's main thread inside draw callbacks, so the
draw functions never raise (an exception would kill the ImGui window for the
session) and their CPU time is measured (``draw_ms_max``).
"""

from __future__ import annotations

import importlib
import math
import os
import time
from collections.abc import Callable
from typing import Any

from vimaan_nlu.overlay_state import Line, OverlayState, Panel, Row

WIDTH = 580
MIN_HEIGHT = 40
MAX_HEIGHT = 600  # the panel grows to fit, up to this
LEFT_MARGIN = 30  # px from the left of the main screen
TOP_MARGIN = 60  # px below its top: clear of X-Plane's menu bar
LINE_HEIGHT = 26  # native renderer
BADGE_COLUMN = 104  # native renderer: px reserved for the badge
PAD_Y = 12  # native renderer, top and bottom
IMGUI_FONT_SCALE = 1.45  # ImGui's own font (no TTF found): scaled up

# Design D: sizes in px (X-Plane boxels).
FONT_FILES = ("B612Mono-Regular.ttf", "DejaVuSansMono.ttf")  # in X-Plane's Resources/fonts
BODY_PX, SMALL_PX = 19.0, 15.0  # ImGui sizes by full height: B612 is 1.215 em
FONT_OVERSAMPLE = 2  # rasterised at twice the size, drawn at half: sharp on Retina
PAD_X, PAD_TOP, PAD_BOTTOM = 12.0, 8.0, 9.0
TAG_PAD_X, TAG_PAD_Y, TAG_GAP = 6.0, 1.0, 8.0
HEADER_GAP = 7.0  # under the header
ROW_GAP = 5.0
WRAP_GAP = 2.0
LABEL_GAP = 14.0
BAR_H, CONF_H, TICK_H = 4.0, 6.0, 12.0
BUTTON_GAP = 12.0
HISTORY_GAP = 3.0
SECTION_GAP = 6.0  # above and below the history's rule
PULSE_HZ = 6.0

RGB = tuple[float, float, float]


def _hex(code: str) -> RGB:
    code = code.lstrip("#")
    r, g, b = (int(code[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b)


TONES: dict[str, RGB] = {
    "live": _hex("#FF6B66"),
    "error": _hex("#FF6B66"),
    "ok": _hex("#3FD07A"),
    "warn": _hex("#F5B841"),
    "accent": _hex("#5ED3F3"),
    "heard": _hex("#5ED3F3"),
    "text": _hex("#F2F2F2"),
    "muted": _hex("#8A929B"),
}
FRAME = _hex("#2A2F35")
TRACK = _hex("#1C2126")
INK_ON_FILL = _hex("#050607")  # a filled tag's text
PANEL_BG = (5 / 255, 6 / 255, 7 / 255, 0.9)
HISTORY_INK = (_hex("#C7CDD3"), _hex("#A3ABB3"), _hex("#858D96"))  # newest first
HISTORY_TAG_ALPHA = (1.0, 0.8, 0.65)


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


def fit_text(text: str, width: float, measure: Callable[[str], float]) -> str:
    """``text`` cut to ``width`` with ".." when it doesn't fit (one line)."""
    if measure(text) <= width:
        return text
    while text and measure(text + "..") > width:
        text = text[:-1]
    return (text.rstrip() + "..") if text else ""


def text_colour(tone: str) -> RGB:
    """The native renderer's line text: plain white for good/neutral news,
    the tone colour when it needs attention."""
    if tone in ("ok", "accent", "text"):
        return TONES["text"]
    return TONES.get(tone, TONES["text"])


class Overlay:
    def __init__(
        self,
        xp: Any,
        state: OverlayState,
        log: Any = print,
        *,
        prefer_imgui: bool = True,
        imgui_modules: tuple[Any, Any] | None = None,
        on_mode: Callable[[str], None] | None = None,
        font_dir: str | None = None,
    ) -> None:
        self.xp = xp
        self.state = state
        self.log = log
        self.prefer_imgui = prefer_imgui
        self._imgui_modules = imgui_modules  # (xp_imgui, imgui) — injectable for tests
        self.on_mode = on_mode  # PIN / OFF clicked: the new mode
        self.font_dir = font_dir  # default: X-Plane's Resources/fonts
        self.backend: str | None = None  # "imgui" | "native"
        self.font: str | None = None  # the TTF loaded (None: ImGui's own)
        self.window: Any = None  # xp_imgui.Window (imgui backend)
        self.window_id: Any = None
        self._shown = False
        self._fonts: tuple[Any, Any] | None = None  # (body, small)
        self._buttons: dict[str, tuple[float, float, float, float]] = {}
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
        xp.setWindowResizingLimits(self.window_id, 380, MIN_HEIGHT, 1600, MAX_HEIGHT)
        return self.backend

    def _initial_geometry(self) -> tuple[int, int, int, int]:
        try:
            left, top, right, _bottom = self.xp.getScreenBoundsGlobal()
        except Exception:
            left, top, right = 0, 768, 1024
        x, t = left + LEFT_MARGIN, top - TOP_MARGIN
        return x, t, x + min(WIDTH, right - left - 2 * LEFT_MARGIN), t - MIN_HEIGHT

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
            self._fonts = self._load_fonts(imgui)
            if self._fonts is None:
                imgui.get_io().font_global_scale = IMGUI_FONT_SCALE
            style = imgui.get_style()
            style.window_padding = (PAD_X, PAD_TOP)
            style.window_border_size = 0.0
            style.colors[imgui.COLOR_WINDOW_BACKGROUND] = PANEL_BG
        except Exception as exc:  # cosmetic only
            self.log(f"overlay: style not applied ({exc})")
        self.backend = "imgui"

    def _font_path(self) -> str | None:
        folder = self.font_dir
        if folder is None:
            try:
                folder = os.path.join(self.xp.getSystemPath(), "Resources", "fonts")
            except Exception:
                return None
        for name in FONT_FILES:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
        return None

    def _load_fonts(self, imgui: Any) -> tuple[Any, Any] | None:
        """B612 Mono at two sizes, into the window's font atlas; None (ImGui's
        own font) if there is no TTF or it won't load."""
        path = self._font_path()
        if path is None:
            self.log("overlay: no TTF font found; using ImGui's own")
            return None
        io = imgui.get_io()
        try:
            body = io.fonts.add_font_from_file_ttf(path, BODY_PX * FONT_OVERSAMPLE)
            small = io.fonts.add_font_from_file_ttf(path, SMALL_PX * FONT_OVERSAMPLE)
            self.window.renderer.refresh_font_texture()  # the atlas, rebuilt with them
            io.font_global_scale = 1.0 / FONT_OVERSAMPLE
        except Exception as exc:
            self.log(f"overlay: font {os.path.basename(path)} not loaded ({exc})")
            return None
        self.font = os.path.basename(path)
        return body, small

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
        the top edge where it is: the panel hangs from the top of the
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
            height = max(MIN_HEIGHT, min(MAX_HEIGHT, wanted))
            if abs((top - bottom) - height) > 2:
                xp.setWindowGeometry(self.window_id, left, top, right, top - height)
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

    def _set_mode(self, mode: str) -> None:
        if self.on_mode is not None:
            self.on_mode(mode)
        else:
            self.state.mode = mode

    # ------------------------------------------------------------------
    # Draw callbacks (main thread, every frame while visible)
    # ------------------------------------------------------------------
    def _timed(self, draw: Callable[[], None]) -> None:
        t = time.perf_counter()
        try:
            draw()
        except Exception as exc:
            self.draw_errors += 1
            if self.draw_errors <= 3:
                self.log(f"overlay: draw failed ({type(exc).__name__}: {exc})")
        self.draw_ms_max = max(self.draw_ms_max, (time.perf_counter() - t) * 1000)

    def _draw_imgui(self, window_id: Any, refcon: Any) -> None:
        self._timed(lambda: self._imgui_panel(self.state.panel()))

    def _push(self, font: Any) -> None:
        if font is not None:
            self._imgui.push_font(font)

    def _pop(self, font: Any) -> None:
        if font is not None:
            self._imgui.pop_font()

    def _imgui_panel(self, panel: Panel) -> None:
        imgui = self._imgui
        draw = imgui.get_window_draw_list()
        u32 = imgui.get_color_u32_rgba

        def col(rgb: RGB, alpha: float = 1.0) -> Any:
            return u32(rgb[0], rgb[1], rgb[2], alpha)

        def measure(text: str) -> float:
            return float(imgui.calc_text_size(text).x)

        body, small = self._fonts or (None, None)
        x0, y0 = imgui.get_window_position()
        w, h = imgui.get_window_width(), imgui.get_window_height()
        frame = TONES[panel.alert] if panel.alert else FRAME
        draw.add_rect(x0 + 0.5, y0 + 0.5, x0 + w - 0.5, y0 + h - 0.5, col(frame), 4.0, 0, 1.0)
        left, right = x0 + PAD_X, x0 + w - PAD_X
        y = y0 + PAD_TOP

        # Header: tags, the title, then (from the right) OFF, PIN, the meta.
        self._push(small)
        tag_h = imgui.get_text_line_height() + 2 * TAG_PAD_Y
        x = left
        pulse = 0.6 + 0.4 * math.sin(imgui.get_time() * PULSE_HZ)
        for tag in panel.tags:
            rgb = TONES.get(tag.tone, TONES["muted"])
            tw = measure(tag.text) + 2 * TAG_PAD_X
            if tag.filled:
                alpha = pulse if tag.tone == "live" else 1.0
                draw.add_rect_filled(x, y, x + tw, y + tag_h, col(rgb, alpha))
                draw.add_text(x + TAG_PAD_X, y + TAG_PAD_Y, col(INK_ON_FILL), tag.text)
            else:
                draw.add_rect(x, y, x + tw, y + tag_h, col(rgb), 0.0, 0, 1.0)
                draw.add_text(x + TAG_PAD_X, y + TAG_PAD_Y, col(rgb), tag.text)
            x += tw + TAG_GAP
        rx = right
        self._buttons = {}
        for name in ("OFF", "PIN"):
            bw = measure(name)
            on = name == "PIN" and panel.pinned
            draw.add_text(rx - bw, y + TAG_PAD_Y, col(TONES["accent" if on else "muted"]), name)
            self._buttons[name] = (rx - bw - 5, y - 3, rx + 5, y + tag_h + 3)
            rx -= bw + BUTTON_GAP
        if panel.meta:
            mw = measure(panel.meta)
            draw.add_text(rx - mw, y + TAG_PAD_Y, col(TONES[panel.meta_tone]), panel.meta)
            rx -= mw + BUTTON_GAP
        if panel.title:
            title = fit_text(panel.title, rx - x, measure)
            draw.add_text(x, y + TAG_PAD_Y, col(TONES[panel.title_tone]), title)
        self._pop(small)
        y += tag_h + HEADER_GAP

        # Rows.
        self._push(body)
        line_h = imgui.get_text_line_height()
        label_w = measure("ACTION") + LABEL_GAP
        text_x = left + label_w
        for row in panel.rows:
            y = self._row(draw, col, measure, row, panel, left, text_x, right, y, line_h, small)
        self._pop(body)

        # History: newest first, dimmer down the list, fading with age.
        if panel.history:
            y += SECTION_GAP - ROW_GAP
            draw.add_line(x0 + 1, y, x0 + w - 1, y, col(FRAME), 1.0)
            y += SECTION_GAP
            self._push(small)
            small_h = imgui.get_text_line_height()
            age_w = measure("-0:00") + 10
            tag_w = measure("DONE") + 10
            for i, past in enumerate(panel.history):
                n = min(i, len(HISTORY_INK) - 1)
                a = past.alpha
                draw.add_text(left, y, col(TONES["muted"], a * HISTORY_TAG_ALPHA[n]), past.age)
                tone = TONES.get(past.tone, TONES["muted"])
                draw.add_text(left + age_w, y, col(tone, a * HISTORY_TAG_ALPHA[n]), past.tag)
                text = fit_text(past.text, right - left - age_w - tag_w, measure)
                draw.add_text(left + age_w + tag_w, y, col(HISTORY_INK[n], a), text)
                y += small_h + HISTORY_GAP
            self._pop(small)
            y -= HISTORY_GAP - ROW_GAP
        self._wanted_height = int(math.ceil(y - ROW_GAP - y0 + PAD_BOTTOM))
        self._clicks()

    def _row(
        self,
        draw: Any,
        col: Callable[..., Any],
        measure: Callable[[str], float],
        row: Row,
        panel: Panel,
        left: float,
        text_x: float,
        right: float,
        y: float,
        line_h: float,
        small: Any,
    ) -> float:
        """Draw one row from ``y``; returns the y of the next."""
        rgb = TONES.get(row.tone, TONES["text"])
        muted = col(TONES["muted"])
        if row.label and row.kind not in ("item", "offer"):
            draw.add_text(left, y, muted, row.label)
        if row.kind == "conf":
            detail_w = measure("100%")
            end = right - detail_w - 10
            mid = y + line_h / 2
            draw.add_rect_filled(text_x, mid - CONF_H / 2, end, mid + CONF_H / 2, col(TRACK))
            if row.value is not None:
                fill = text_x + (end - text_x) * min(1.0, max(0.0, row.value))
                draw.add_rect_filled(text_x, mid - CONF_H / 2, fill, mid + CONF_H / 2, col(rgb))
            tick = text_x + (end - text_x) * panel.floor
            ink = TONES["text"] if row.value is not None else TONES["muted"]
            draw.add_rect_filled(tick - 1, mid - TICK_H / 2, tick + 1, mid + TICK_H / 2, col(ink))
            dw = measure(row.detail)
            draw.add_text(
                right - dw, y, col(TONES.get(row.detail_tone, TONES["muted"])), row.detail
            )
            return y + line_h + ROW_GAP
        if row.kind == "bar":
            bar_y = y + 1
            draw.add_rect_filled(left, bar_y, right, bar_y + BAR_H, col(TRACK))
            fill = left + (right - left) * min(1.0, max(0.0, row.value or 0.0))
            if fill > left:
                draw.add_rect_filled(left, bar_y, fill, bar_y + BAR_H, col(rgb))
            return bar_y + BAR_H + ROW_GAP + 1
        if row.kind == "spinner":
            mid = y + line_h / 2
            draw.add_rect_filled(text_x, mid - BAR_H / 2, right, mid + BAR_H / 2, col(TRACK))
            span = right - text_x
            seg = 0.22 * span
            start = text_x + ((self._imgui.get_time() * 0.5) % 1.3 - 0.3) * span
            a, b = max(text_x, start), min(right, start + seg)
            if b > a:
                draw.add_rect_filled(a, mid - BAR_H / 2, b, mid + BAR_H / 2, col(rgb))
            return y + line_h + ROW_GAP
        if row.kind in ("item", "offer"):
            return self._item(draw, col, measure, row, left, right, y, line_h, small)
        if row.kind == "note":
            self._push(small)
            for i, part in enumerate(wrap_text(row.text, right - left, measure)):
                draw.add_text(left, y, col(rgb), part)
                y += self._imgui.get_text_line_height() + WRAP_GAP
            self._pop(small)
            return y - WRAP_GAP + ROW_GAP
        # Text: wrapped under itself, the detail at the right.
        start = text_x if row.label else left
        detail_w = measure(row.detail) + 12 if row.detail else 0.0
        parts = wrap_text(row.text, max(80.0, right - detail_w - start), measure)
        for i, part in enumerate(parts):
            draw.add_text(start, y + i * (line_h + WRAP_GAP), col(rgb), part)
        if row.detail:
            tone = TONES.get(row.detail_tone, TONES["muted"])
            draw.add_text(right - measure(row.detail), y, col(tone), row.detail)
        return y + len(parts) * line_h + (len(parts) - 1) * WRAP_GAP + ROW_GAP

    def _item(
        self,
        draw: Any,
        col: Callable[..., Any],
        measure: Callable[[str], float],
        row: Row,
        left: float,
        right: float,
        y: float,
        line_h: float,
        small: Any,
    ) -> float:
        """A checklist line: NAME ....... READING; the offered one boxed, with
        the offer under it."""
        rgb = TONES.get(row.tone, TONES["text"])
        offer = row.kind == "offer"
        top = y
        if offer:
            y += 3
            name_rgb = value_rgb = TONES["text"]
        else:
            name_rgb, value_rgb = rgb, TONES.get(row.detail_tone, rgb)
        value = row.detail or "--"
        vw = measure(value)
        name = fit_text(row.label, right - left - vw - 30, measure)
        nw = measure(name)
        draw.add_text(left, y, col(name_rgb), name)
        draw.add_text(right - vw, y, col(value_rgb), value)
        dot_w = measure(".") or 1.0
        gap = right - vw - left - nw - 16
        if gap > dot_w:
            leader = "." * int(gap / dot_w)
            draw.add_text(left + nw + 8, y, col(name_rgb, 0.4), leader)
        y += line_h
        if offer:
            self._push(small)
            small_h = self._imgui.get_text_line_height()
            for part in wrap_text(row.text, right - left, measure):
                y += WRAP_GAP
                draw.add_text(left, y, col(rgb), part)
                y += small_h
            self._pop(small)
            y += 3
            draw.add_rect(left - 6, top, right + 6, y, col(rgb), 0.0, 0, 1.0)
        return y + ROW_GAP

    def _clicks(self) -> None:
        """PIN and OFF, clicked: xp_imgui passes clicks in the window to ImGui."""
        imgui = self._imgui
        if not imgui.is_mouse_clicked(0):
            return
        mx, my = imgui.get_mouse_pos()
        for name, (x0, y0, x1, y1) in self._buttons.items():
            if x0 <= mx <= x1 and y0 <= my <= y1:
                if name == "OFF":
                    self._set_mode("off")
                else:
                    self._set_mode("auto" if self.state.mode == "pinned" else "pinned")
                return

    # ------------------------------------------------------------------
    def _draw_native(self, window_id: Any, refcon: Any) -> None:
        self._timed(lambda: self._native_lines(window_id, self.state.lines()))

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
            colour = list(TONES.get(line.tone, TONES["text"]))
            if line.badge:
                xp.drawString(colour, left + 12, y, line.badge, None, font)
            detail_w = measure(line.detail) + 18 if line.detail else 0.0
            avail = max(80.0, right - 12 - detail_w - (left + BADGE_COLUMN))
            first_y = y
            for part in wrap_text(line.text, avail, measure):
                if part:
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
