"""Stage 4 — diagrams drawn by code (Pillow), 1920x1080.

Templates: ``big_number``, ``counter`` (animated), ``icon_grid``, ``timeline``,
``bar_compare``. Dark background, numbers in ``highlight_color``. On-screen text
comes *only* from ``params`` — nothing is invented here, which keeps the picture
language-neutral for dubbed tracks.

Shapes are drawn at 2x and downsampled (Pillow does not anti-alias primitives).
"""
from __future__ import annotations

import io
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont

DIAGRAM_VERSION = 2
SUPERSAMPLE = 2
ANIMATED_TEMPLATES = frozenset({"counter"})

_SYSTEM_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
)


@dataclass(frozen=True)
class DiagramStyle:
    background: str = "#0E1116"
    foreground: str = "#FFFFFF"
    highlight: str = "#FFD400"
    muted: str = "#3B4354"
    font_path: str | None = None


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_font_file(assets_dir: Path | None, preferred_name: str | None = None) -> str | None:
    """Pick a font: ``$PIPELINE_FONT_PATH``, then ``assets/fonts`` (name match first), then system fonts."""
    env = os.environ.get("PIPELINE_FONT_PATH")
    if env and Path(env).is_file():
        return env
    if assets_dir is not None and (assets_dir / "fonts").is_dir():
        fonts = sorted(p for p in (assets_dir / "fonts").iterdir() if p.suffix.lower() in (".ttf", ".otf", ".ttc"))
        if preferred_name:
            wanted = _norm(preferred_name)
            for font in fonts:
                stem = _norm(font.stem)
                if wanted and (wanted in stem or stem in wanted):
                    return str(font)
        if fonts:
            return str(fonts[0])
    for candidate in _SYSTEM_FONTS:
        if Path(candidate).is_file():
            return candidate
    return None


@lru_cache(maxsize=256)
def load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(int(size), 8)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: Any) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def fit_font(draw: ImageDraw.ImageDraw, texts: Sequence[str], font_path: str | None, max_w: float, max_h: float,
             start: int) -> Any:
    """Largest font (<= start) at which every text fits into max_w x max_h."""
    lo, hi, best = 8, max(int(start), 8), 8
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(font_path, mid)
        if all(text_size(draw, t, font)[0] <= max_w and text_size(draw, t, font)[1] <= max_h for t in texts if t):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return load_font(font_path, best)


def draw_centered(draw: ImageDraw.ImageDraw, text: str, cx: float, cy: float, font: Any, fill: str) -> None:
    draw.text((cx, cy), text, font=font, fill=fill, anchor="mm")


# --------------------------------------------------------------------------- #
# icons (vector primitives, no text)
# --------------------------------------------------------------------------- #
IconFn = Callable[[ImageDraw.ImageDraw, float, float, float, str, str], None]


def _icon_server(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    w, h = s * 0.62, s
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=s * 0.07, fill=color)
    unit = h / 4
    for row in range(4):
        y = cy - h / 2 + unit * row + unit / 2
        d.rounded_rectangle([cx - w * 0.36, y - unit * 0.2, cx + w * 0.12, y + unit * 0.2], radius=unit * 0.1, fill=bg)
        r = unit * 0.13
        d.ellipse([cx + w * 0.26 - r, y - r, cx + w * 0.26 + r, y + r], fill=bg)


def _icon_mail(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    w, h = s, s * 0.68
    box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    d.rounded_rectangle(box, radius=s * 0.06, fill=color)
    lw = max(int(s * 0.055), 2)
    d.line([(box[0] + lw, box[1] + lw), (cx, cy + h * 0.1), (box[2] - lw, box[1] + lw)], fill=bg, width=lw, joint="curve")


def _icon_user(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    r = s * 0.22
    d.ellipse([cx - r, cy - s * 0.46, cx + r, cy - s * 0.46 + 2 * r], fill=color)
    d.pieslice([cx - s * 0.42, cy + s * 0.02, cx + s * 0.42, cy + s * 0.86], start=180, end=360, fill=color)


def _icon_coin(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    r = s / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    d.ellipse([cx - r * 0.78, cy - r * 0.78, cx + r * 0.78, cy + r * 0.78], outline=bg, width=max(int(s * 0.04), 2))
    d.rounded_rectangle([cx - s * 0.05, cy - r * 0.5, cx + s * 0.05, cy + r * 0.5], radius=s * 0.02, fill=bg)


def _icon_clock(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    r = s / 2
    lw = max(int(s * 0.09), 2)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=lw)
    d.line([(cx, cy), (cx, cy - r * 0.6)], fill=color, width=lw)
    d.line([(cx, cy), (cx + r * 0.42, cy + r * 0.18)], fill=color, width=lw)


def _icon_warning(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    d.polygon([(cx, cy - s * 0.46), (cx + s * 0.5, cy + s * 0.42), (cx - s * 0.5, cy + s * 0.42)], fill=color)
    d.rounded_rectangle([cx - s * 0.04, cy - s * 0.16, cx + s * 0.04, cy + s * 0.14], radius=s * 0.02, fill=bg)
    r = s * 0.05
    d.ellipse([cx - r, cy + s * 0.22, cx + r, cy + s * 0.22 + 2 * r], fill=bg)


def _icon_file(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    w, h = s * 0.7, s
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=s * 0.06, fill=color)
    for row in range(3):
        y = cy - h * 0.2 + row * h * 0.2
        d.rounded_rectangle([cx - w * 0.3, y - h * 0.035, cx + w * 0.3, y + h * 0.035], radius=h * 0.02, fill=bg)


def _icon_default(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str, bg: str) -> None:
    d.rounded_rectangle([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], radius=s * 0.18, fill=color)


_ICONS: dict[str, IconFn] = {
    "server": _icon_server, "rack": _icon_server,
    "mail": _icon_mail, "email": _icon_mail, "envelope": _icon_mail,
    "user": _icon_user, "person": _icon_user, "people": _icon_user,
    "coin": _icon_coin, "dollar": _icon_coin, "money": _icon_coin, "cash": _icon_coin,
    "clock": _icon_clock, "time": _icon_clock,
    "warning": _icon_warning, "alert": _icon_warning, "error": _icon_warning,
    "file": _icon_file, "document": _icon_file, "doc": _icon_file,
}


def draw_icon(d: ImageDraw.ImageDraw, name: str | None, cx: float, cy: float, size: float, color: str, bg: str) -> None:
    _ICONS.get(str(name or "").lower(), _icon_default)(d, cx, cy, size, color, bg)


def _draw_mark(d: ImageDraw.ImageDraw, mark: str, cx: float, cy: float, s: float, color: str, font_path: str | None) -> None:
    lw = max(int(s * 0.12), 3)
    r = s * 0.42
    mark = mark.lower()
    if mark in ("x", "cross", "×"):
        d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=color, width=lw)
        d.line([(cx - r, cy + r), (cx + r, cy - r)], fill=color, width=lw)
    elif mark in ("check", "tick", "✓", "v"):
        d.line([(cx - r, cy), (cx - r * 0.25, cy + r * 0.7), (cx + r, cy - r * 0.7)], fill=color, width=lw, joint="curve")
    elif mark in ("!", "?"):
        draw_centered(d, mark, cx, cy, load_font(font_path, int(s * 0.9)), color)


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
def _blend(a: str, b: str, t: float) -> str:
    ca, cb = ImageColor.getrgb(a), ImageColor.getrgb(b)
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(ca[:3], cb[:3]))


def _big_number(d: ImageDraw.ImageDraw, w: int, h: int, p: dict[str, Any], st: DiagramStyle, progress: float) -> None:
    value = str(p.get("value", ""))
    sub = p.get("sub_value")
    icon = p.get("icon")
    font = fit_font(d, [value], st.font_path, w * 0.84, h * 0.36, int(h * 0.34))
    _, vh = text_size(d, value, font)
    cy = h * (0.45 if sub else 0.5) + (h * 0.07 if icon else 0)
    if icon:
        draw_icon(d, icon, w / 2, cy - vh / 2 - h * 0.15, h * 0.16, st.foreground, st.background)
    draw_centered(d, value, w / 2, cy, font, st.highlight)
    if sub:
        sub_font = fit_font(d, [str(sub)], st.font_path, w * 0.7, h * 0.14, max(int(font.size * 0.32), 40))
        draw_centered(d, str(sub), w / 2, cy + vh / 2 + h * 0.13, sub_font, st.foreground)


def format_counter(p: dict[str, Any], value: float) -> str:
    decimals = int(p.get("decimals", 0) or 0)
    number = f"{value:,.{decimals}f}" if decimals else f"{int(round(value)):,}"
    return f"{p.get('prefix', '')}{number}{p.get('suffix', '')}"


def counter_value(p: dict[str, Any], progress: float) -> float:
    """Value shown at ``progress`` in [0, 1] (ease-out so the count settles gently)."""
    start, end = float(p.get("from", 0)), float(p.get("to", 0))
    t = max(0.0, min(1.0, progress))
    eased = 1 - (1 - t) ** 3
    return start + (end - start) * eased


def _counter(d: ImageDraw.ImageDraw, w: int, h: int, p: dict[str, Any], st: DiagramStyle, progress: float) -> None:
    text = format_counter(p, counter_value(p, progress))
    widest = [format_counter(p, float(p.get("from", 0))), format_counter(p, float(p.get("to", 0)))]
    font = fit_font(d, widest, st.font_path, w * 0.8, h * 0.4, int(h * 0.38))  # constant size while counting
    icon = p.get("icon")
    cy = h * 0.57 if icon else h * 0.5
    if icon:
        draw_icon(d, icon, w / 2, h * 0.24, h * 0.2, st.foreground, st.background)
    draw_centered(d, text, w / 2, cy, font, st.highlight)


def _icon_grid(d: ImageDraw.ImageDraw, w: int, h: int, p: dict[str, Any], st: DiagramStyle, progress: float) -> None:
    total = max(1, min(int(p.get("total", 1) or 1), 400))
    highlighted = set(int(i) for i in p.get("highlight_indices", []) if isinstance(i, (int, float)))
    if isinstance(p.get("highlight_index"), (int, float)):
        highlighted.add(int(p["highlight_index"]))  # zero-based
    mark = p.get("highlight_mark")
    cols = int(p.get("columns") or (total if total <= 10 else math.ceil(math.sqrt(total * 16 / 9))))
    cols = max(1, min(cols, total))
    rows = math.ceil(total / cols)
    cell = min(w * 0.86 / cols, h * 0.74 / rows)
    size = cell * 0.7
    x0 = w / 2 - cell * cols / 2 + cell / 2
    y0 = h / 2 - cell * rows / 2 + cell / 2
    for index in range(total):
        row, col = divmod(index, cols)
        in_row = cols if row < rows - 1 else total - cols * (rows - 1)
        shift = (cols - in_row) * cell / 2  # centre a short last row
        cx, cy = x0 + col * cell + shift, y0 + row * cell
        if index in highlighted:
            if mark:
                draw_icon(d, p.get("icon"), cx, cy, size, st.muted, st.background)
                _draw_mark(d, str(mark), cx, cy, size, st.highlight, st.font_path)
            else:
                draw_icon(d, p.get("icon"), cx, cy, size, st.highlight, st.background)
        else:
            draw_icon(d, p.get("icon"), cx, cy, size, st.foreground, st.background)


def _points(p: dict[str, Any], key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in p.get(key) or []:
        out.append(dict(item) if isinstance(item, dict) else {"label": str(item)})
    return out


def _timeline(d: ImageDraw.ImageDraw, w: int, h: int, p: dict[str, Any], st: DiagramStyle, progress: float) -> None:
    points = _points(p, "points") or _points(p, "events") or _points(p, "labels")
    if not points:
        return
    hi = p.get("highlight_index")
    n = len(points)
    left, right, y = w * 0.12, w * 0.88, h * 0.5
    lw = max(int(h * 0.008), 2)
    d.line([(left, y), (right, y)], fill=st.muted, width=lw)
    step = (right - left) / max(n - 1, 1)
    labels = [str(pt.get("label", "")) for pt in points]
    font = fit_font(d, labels, st.font_path, max(step * 0.86, w * 0.1) if n > 1 else w * 0.6, h * 0.14, int(h * 0.11))
    for index, point in enumerate(points):
        x = left + step * index if n > 1 else w / 2
        on = bool(point.get("highlight")) or (isinstance(hi, (int, float)) and int(hi) == index)
        r = h * (0.032 if on else 0.02)
        d.ellipse([x - r, y - r, x + r, y + r], fill=st.highlight if on else st.foreground)
        if labels[index]:
            draw_centered(d, labels[index], x, y + h * 0.13, font, st.highlight if on else st.foreground)
        sub = point.get("sub_label")
        if sub:
            sub_font = fit_font(d, [str(sub)], st.font_path, max(step * 0.86, w * 0.1), h * 0.08, int(font.size * 0.55))
            draw_centered(d, str(sub), x, y - h * 0.11, sub_font, st.foreground)


def _bar_compare(d: ImageDraw.ImageDraw, w: int, h: int, p: dict[str, Any], st: DiagramStyle, progress: float) -> None:
    bars = _points(p, "bars") or _points(p, "items")
    if not bars:
        return
    values = []
    for bar in bars:
        try:
            values.append(max(float(bar.get("value", 0)), 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    top = max(values) or 1.0
    hi = p.get("highlight_index")
    hi_index = int(hi) if isinstance(hi, (int, float)) else values.index(max(values))
    n = len(bars)
    slot = w * 0.76 / n
    bar_w = min(slot * 0.62, w * 0.16)
    base_y, max_h = h * 0.84, h * 0.56
    labels = [str(bar.get("label", "")) for bar in bars]
    font = fit_font(d, labels, st.font_path, slot * 0.92, h * 0.12, int(h * 0.1))
    d.line([(w * 0.1, base_y), (w * 0.9, base_y)], fill=st.muted, width=max(int(h * 0.005), 2))
    for index, value in enumerate(values):
        cx = w * 0.12 + slot * index + slot / 2
        bar_h = max(max_h * value / top, h * 0.006)
        color = st.highlight if index == hi_index else _blend(st.foreground, st.background, 0.35)
        d.rounded_rectangle([cx - bar_w / 2, base_y - bar_h, cx + bar_w / 2, base_y], radius=bar_w * 0.06, fill=color)
        if labels[index]:
            draw_centered(d, labels[index], cx, base_y - bar_h - h * 0.07, font,
                          st.highlight if index == hi_index else st.foreground)


_TEMPLATES: dict[str, Callable[[ImageDraw.ImageDraw, int, int, dict[str, Any], DiagramStyle, float], None]] = {
    "big_number": _big_number,
    "counter": _counter,
    "icon_grid": _icon_grid,
    "timeline": _timeline,
    "bar_compare": _bar_compare,
}


def render_diagram(template: str, params: dict[str, Any], style: DiagramStyle,
                   size: tuple[int, int] = (1920, 1080), progress: float = 1.0) -> Image.Image:
    """Render one diagram frame. ``progress`` (0..1) only matters for animated templates."""
    if template not in _TEMPLATES:
        raise ValueError(f"unknown diagram template '{template}'")
    w, h = size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE
    image = Image.new("RGB", (w, h), style.background)
    _TEMPLATES[template](ImageDraw.Draw(image), w, h, params, style, progress)
    return image.resize(size, Image.LANCZOS)


def is_animated(template: str) -> bool:
    return template in ANIMATED_TEMPLATES


def render_counter_sequence(params: dict[str, Any], style: DiagramStyle, size: tuple[int, int], *,
                            total_frames: int, count_frames: int, out_dir: Path) -> None:
    """Write ``%05d.png`` frames: the counter runs for ``count_frames`` and then holds its final value."""
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded: dict[str, bytes] = {}
    for frame in range(total_frames):
        progress = min(1.0, frame / max(count_frames - 1, 1))
        key = format_counter(params, counter_value(params, progress))
        if key not in encoded:
            buf = io.BytesIO()
            render_diagram("counter", params, style, size, progress).save(buf, "PNG", compress_level=1)
            encoded[key] = buf.getvalue()
        (out_dir / f"{frame:05d}.png").write_bytes(encoded[key])
