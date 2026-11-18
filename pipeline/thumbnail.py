"""Stage ``thumbnail`` — ``final/thumbnail.jpg`` (1280x720 JPEG under 2 MB).

The background comes from the same :class:`~pipeline.images.ImageProvider` as the
scene pictures (same cache, seed strategy and mock mode) with the prompt
``thumbnail.image_prompt`` + ``visual_style.global_prompt_suffix``.
``thumbnail.text_overlay`` is drawn here with Pillow — never by the image model.

Text placement rules:
* letters at least ~12 % of the frame height (measured on a capital "H");
* nothing outside the safe margins;
* nothing in the bottom-right corner, where YouTube draws the duration badge;
* outline + a soft dark backing so the text reads on any picture.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageOps

from pipeline.context import RunContext
from pipeline.diagrams import find_font_file, load_font
from pipeline.images import _join_sentences, generate_picture, make_image_provider
from pipeline.state import atomic_write_bytes, file_hash, stable_hash

THUMBNAIL_VERSION = 1
THUMB_SIZE = (1280, 720)          # YouTube's recommended custom-thumbnail size
MAX_BYTES = 2_000_000             # "under 2 MB" whichever way a megabyte is counted

SAFE_MARGIN_X = 0.05              # share of the frame width kept free on the left and right
SAFE_MARGIN_Y = 0.06              # share of the frame height kept free at the top and bottom
BADGE_ZONE_W = 0.24               # bottom-right corner reserved for YouTube's duration badge
BADGE_ZONE_H = 0.20
MIN_LETTER_RATIO = 0.12           # letters (capital height) at least this share of the frame height
MAX_LETTER_RATIO = 0.18           # ...and no bigger than this: the picture has to stay visible
MAX_BLOCK_HEIGHT = 0.52           # the whole text block takes at most this share of the frame height
MAX_LINES = 3
LINE_PITCH = 1.06                 # distance between baselines, in font sizes
STROKE_RATIO = 0.07               # outline width, in font sizes
JPEG_QUALITIES = (95, 92, 88, 84, 80, 75, 70, 65, 60, 50, 40, 30)

Box = tuple[int, int, int, int]   # left, top, right, bottom (right/bottom exclusive)


class ThumbnailError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextLayout:
    """Where the overlay text ends up. ``box`` includes the outline."""

    lines: tuple[str, ...]
    font_size: int
    letter_height: int             # height of a capital "H" in pixels
    stroke: int
    box: Box
    baselines: tuple[tuple[int, int], ...]   # (x, y) of each line's left baseline point

    def as_info(self) -> dict[str, Any]:
        return {"lines": list(self.lines), "font_size": self.font_size, "letter_height": self.letter_height,
                "stroke": self.stroke, "box": list(self.box)}


def safe_box(size: tuple[int, int] = THUMB_SIZE) -> Box:
    w, h = size
    mx, my = round(w * SAFE_MARGIN_X), round(h * SAFE_MARGIN_Y)
    return mx, my, w - mx, h - my


def badge_box(size: tuple[int, int] = THUMB_SIZE) -> Box:
    """The bottom-right corner where YouTube shows the video duration."""
    w, h = size
    return round(w * (1 - BADGE_ZONE_W)), round(h * (1 - BADGE_ZONE_H)), w, h


def boxes_intersect(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _line_splits(words: list[str], max_lines: int) -> Iterator[tuple[str, ...]]:
    """Every way to break the words into 1..max_lines consecutive lines."""
    n = len(words)

    def cut(start: int, lines_left: int) -> Iterator[list[str]]:
        if lines_left == 1:
            yield [" ".join(words[start:])]
            return
        for end in range(start + 1, n - lines_left + 2):
            for rest in cut(end, lines_left - 1):
                yield [" ".join(words[start:end]), *rest]

    for count in range(1, min(max_lines, n) + 1):
        for lines in cut(0, count):
            yield tuple(lines)


def _measure(lines: tuple[str, ...], font_path: str | None, font_size: int) -> tuple[Box, int, int, list[int]]:
    """Ink box (with outline) of the block when the first baseline starts at (0, 0)."""
    font = load_font(font_path, font_size)
    stroke = max(2, round(font_size * STROKE_RATIO))
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    pitch = round(font_size * LINE_PITCH)
    left = top = 10 ** 9
    right = bottom = -10 ** 9
    offsets = []
    for index, line in enumerate(lines):
        y = index * pitch
        offsets.append(y)
        l, t, r, b = draw.textbbox((0, y), line, font=font, anchor="ls", stroke_width=stroke)
        left, top, right, bottom = min(left, l), min(top, t), max(right, r), max(bottom, b)
    cap = font.getbbox("H", anchor="ls")
    return (left, top, right, bottom), cap[3] - cap[1], stroke, offsets


def layout_text(text: str, font_path: str | None, size: tuple[int, int] = THUMB_SIZE) -> TextLayout | None:
    """Biggest readable placement of ``text``: bottom-left, inside the safe box, clear of the badge corner.

    Two candidate areas are tried — the strip left of the badge corner and the area above it — with
    every line break of the words; the largest letters win (fewer lines on a tie).
    """
    words = text.split()
    if not words:
        return None
    w, h = size
    sl, st, sr, sb = safe_box(size)
    bl, bt, _, _ = badge_box(size)
    gap = round(h * 0.02)
    regions: list[Box] = [(sl, st, bl - gap, sb),      # left of the badge corner, down to the safe bottom
                          (sl, st, sr, bt - gap)]      # full safe width, above the badge corner
    max_block_h = h * MAX_BLOCK_HEIGHT
    max_letter = h * MAX_LETTER_RATIO

    best: tuple[tuple[int, int, int], TextLayout] | None = None
    for lines in _line_splits(words, MAX_LINES):
        for region_index, (rl, rt, rr, rb) in enumerate(regions):
            lo, hi, found = 8, h, None
            while lo <= hi:
                mid = (lo + hi) // 2
                box, letter, stroke, offsets = _measure(lines, font_path, mid)
                fits = (box[2] - box[0] <= rr - rl and box[3] - box[1] <= min(rb - rt, max_block_h)
                        and letter <= max_letter)
                if fits:
                    found, lo = (mid, box, letter, stroke, offsets), mid + 1
                else:
                    hi = mid - 1
            if found is None:
                continue
            font_size, box, letter, stroke, offsets = found
            dx, dy = rl - box[0], rb - box[3]              # anchor the block to the region's bottom-left
            placed = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
            layout = TextLayout(lines=lines, font_size=font_size, letter_height=letter, stroke=stroke, box=placed,
                                baselines=tuple((dx, dy + off) for off in offsets))
            score = (letter, -len(lines), -region_index)
            if best is None or score > best[0]:
                best = (score, layout)
    return best[1] if best else None


def _is_dark(rgb: tuple[int, ...]) -> bool:
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 90


def compose_thumbnail(background: Image.Image, text: str, color: str | None, font_path: str | None,
                      size: tuple[int, int] = THUMB_SIZE) -> tuple[Image.Image, TextLayout | None]:
    """Fit the background to the thumbnail frame and draw the overlay text on it."""
    image = ImageOps.fit(background.convert("RGB"), size, Image.LANCZOS, centering=(0.5, 0.5))
    layout = layout_text(text, font_path, size)
    if layout is None:
        return image, None
    fill = ImageColor.getrgb(color or "#FFFFFF")[:3]
    # Light text gets a black outline on a dark backing; dark text gets the opposite.
    outline, backing = ((255, 255, 255), (255, 255, 255)) if _is_dark(fill) else ((0, 0, 0), (0, 0, 0))

    # Backing: a soft-edged patch behind the block, so the text also reads on a busy or bright picture.
    pad = round(layout.font_size * 0.35)
    mask = Image.new("L", size, 0)
    l, t, r, b = layout.box
    ImageDraw.Draw(mask).rounded_rectangle([l - pad, t - pad, r + pad, b + pad], radius=pad, fill=105)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(pad * 0.9, 1)))
    image.paste(Image.new("RGB", size, backing), (0, 0), mask)

    draw = ImageDraw.Draw(image)
    font = load_font(font_path, layout.font_size)
    for line, (x, y) in zip(layout.lines, layout.baselines):
        draw.text((x, y), line, font=font, fill=fill, anchor="ls", stroke_width=layout.stroke, stroke_fill=outline)
    return image, layout


def encode_jpeg(image: Image.Image, max_bytes: int = MAX_BYTES) -> tuple[bytes, int]:
    """Highest JPEG quality that stays under ``max_bytes``. Returns (data, quality).

    4:4:4 chroma first (keeps coloured text edges clean); 4:2:0 only if nothing else fits.
    """
    rgb = image.convert("RGB")
    for subsampling in (0, 2):
        for quality in JPEG_QUALITIES:
            buf = io.BytesIO()
            try:
                rgb.save(buf, "JPEG", quality=quality, subsampling=subsampling, optimize=True)
            except OSError:
                # libjpeg's optimising pass needs the whole file in one buffer; Pillow sizes that buffer from the
                # pixel count, and a very detailed picture at high quality overflows it. Plain Huffman tables then.
                buf = io.BytesIO()
                rgb.save(buf, "JPEG", quality=quality, subsampling=subsampling)
            if buf.tell() < max_bytes:
                return buf.getvalue(), quality
    raise ThumbnailError(f"cannot fit the thumbnail under {max_bytes} bytes even at JPEG quality {JPEG_QUALITIES[-1]}")


def thumbnail_prompt(ctx: RunContext) -> tuple[str, str]:
    """(prompt, negative_prompt): ``thumbnail.image_prompt`` + the global style suffix."""
    style = ctx.script.production.visual_style
    return _join_sentences([ctx.script.thumbnail.image_prompt, style.global_prompt_suffix]), style.negative_prompt.strip()


def _font_identity(font_path: str | None) -> str:
    if not font_path or not Path(font_path).is_file():
        return "default"
    return f"{Path(font_path).name}:{file_hash(Path(font_path))}"


def run_thumbnail_stage(ctx: RunContext) -> None:
    thumb = ctx.script.thumbnail
    record = ctx.state.thumbnail

    # 1. background picture — same provider, cache, seed strategy and mock mode as the scenes
    provider = make_image_provider(ctx)
    prompt, negative = thumbnail_prompt(ctx)
    record.image = generate_picture(ctx, provider, label="thumbnail", prompt=prompt, negative=negative, section=None,
                                    out_path=ctx.paths.images / "thumbnail_bg.png", previous=record.image)
    ctx.save()

    # 2. text overlay + JPEG
    font_path = find_font_file(ctx.settings.assets_dir, ctx.script.production.subtitles.font)
    out = ctx.paths.final / "thumbnail.jpg"
    thumb_hash = stable_hash({"v": THUMBNAIL_VERSION, "background": record.image.hash, "text": thumb.text_overlay,
                              "color": thumb.text_overlay_color, "font": _font_identity(font_path),
                              "size": THUMB_SIZE, "max_bytes": MAX_BYTES})
    if record.hash == thumb_hash and out.is_file() and not ctx.force:
        ctx.log.info("thumbnail: %s up to date", ctx.settings.rel(out))
        return
    with Image.open(ctx.settings.abs(record.image.path)) as background:
        image, layout = compose_thumbnail(background, thumb.text_overlay, thumb.text_overlay_color, font_path)
    data, quality = encode_jpeg(image)
    atomic_write_bytes(out, data)

    if layout is None:
        ctx.log.warning("thumbnail: text_overlay is empty — the thumbnail has no text")
    elif layout.letter_height < round(THUMB_SIZE[1] * MIN_LETTER_RATIO):
        ctx.log.warning("thumbnail: '%s' is too long for %d%%-high letters — drawn at %.1f%% of the frame height; "
                        "shorten thumbnail.text_overlay", thumb.text_overlay, int(MIN_LETTER_RATIO * 100),
                        100 * layout.letter_height / THUMB_SIZE[1])
    record.hash, record.path = thumb_hash, ctx.settings.rel(out)
    record.bytes, record.jpeg_quality = len(data), quality
    record.info = {"size": list(THUMB_SIZE), "font": Path(font_path).name if font_path else "default",
                   **(layout.as_info() if layout else {})}
    ctx.save()
    ctx.log.info("thumbnail: %s — %dx%d, %.0f KB, JPEG quality %d%s", ctx.settings.rel(out), THUMB_SIZE[0],
                 THUMB_SIZE[1], len(data) / 1000, quality,
                 f", text {len(layout.lines)} line(s), letters {layout.letter_height} px" if layout else "")
