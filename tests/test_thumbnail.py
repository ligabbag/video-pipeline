"""Stage ``thumbnail``: size, JPEG weight, text placement, cache/seed behaviour, state accounting."""
from __future__ import annotations

import io
import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageChops

from pipeline import images as images_mod
from pipeline.cli import main
from pipeline.diagrams import find_font_file
from pipeline.images import ImageProvider, ImageRequest
from pipeline.thumbnail import (BADGE_ZONE_H, BADGE_ZONE_W, MAX_BYTES, MIN_LETTER_RATIO, THUMB_SIZE, badge_box,
                                boxes_intersect, compose_thumbnail, encode_jpeg, layout_text, run_thumbnail_stage,
                                safe_box, thumbnail_prompt)
from tests.conftest import EXAMPLE, make_context

VIDEO_ID = "tech-0001-knight-capital"
FONT = find_font_file(None, None)
TEXTS = ["$460M IN 45 MIN", "A", "97 emails ignored", "ONE TWO THREE FOUR FIVE", "gone by Friday, okay?",
         "ОШИБКА НА $460 МЛН", "WHY?", "8 SERVERS 1 OLD FLAG"]


def inside(box: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return box[0] >= outer[0] and box[1] >= outer[1] and box[2] <= outer[2] and box[3] <= outer[3]


Span = tuple[int, int]


def colour_mask(image: Image.Image, red: Span, green: Span, blue: Span) -> Image.Image:
    """White where every channel lies inside its (lo, hi) span, black elsewhere. Pillow only — no numpy."""
    mask: Image.Image | None = None
    for band, (lo, hi) in zip(image.convert("RGB").split(), (red, green, blue)):
        hit = band.point([255 if lo <= value <= hi else 0 for value in range(256)])
        mask = hit if mask is None else ImageChops.multiply(mask, hit)
    assert mask is not None
    return mask


def count(mask: Image.Image) -> int:
    return mask.histogram()[255]


@pytest.mark.parametrize("text", TEXTS)
def test_layout_keeps_text_big_inside_the_safe_box_and_off_the_duration_badge(text: str) -> None:
    layout = layout_text(text, FONT)
    assert layout is not None
    assert inside(layout.box, safe_box())                               # the box includes the outline
    assert not boxes_intersect(layout.box, badge_box())                 # bottom-right corner stays free
    assert layout.letter_height >= round(THUMB_SIZE[1] * MIN_LETTER_RATIO)   # letters >= ~12 % of the frame
    assert " ".join(layout.lines) == " ".join(text.split())             # nothing lost, nothing reordered


def test_badge_zone_is_the_bottom_right_corner() -> None:
    left, top, right, bottom = badge_box()
    assert (right, bottom) == THUMB_SIZE
    assert left == round(1280 * (1 - BADGE_ZONE_W)) and top == round(720 * (1 - BADGE_ZONE_H))


def test_text_too_long_for_big_letters_still_stays_in_the_safe_box() -> None:
    layout = layout_text("W" * 24, FONT)                                # one unbreakable 24-character word
    assert layout is not None and inside(layout.box, safe_box()) and not boxes_intersect(layout.box, badge_box())
    assert layout.letter_height < THUMB_SIZE[1] * MIN_LETTER_RATIO      # the stage logs a warning in this case
    assert layout_text("   ", FONT) is None and layout_text("", FONT) is None


def test_rendered_pixels_of_the_text_are_in_the_safe_zone() -> None:
    """Measure the actual picture, not the layout numbers: where do text-coloured pixels end up?"""
    background = Image.new("RGB", (2400, 1350), "#1E3A5F")
    image, layout = compose_thumbnail(background, "$460M IN 45 MIN", "#FFD400", FONT)
    assert image.size == THUMB_SIZE and layout is not None
    data, _ = encode_jpeg(image)
    decoded = Image.open(io.BytesIO(data)).convert("RGB")
    yellow = colour_mask(decoded, (216, 255), (173, 251), (0, 59))      # #FFD400 give or take JPEG noise
    assert count(yellow) > 5000                                         # the text really is there, in its colour
    ink = yellow.getbbox()
    assert ink is not None and inside(ink, safe_box()) and not boxes_intersect(ink, badge_box())
    assert inside(ink, layout.box)                                      # fill sits inside the reported outline box
    assert layout.letter_height >= THUMB_SIZE[1] * MIN_LETTER_RATIO     # one line of capitals is >= 12 % ...
    assert ink[3] - ink[1] >= len(layout.lines) * layout.letter_height  # ... and that is what was really drawn
    # the outline is there: pure black exists only around the letters (the backing merely darkens the blue)
    black = colour_mask(decoded, (0, 24), (0, 24), (0, 24))
    assert count(black) > 3000
    outline = black.getbbox()
    assert outline is not None and inside(outline, layout.box) and inside(outline, safe_box())
    assert not boxes_intersect(outline, badge_box())


def test_dark_text_gets_a_light_outline() -> None:
    image, layout = compose_thumbnail(Image.new("RGB", (1280, 720), "#202020"), "DARK", "#101010", FONT)
    assert layout is not None
    white = colour_mask(image.crop(layout.box), (241, 255), (241, 255), (241, 255))
    assert count(white) > 500                                           # white outline pixels around dark letters


def test_jpeg_quality_is_lowered_until_the_file_fits() -> None:
    noise = Image.frombytes("RGB", THUMB_SIZE, random.Random(7).randbytes(THUMB_SIZE[0] * THUMB_SIZE[1] * 3))
    data, quality = encode_jpeg(noise)
    assert len(data) < MAX_BYTES == 2_000_000
    small, low_quality = encode_jpeg(noise, max_bytes=250_000)          # same search, tighter budget
    assert len(small) < 250_000 and low_quality < quality
    assert Image.open(io.BytesIO(small)).size == (1280, 720)
    with pytest.raises(Exception, match="cannot fit"):
        encode_jpeg(noise, max_bytes=2_000)


def run_stage_via_cli(root: Path, *extra: str) -> int:
    script = root / "video_script.example.json"
    if not script.exists():
        shutil.copyfile(EXAMPLE, script)
    return main(["run", str(script), "--mock", "--stages", "thumbnail", "--root", str(root), *extra])


def test_stage_writes_a_1280x720_jpeg_under_2mb_and_accounts_for_it_in_state(tmp_path: Path) -> None:
    assert run_stage_via_cli(tmp_path) == 0
    out = tmp_path / "output" / VIDEO_ID
    thumb = out / "final" / "thumbnail.jpg"
    with Image.open(thumb) as image:
        assert image.format == "JPEG" and image.size == (1280, 720)
    assert 10_000 < thumb.stat().st_size < 2_000_000
    assert (out / "images" / "thumbnail_bg.png").is_file()

    state = json.loads((tmp_path / "state" / f"{VIDEO_ID}.json").read_text(encoding="utf-8"))
    record = state["thumbnail"]
    assert record["stage"]["status"] == "done" and record["stage"]["finished_at"]
    assert record["hash"] and record["path"].endswith("final/thumbnail.jpg") and record["bytes"] == thumb.stat().st_size
    assert record["image"]["provider"] == "mock" and record["image"]["hash"]
    assert record["info"]["letter_height"] >= 0.12 * 720 and " ".join(record["info"]["lines"]) == "$460M IN 45 MIN"
    assert inside(tuple(record["info"]["box"]), safe_box())
    assert set(state["stages"]) == {"tts", "images", "render", "shorts"}          # task-1 layout untouched
    assert state["stages"]["images"]["status"] == "pending"                       # only the asked stage ran
    assert state["costs"]["images_generated"] == 0                                # mock bills nothing

    # second run: nothing is redone; --force redraws
    mtime = thumb.stat().st_mtime_ns
    assert run_stage_via_cli(tmp_path) == 0
    assert thumb.stat().st_mtime_ns == mtime
    assert "thumbnail.jpg up to date" in (out / "pipeline.log").read_text(encoding="utf-8")
    assert run_stage_via_cli(tmp_path, "--force") == 0
    assert thumb.stat().st_mtime_ns > mtime


def test_thumbnail_is_part_of_the_default_run_right_after_images() -> None:
    from pipeline.cli import parse_stages
    assert parse_stages(None) == ["tts", "images", "thumbnail", "render", "shorts"]
    assert parse_stages("render,thumbnail") == ["thumbnail", "render"]


class RecordingProvider(ImageProvider):
    """A billable provider with seed support, to check prompt, seed, cache and cost accounting."""

    name = "recording"
    supports_seed = True

    def __init__(self) -> None:
        self.requests: list[ImageRequest] = []

    def cache_identity(self) -> dict[str, Any]:
        return {"model": "test"}

    def generate(self, request: ImageRequest) -> bytes:
        self.requests.append(request)
        buf = io.BytesIO()
        Image.new("RGB", (1600, 900), "#6b4f2a").save(buf, "PNG")
        return buf.getvalue()


def test_background_uses_the_image_provider_with_prompt_suffix_seed_and_cache(
        tmp_path: Path, example_doc: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RecordingProvider()
    monkeypatch.setattr("pipeline.thumbnail.make_image_provider", lambda ctx: provider)
    ctx = make_context(tmp_path, example_doc, mock=False)
    run_thumbnail_stage(ctx)

    request = provider.requests[0]
    style = example_doc["production"]["visual_style"]
    assert request.prompt.startswith("a single server rack glowing red")
    assert style["global_prompt_suffix"] in request.prompt                       # image_prompt + global suffix
    assert request.prompt.endswith(f"Avoid: {style['negative_prompt']}.")         # same folding as scene images
    assert "trading floor" not in request.prompt                                 # no scene entities
    assert thumbnail_prompt(ctx)[0] in request.prompt
    # fixed_per_video: the thumbnail shares the video's seed with the scenes
    assert request.seed == images_mod.seed_for(ctx.script, ctx.script.sections[0], "fixed_per_video")
    assert ctx.state.costs.images_generated == 1 and ctx.state.thumbnail.image.seed == request.seed
    with Image.open(tmp_path / "output" / VIDEO_ID / "final" / "thumbnail.jpg") as image:
        assert image.size == (1280, 720)

    # a fresh state (new machine, same cache/): the paid picture comes from cache, nothing is requested again
    (tmp_path / "state" / f"{VIDEO_ID}.json").unlink()
    ctx2 = make_context(tmp_path, example_doc, mock=False)
    run_thumbnail_stage(ctx2)
    assert len(provider.requests) == 1
    assert ctx2.state.costs.images_cache_hits == 1 and ctx2.state.thumbnail.image.from_cache is True

    # changing only the overlay text redraws the JPEG but does not touch the image API
    doc = json.loads(json.dumps(example_doc))
    doc["thumbnail"]["text_overlay"] = "ONE OLD FLAG"
    ctx3 = make_context(tmp_path, doc, mock=False)
    old_hash = ctx3.state.thumbnail.hash
    run_thumbnail_stage(ctx3)
    assert len(provider.requests) == 1 and ctx3.state.thumbnail.hash != old_hash
    assert ctx3.state.thumbnail.info["lines"] and "FLAG" in " ".join(ctx3.state.thumbnail.info["lines"])

    # fixed_per_section: the thumbnail is its own "section"
    assert images_mod.seed_for(ctx.script, None, "fixed_per_section") not in (
        None, images_mod.seed_for(ctx.script, ctx.script.sections[0], "fixed_per_section"))
    assert images_mod.seed_for(ctx.script, None, "random") is None
