"""Prompt assembly, image providers (HTTP faked), post-processing and diagram templates."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pipeline import images as images_mod
from pipeline.contract import Script
from pipeline.diagrams import DiagramStyle, counter_value, format_counter, render_counter_sequence, render_diagram
from pipeline.images import (FluxImageProvider, GptImageProvider, ImageError, ImageRequest, build_prompt,
                             fit_to_frame, prompt_for_provider, run_images_stage, seed_for)
from tests.conftest import FakeResponse, FakeSession, approved, make_context, small_doc


def png_bytes(size: tuple[int, int], color: str = "#336699") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


# ---- prompts / seeds --------------------------------------------------------------
def test_prompt_is_subject_plus_entities_plus_style(example_doc: dict[str, Any]) -> None:
    script = Script.model_validate(example_doc)
    scene = script.scene_by_id("s009")                       # refs: technician, server_row
    prompt, negative = build_prompt(script, scene)
    entities = script.entity_map()
    subject = scene.visual.image_prompt or ""
    suffix = script.production.visual_style.global_prompt_suffix
    assert prompt.startswith(subject)
    assert prompt.index(subject) < prompt.index(entities["technician"].visual_description) \
        < prompt.index(entities["server_row"].visual_description) < prompt.index(suffix)
    assert negative == script.production.visual_style.negative_prompt
    assert negative not in prompt                            # kept separate at this point


def test_negative_prompt_is_folded_in_only_when_unsupported() -> None:
    gpt = GptImageProvider("k")
    assert prompt_for_provider(gpt, "A cat.", "text, logo") == ("A cat. Avoid: text, logo.", None)

    class Native(GptImageProvider):
        supports_negative_prompt = True

    assert prompt_for_provider(Native("k"), "A cat.", "text, logo") == ("A cat.", "text, logo")


def test_seed_strategies(example_doc: dict[str, Any]) -> None:
    script = Script.model_validate(example_doc)
    sec1, sec2 = script.sections[0], script.sections[1]
    assert seed_for(script, sec1, "fixed_per_video") == seed_for(script, sec2, "fixed_per_video")
    assert seed_for(script, sec1, "fixed_per_section") != seed_for(script, sec2, "fixed_per_section")
    assert seed_for(script, sec1, "fixed_per_section") == seed_for(script, sec1, "fixed_per_section")
    assert seed_for(script, sec1, "random") is None


# ---- post-processing ----------------------------------------------------------------
@pytest.mark.parametrize("source,expected", [
    ((1536, 1024), (2400, 1350)),      # 3:2 -> cropped to 16:9, upscaled to 1.25x frame
    ((2560, 1440), (2560, 1440)),      # already 16:9 with headroom: untouched
    ((1024, 1024), (2400, 1350)),
    ((5120, 2880), (3840, 2160)),      # capped at 2x
])
def test_fit_to_frame(source: tuple[int, int], expected: tuple[int, int]) -> None:
    image = fit_to_frame(png_bytes(source), (1920, 1080))
    assert image.size == expected
    assert image.size[0] >= 1920 * 1.25 and abs(image.size[0] / image.size[1] - 16 / 9) < 0.01


# ---- providers ------------------------------------------------------------------------
def test_gpt_image_request_and_sizes() -> None:
    data = png_bytes((64, 36))
    session = FakeSession(lambda m, u, kw: FakeResponse(200, {"data": [{"b64_json": base64.b64encode(data).decode()}]}))
    provider = GptImageProvider("sk-openai-secret", session=session, sleep=lambda s: None)
    assert provider.generate(ImageRequest("s001", "A prompt.", None, None)) == data
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", "https://api.openai.com/v1/images/generations")
    assert kwargs["headers"]["Authorization"] == "Bearer sk-openai-secret"
    assert kwargs["json"] == {"model": "gpt-image-2", "prompt": "A prompt.", "size": "2560x1440",
                              "quality": "medium", "n": 1, "output_format": "png"}
    assert not provider.supports_seed and not provider.supports_negative_prompt
    assert GptImageProvider("k", model="gpt-image-2.5-flare").size == "2560x1440"
    assert GptImageProvider("k", model="gpt-image-1.5").size == "1536x1024"       # no custom sizes there
    assert GptImageProvider("k", model="gpt-image-1.5", size="1024x1024").size == "1024x1024"


def test_gpt_image_errors_do_not_leak_the_key() -> None:
    session = FakeSession(lambda m, u, kw: FakeResponse(400, text='{"error":{"message":"moderation"}}'))
    with pytest.raises(ImageError) as err:
        GptImageProvider("sk-openai-secret", session=session, sleep=lambda s: None).generate(
            ImageRequest("s001", "p", None, None))
    assert "sk-openai-secret" not in str(err.value) and "400" in str(err.value)


def test_flux_submit_poll_download() -> None:
    data = png_bytes((64, 36))
    polls = iter(["Pending", "Pending", "Ready"])

    def handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST":
            return FakeResponse(200, {"id": "abc", "polling_url": "https://api.bfl.ai/v1/get_result?id=abc"})
        if "get_result" in url:
            status = next(polls)
            result = {"sample": "https://delivery.bfl.ai/signed/abc.png"} if status == "Ready" else None
            return FakeResponse(200, {"status": status, "result": result})
        return FakeResponse(200, content=data)

    session = FakeSession(handler)
    provider = FluxImageProvider("bfl-secret", session=session, sleep=lambda s: None)
    assert provider.generate(ImageRequest("s001", "A prompt.", None, 42)) == data
    post = session.calls[0]
    assert post[1] == "https://api.bfl.ai/v1/flux-2-pro" and post[2]["headers"]["x-key"] == "bfl-secret"
    assert post[2]["json"] == {"prompt": "A prompt.", "width": 2560, "height": 1440, "output_format": "png", "seed": 42}
    assert [c[1] for c in session.calls[1:4]] == ["https://api.bfl.ai/v1/get_result?id=abc"] * 3
    download = session.calls[4]
    assert download[1].startswith("https://delivery.bfl.ai/") and "x-key" not in download[2].get("headers", {})


def test_flux_moderation_is_fatal_and_bad_size_rejected() -> None:
    def handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if method == "POST":
            return FakeResponse(200, {"id": "abc", "polling_url": "https://api.bfl.ai/v1/get_result?id=abc"})
        return FakeResponse(200, {"status": "Content Moderated", "result": None})

    session = FakeSession(handler)
    with pytest.raises(ImageError, match="Moderated"):
        FluxImageProvider("k", session=session, sleep=lambda s: None).generate(ImageRequest("s001", "p", None, 1))
    assert len(session.calls) == 2                            # not retried
    with pytest.raises(ImageError, match="multiples of 16"):
        FluxImageProvider("k", size="1921x1080")


# ---- stage ----------------------------------------------------------------------------
class CountingProvider(images_mod.ImageProvider):
    name = "flux"
    supports_seed = True

    def __init__(self) -> None:
        self.requests: list[ImageRequest] = []
        self.fail_for: set[str] = set()

    def cache_identity(self) -> dict[str, Any]:
        return {"model": "test"}

    def generate(self, request: ImageRequest) -> bytes:
        self.requests.append(request)
        if request.scene_id in self.fail_for:
            raise ImageError("boom")
        return png_bytes((1536, 1024))


def test_images_stage_cache_reuse_and_resume(tmp_path: Path, example_doc: dict[str, Any],
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    doc = approved(example_doc)
    doc["sections"] = doc["sections"][:2]                    # s001..s008 (s007 reuses s003, s002 is a diagram)
    doc["shorts"] = []
    doc["distribution"]["telegram_brief"]["image_scene_ids"] = ["s004"]
    provider = CountingProvider()
    provider.fail_for = {"s005"}
    monkeypatch.setattr(images_mod, "make_image_provider", lambda ctx: provider)
    ctx = make_context(tmp_path, doc, mock=False)

    with pytest.raises(ImageError, match="1 image"):
        run_images_stage(ctx)                                 # one failure does not stop the others
    done = {sid for sid, s in ctx.state.scenes.items() if s.image}
    assert done == {"s001", "s002", "s003", "s004", "s006", "s007", "s008"}
    assert ctx.state.costs.images_generated == 5            # s005 failed, nothing billed for it
    first_round = len(provider.requests)

    provider.fail_for = set()
    run_images_stage(ctx)                                     # resume: only the failed one is requested
    assert len(provider.requests) == first_round + 1 and provider.requests[-1].scene_id == "s005"
    assert ctx.state.costs.images_generated == 6

    s3, s7, s2 = (ctx.state.scenes[k].image for k in ("s003", "s007", "s002"))
    assert s3 and s7 and s2
    assert s7.kind == "reuse" and s7.path == s3.path          # same picture, own camera move later
    assert s2.kind == "diagram" and Image.open(tmp_path / s2.path).size == (1920, 1080)
    assert Image.open(tmp_path / s3.path).size == (2400, 1350)
    seeds = {r.seed for r in provider.requests}
    assert len(seeds) == 1 and None not in seeds              # fixed_per_video
    assert all("Avoid: text, letters" in r.prompt and r.negative_prompt is None for r in provider.requests)

    ctx.state.scenes.clear()                                  # state lost: the prompt cache still saves the money
    run_images_stage(ctx)
    assert len(provider.requests) == first_round + 1 and ctx.state.costs.images_cache_hits == 6


def test_unimplemented_provider_is_a_clear_error(tmp_path: Path, example_doc: dict[str, Any]) -> None:
    doc = approved(small_doc(example_doc, ["s001", "s002"]))
    doc["production"]["visual_style"]["image_provider"] = "ideogram"
    ctx = make_context(tmp_path, doc, mock=False)
    with pytest.raises(ImageError, match="not implemented"):
        run_images_stage(ctx)


# ---- diagrams ---------------------------------------------------------------------------
STYLE = DiagramStyle()


@pytest.mark.parametrize("template,params", [
    ("big_number", {"value": "$460,000,000", "sub_value": "45 min"}),
    ("counter", {"from": 0, "to": 97, "icon": "mail"}),
    ("icon_grid", {"icon": "server", "total": 8, "highlight_index": 7, "highlight_mark": "x"}),
    ("timeline", {"points": [{"label": "2003"}, {"label": "2012", "highlight": True}, {"label": "2013"}]}),
    ("bar_compare", {"bars": [{"label": "$12M", "value": 12}, {"label": "$460M", "value": 460}]}),
])
def test_templates_render_full_hd_on_dark_background_with_highlight(template: str, params: dict[str, Any]) -> None:
    image = render_diagram(template, params, STYLE)
    assert image.size == (1920, 1080)
    assert image.getpixel((4, 4)) == (14, 17, 22)                         # dark background
    colors = {c for _, c in image.getcolors(maxcolors=1 << 24)}
    assert (255, 212, 0) in colors                                        # highlight colour is used


def test_diagram_draws_nothing_when_params_are_empty() -> None:
    for template in ("big_number", "timeline", "bar_compare"):
        image = render_diagram(template, {}, STYLE)
        assert len(image.getcolors(maxcolors=1 << 24)) == 1               # no invented text or shapes


def test_counter_animation(tmp_path: Path) -> None:
    params = {"from": 0, "to": 97, "icon": "mail"}
    assert counter_value(params, 0.0) == 0 and counter_value(params, 1.0) == 97
    assert format_counter({"prefix": "$", "suffix": "M", "decimals": 1}, 4.56) == "$4.6M"
    assert format_counter({}, 12345) == "12,345"
    render_counter_sequence(params, STYLE, (480, 270), total_frames=20, count_frames=12, out_dir=tmp_path)
    frames = sorted(tmp_path.glob("*.png"))
    assert len(frames) == 20
    first, mid, last = (frames[i].read_bytes() for i in (0, 6, 19))
    assert first != mid != last
    assert frames[11].read_bytes() == last == frames[15].read_bytes()     # holds the final value
    assert render_diagram("counter", params, STYLE, (480, 270), 1.0).size == (480, 270)
