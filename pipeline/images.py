"""Stage 5 — scene images.

``ImageProvider`` implementations: ``mock`` (flat placeholders), ``gpt_image``
(OpenAI Images API) and ``flux`` (Black Forest Labs FLUX.2 API). Diagram scenes
are drawn by :mod:`pipeline.diagrams`; ``reuse`` scenes point at an earlier
scene's picture.

API facts checked against the docs in 2026-11:
* OpenAI ``POST /v1/images/generations`` — GPT image models always return
  ``data[0].b64_json``; there is no seed and no negative prompt. gpt-image-2 and
  gpt-image-2.5 accept custom ``WIDTHxHEIGHT`` sizes (multiples of 16, up to
  2560x1440 non-experimental); other GPT image models top out at 1536x1024.
  gpt-image-1 was shut down on 2026-10-23.
* BFL ``POST /v1/flux-2-pro`` — ``prompt``, ``width``, ``height``, ``seed``,
  ``output_format``; answers ``{id, polling_url}``; poll until ``status == "Ready"``
  and download ``result.sample`` (signed URL, valid ~10 minutes). No negative prompt.
"""
from __future__ import annotations

import base64
import hashlib
import io
import random
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image, ImageDraw

from pipeline.context import RunContext
from pipeline.contract import Scene, Script, Section
from pipeline.diagrams import DIAGRAM_VERSION, DiagramStyle, find_font_file, load_font, render_diagram
from pipeline.http import APIError, RetryableAPIError, call_with_retries, send
from pipeline.logging_setup import get_logger
from pipeline.state import ImageRecord, atomic_write_bytes, stable_hash

KEN_BURNS_RESERVE = 1.25   # stored image is at least this many times the frame
MAX_RESERVE = 2.0
IMAGE_POST_VERSION = 1


class ImageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageRequest:
    scene_id: str
    prompt: str
    negative_prompt: str | None
    seed: int | None


class ImageProvider(ABC):
    """An image source. Returns encoded image bytes (PNG/JPEG/WebP) for one prompt."""

    name: str = "abstract"
    supports_negative_prompt: bool = False
    supports_seed: bool = False
    billable: bool = True

    @abstractmethod
    def cache_identity(self) -> dict[str, Any]:
        """Model/size/quality — everything besides the prompt that changes the picture."""

    @abstractmethod
    def generate(self, request: ImageRequest) -> bytes:
        ...


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
class MockImageProvider(ImageProvider):
    """Flat-colour placeholder with the scene number. No network, no keys."""

    name = "mock"
    billable = False

    def __init__(self, size: tuple[int, int], font_path: str | None) -> None:
        self.size = size
        self.font_path = font_path

    def cache_identity(self) -> dict[str, Any]:
        return {"size": self.size, "v": 2}

    def generate(self, request: ImageRequest) -> bytes:
        digits = "".join(ch for ch in request.scene_id if ch.isdigit()) or "0"
        number = int(digits)
        hue = (number * 47) % 360
        image = Image.new("RGB", self.size, f"hsl({hue}, 35%, 28%)")
        draw = ImageDraw.Draw(image)
        font = load_font(self.font_path, int(self.size[1] * 0.42))
        draw.text((self.size[0] / 2, self.size[1] / 2), str(number), font=font, fill=f"hsl({hue}, 40%, 78%)", anchor="mm")
        buf = io.BytesIO()
        image.save(buf, "PNG", compress_level=3)
        return buf.getvalue()


_OPENAI_CUSTOM_SIZE_PREFIXES = ("gpt-image-2-", "gpt-image-2.5-")


class GptImageProvider(ImageProvider):
    """OpenAI Images API (``/v1/images/generations``) with a GPT image model."""

    name = "gpt_image"

    def __init__(self, api_key: str, *, model: str = "gpt-image-2", size: str | None = None, quality: str = "medium",
                 base_url: str = "https://api.openai.com", attempts: int = 6,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise ImageError("OPENAI_API_KEY is not set (put it into .env or the environment)")
        self._api_key = api_key
        self.model = model
        custom = model == "gpt-image-2" or model.startswith(_OPENAI_CUSTOM_SIZE_PREFIXES)
        # Largest landscape size: 16:9 2560x1440 where custom sizes exist, else the 3:2 preset.
        self.size = size or ("2560x1440" if custom else "1536x1024")
        self.quality = quality
        self.base_url = base_url.rstrip("/")
        self.attempts = attempts
        self._session = session or requests.Session()
        self._sleep = sleep

    def cache_identity(self) -> dict[str, Any]:
        return {"model": self.model, "size": self.size, "quality": self.quality}

    def generate(self, request: ImageRequest) -> bytes:
        body = {"model": self.model, "prompt": request.prompt, "size": self.size, "quality": self.quality,
                "n": 1, "output_format": "png"}

        def call() -> requests.Response:
            return send(self._session, "POST", f"{self.base_url}/v1/images/generations", service="OpenAI Images",
                        headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                        json=body, timeout=(10, 300))

        try:
            response = call_with_retries(call, what=f"image {request.scene_id}", log=get_logger("images"),
                                         attempts=self.attempts, sleep=self._sleep)
            return base64.b64decode(response.json()["data"][0]["b64_json"])
        except APIError as exc:
            raise ImageError(str(exc)) from None
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ImageError(f"OpenAI Images: unexpected response shape ({type(exc).__name__})") from None


class FluxImageProvider(ImageProvider):
    """Black Forest Labs FLUX.2 API: submit, poll ``polling_url``, download the signed result URL."""

    name = "flux"
    supports_seed = True

    def __init__(self, api_key: str, *, model: str = "flux-2-pro", size: str = "2560x1440",
                 base_url: str = "https://api.bfl.ai", attempts: int = 6, poll_interval: float = 1.0,
                 poll_timeout: float = 300.0, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise ImageError("BFL_API_KEY is not set (put it into .env or the environment)")
        self._api_key = api_key
        self.model = model
        width, _, height = size.lower().partition("x")
        self.width, self.height = int(width), int(height)
        if self.width % 16 or self.height % 16:
            raise ImageError(f"BFL_IMAGE_SIZE {size}: width and height must be multiples of 16")
        self.base_url = base_url.rstrip("/")
        self.attempts = attempts
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._session = session or requests.Session()
        self._sleep = sleep

    def cache_identity(self) -> dict[str, Any]:
        return {"model": self.model, "width": self.width, "height": self.height}

    def _attempt(self, request: ImageRequest) -> bytes:
        headers = {"x-key": self._api_key, "accept": "application/json"}
        body: dict[str, Any] = {"prompt": request.prompt, "width": self.width, "height": self.height,
                                "output_format": "png"}
        if request.seed is not None:
            body["seed"] = request.seed
        created = send(self._session, "POST", f"{self.base_url}/v1/{self.model}", service="BFL",
                       headers={**headers, "Content-Type": "application/json"}, json=body).json()
        polling_url = created.get("polling_url")
        if not polling_url:
            raise ImageError("BFL: response has no polling_url")
        waited = 0.0
        while waited < self.poll_timeout:
            self._sleep(self.poll_interval)
            waited += self.poll_interval
            result = send(self._session, "GET", polling_url, service="BFL", headers=headers).json()
            status = str(result.get("status", ""))
            if status == "Ready":
                sample = (result.get("result") or {}).get("sample")
                if not sample:
                    raise ImageError("BFL: Ready without result.sample")
                # Signed URL: no API key header on this request.
                return send(self._session, "GET", sample, service="BFL download").content
            if status in ("Error", "Failed"):
                raise RetryableAPIError(f"BFL: generation {status.lower()}")
            if "Moderated" in status:
                raise ImageError(f"BFL: {status} for scene {request.scene_id} — rewrite the prompt")
            if status == "Task not found":
                raise RetryableAPIError("BFL: task not found")
        raise RetryableAPIError("BFL: timed out waiting for the image")

    def generate(self, request: ImageRequest) -> bytes:
        try:
            return call_with_retries(lambda: self._attempt(request), what=f"image {request.scene_id}",
                                     log=get_logger("images"), attempts=self.attempts, sleep=self._sleep)
        except APIError as exc:
            raise ImageError(str(exc)) from None
        except RetryableAPIError as exc:
            raise ImageError(f"{exc} (gave up after {self.attempts} attempts)") from None
        except (ValueError, TypeError) as exc:
            raise ImageError(f"BFL: unexpected response shape ({type(exc).__name__})") from None


def make_image_provider(ctx: RunContext) -> ImageProvider:
    settings = ctx.settings
    w, h = ctx.frame_size
    if ctx.mock:
        font = find_font_file(settings.assets_dir, None)
        return MockImageProvider((int(w * KEN_BURNS_RESERVE), int(h * KEN_BURNS_RESERVE)), font)
    name = settings.image_provider_override or ctx.script.production.visual_style.image_provider
    if name == "gpt_image":
        return GptImageProvider(settings.openai_api_key or "", model=settings.openai_image_model,
                                size=settings.openai_image_size, quality=settings.openai_image_quality,
                                base_url=settings.openai_base_url, attempts=settings.http_attempts)
    if name == "flux":
        return FluxImageProvider(settings.bfl_api_key or "", model=settings.bfl_model, size=settings.bfl_image_size,
                                 base_url=settings.bfl_base_url, attempts=settings.http_attempts)
    raise ImageError(f"image provider '{name}' is not implemented (available: gpt_image, flux, mock). "
                     f"Set PIPELINE_IMAGE_PROVIDER=gpt_image or flux to override the script.")


# --------------------------------------------------------------------------- #
# prompts, seeds, post-processing
# --------------------------------------------------------------------------- #
def _join_sentences(parts: list[str]) -> str:
    cleaned = [p.strip().rstrip(".").strip() for p in parts if p and p.strip()]
    return ". ".join(cleaned) + ("." if cleaned else "")


def build_prompt(script: Script, scene: Scene) -> tuple[str, str]:
    """(prompt, negative_prompt): scene subject + referenced entities + the global style suffix."""
    visual = scene.visual
    subject = visual.image_prompt or visual.stock_query or ""
    entities = script.entity_map()
    descriptions = [entities[ref].visual_description for ref in visual.entity_refs if ref in entities]
    style = script.production.visual_style
    return _join_sentences([subject, *descriptions, style.global_prompt_suffix]), style.negative_prompt.strip()


def prompt_for_provider(provider: ImageProvider, prompt: str, negative: str) -> tuple[str, str | None]:
    """Negative prompt goes separately when supported, otherwise it is folded into the prompt."""
    if not negative:
        return prompt, None
    if provider.supports_negative_prompt:
        return prompt, negative
    return f"{prompt} Avoid: {negative}.", None


def seed_from_key(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 2_147_483_647


def seed_for(script: Script, section: Section | None, strategy: str) -> int | None:
    """Deterministic seed per video / per section; ``None`` for the random strategy.

    ``section=None`` is the thumbnail: it is its own "section" under ``fixed_per_section``.
    """
    if strategy == "fixed_per_video":
        key = script.video_id
    elif strategy == "fixed_per_section":
        key = f"{script.video_id}/{section.section_id if section is not None else 'thumbnail'}"
    else:
        return None
    return seed_from_key(key)


def fit_to_frame(data: bytes, frame: tuple[int, int]) -> Image.Image:
    """Centre-crop to the frame's aspect ratio and make sure there is Ken Burns headroom.

    The result is at least 1.25x the frame (Lanczos upscale if needed) and at most 2x.
    """
    image = Image.open(io.BytesIO(data)).convert("RGB")
    fw, fh = frame
    target_ratio = fw / fh
    w, h = image.size
    if w / h > target_ratio:
        new_w = round(h * target_ratio)
        left = (w - new_w) // 2
        image = image.crop((left, 0, left + new_w, h))
    elif w / h < target_ratio:
        new_h = round(w / target_ratio)
        top = (h - new_h) // 2
        image = image.crop((0, top, w, top + new_h))
    scale = min(max(image.size[0] / fw, KEN_BURNS_RESERVE), MAX_RESERVE)
    out_size = (int(round(fw * scale / 2)) * 2, int(round(fh * scale / 2)) * 2)
    if image.size != out_size:
        image = image.resize(out_size, Image.LANCZOS)
    return image


def diagram_style(ctx: RunContext) -> DiagramStyle:
    subs = ctx.script.production.subtitles
    return DiagramStyle(foreground=subs.base_color or "#FFFFFF", highlight=subs.highlight_color,
                        font_path=find_font_file(ctx.settings.assets_dir, subs.font))


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", compress_level=4)
    return buf.getvalue()


def generate_picture(ctx: RunContext, provider: ImageProvider, *, label: str, prompt: str, negative: str,
                     section: Section | None, out_path: Path, previous: ImageRecord | None,
                     kind: str = "generate") -> ImageRecord:
    """One provider picture with everything the scenes get: prompt folding, seed strategy, cache, cost counters.

    Shared by scene images and the thumbnail background. ``label`` is the scene id (or ``"thumbnail"``).
    """
    final_prompt, native_negative = prompt_for_provider(provider, prompt, negative)
    strategy = ctx.script.production.visual_style.seed_strategy
    seed = seed_for(ctx.script, section, strategy) if provider.supports_seed else None
    key = {"provider": provider.name, "identity": provider.cache_identity(), "prompt": final_prompt,
           "negative": native_negative, "seed": seed, "frame": ctx.frame_size, "post": IMAGE_POST_VERSION}
    if provider.name == "mock":
        key["scene"] = label
    image_hash = stable_hash(key)
    if previous and previous.hash == image_hash and ctx.settings.abs(previous.path).is_file():
        return previous

    cache_file = ctx.settings.cache_dir / "images" / f"{image_hash}.png"
    from_cache = not ctx.no_cache and cache_file.is_file()
    if from_cache:
        data = cache_file.read_bytes()
    else:
        if provider.supports_seed and seed is None:
            seed = random.randrange(1, 2_147_483_647)  # "random" strategy: still record what was used
        raw = provider.generate(ImageRequest(label, final_prompt, native_negative, seed))
        data = _png_bytes(fit_to_frame(raw, ctx.frame_size))
        atomic_write_bytes(cache_file, data)
    atomic_write_bytes(out_path, data)
    with ctx.store.lock:
        if from_cache:
            ctx.state.costs.images_cache_hits += 1
        elif provider.billable:
            ctx.state.costs.images_generated += 1
    ctx.log.info("image %s: %s%s", label, provider.name, " (cache)" if from_cache else "")
    return ImageRecord(hash=image_hash, path=ctx.settings.rel(out_path), kind=kind, provider=provider.name,
                       seed=seed, from_cache=from_cache)


def _generate_one(ctx: RunContext, provider: ImageProvider, section: Section, scene: Scene) -> ImageRecord:
    prompt, negative = build_prompt(ctx.script, scene)
    return generate_picture(ctx, provider, label=scene.scene_id, prompt=prompt, negative=negative, section=section,
                            out_path=ctx.paths.images / f"{scene.scene_id}.png",
                            previous=ctx.state.scenes[scene.scene_id].image,
                            kind="stock" if scene.visual.source == "stock" else "generate")


def _diagram_one(ctx: RunContext, scene: Scene, style: DiagramStyle) -> ImageRecord:
    diagram = scene.visual.diagram
    assert diagram is not None
    image_hash = stable_hash({"template": diagram.template, "params": diagram.params, "style": style.__dict__,
                              "frame": ctx.frame_size, "v": DIAGRAM_VERSION})
    out_path = ctx.paths.images / f"{scene.scene_id}.png"
    previous = ctx.state.scenes[scene.scene_id].image
    if previous and previous.hash == image_hash and ctx.settings.abs(previous.path).is_file():
        return previous
    image = render_diagram(diagram.template, diagram.params, style, ctx.frame_size, progress=1.0)
    atomic_write_bytes(out_path, _png_bytes(image))
    ctx.log.info("image %s: diagram %s", scene.scene_id, diagram.template)
    return ImageRecord(hash=image_hash, path=ctx.settings.rel(out_path), kind="diagram")


def run_images_stage(ctx: RunContext) -> None:
    scenes = ctx.script.scenes_in_order()
    for section, scene in scenes:
        ctx.state.scene(scene.scene_id, section.section_id)
    style = diagram_style(ctx)

    # 1. diagrams (local, instant)
    for _, scene in scenes:
        if scene.visual.source == "diagram":
            ctx.state.scenes[scene.scene_id].image = _diagram_one(ctx, scene, style)
    ctx.save()

    # 2. generated pictures: at most `image_workers` (<= 4) requests in flight
    jobs = [(section, scene) for section, scene in scenes if scene.visual.source in ("generate", "stock")]
    failures: list[str] = []
    if jobs:
        provider = make_image_provider(ctx)
        for _, scene in jobs:
            if scene.visual.source == "stock":
                ctx.log.warning("image %s: stock search is not implemented — generating from stock_query instead",
                                scene.scene_id)
        ctx.log.info("images: %d to make via %s (up to %d in parallel)", len(jobs), provider.name,
                     ctx.settings.image_workers)
        with ThreadPoolExecutor(max_workers=min(4, ctx.settings.image_workers)) as pool:
            futures = {pool.submit(_generate_one, ctx, provider, section, scene): scene for section, scene in jobs}
            for future in as_completed(futures):
                scene = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001 — keep the other images going, fail the stage at the end
                    failures.append(f"{scene.scene_id}: {exc}")
                    ctx.log.error("image %s failed: %s", scene.scene_id, exc)
                    continue
                with ctx.store.lock:
                    ctx.state.scenes[scene.scene_id].image = record
                    ctx.save()

    # 3. reuse: the same picture as an earlier scene (its own camera move is applied at render time)
    for _, scene in scenes:
        if scene.visual.source != "reuse":
            continue
        target = ctx.state.scenes.get(scene.visual.reuse_scene_id or "")
        if target is None or target.image is None:
            failures.append(f"{scene.scene_id}: reuse target {scene.visual.reuse_scene_id} has no image")
            continue
        ctx.state.scenes[scene.scene_id].image = ImageRecord(
            hash=target.image.hash, path=target.image.path, kind="reuse", provider=target.image.provider,
            seed=target.image.seed)
    ctx.save()
    if failures:
        raise ImageError(f"{len(failures)} image(s) failed: " + "; ".join(failures[:5]))
    ctx.log.info("images done: %d generated via API so far, %d cache hits",
                 ctx.state.costs.images_generated, ctx.state.costs.images_cache_hits)
