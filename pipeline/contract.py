"""The script contract: pydantic v2 models mirroring ``video_script.schema.json``.

Validation is layered:
1. the supplied ``validate_script.py`` (JSON Schema + semantic checks) — used as is;
2. these models, which give the rest of the code typed, read-only access.

The script file is read-only for the pipeline; nothing here ever writes to it.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VALIDATOR_FILE = "validate_script.py"
SCHEMA_FILE = "video_script.schema.json"

MusicMood = Literal["tense_minimal", "curious_pulse", "dread_drone", "uplifting_resolve", "reflective_piano"]
MotionType = Literal["static", "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]
Transition = Literal["cut", "crossfade", "dip_to_black", "whip"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---- editorial / seo / thumbnail -------------------------------------------
class Source(_Model):
    source_id: str
    title: str
    url: str
    kind: Literal["primary", "secondary"]


class Editorial(_Model):
    working_title: str
    logline: str
    original_angle: str
    target_duration_sec: int
    sources: list[Source]


class Seo(_Model):
    title_seo: str
    title_variants: list[str] = Field(default_factory=list)
    description: str
    youtube_tags: list[str]
    category_id: int
    made_for_kids: bool
    contains_synthetic_media: bool


class Thumbnail(_Model):
    image_prompt: str
    text_overlay: str
    text_overlay_color: str | None = None


# ---- production --------------------------------------------------------------
class Pacing(_Model):
    wpm: int
    scene_min_sec: float
    scene_max_sec: float


class VoiceSettings(_Model):
    stability: float | None = None
    similarity_boost: float | None = None
    style: float | None = None


class Tts(_Model):
    provider: Literal["elevenlabs"]
    voice_id: str
    model_id: str
    voice_settings: VoiceSettings | None = None
    stitching: Literal["previous_next_text", "none"]


class VisualStyle(_Model):
    style_id: str
    image_provider: Literal["flux", "gpt_image", "nano_banana", "ideogram"]
    global_prompt_suffix: str
    negative_prompt: str
    seed_strategy: Literal["fixed_per_video", "fixed_per_section", "random"]


class BurnIn(_Model):
    long: bool
    shorts: bool


class Subtitles(_Model):
    burn_in: BurnIn
    caption_tracks: bool
    style: Literal["yellow_dynamic", "plain"]
    font: str
    max_words_per_chunk: int
    base_color: str | None = None
    highlight_color: str
    position: Literal["bottom_center", "center"]


class Music(_Model):
    source: Literal["youtube_audio_library", "licensed_library"]
    default_mood: MusicMood
    ducking_db: float


class Entity(_Model):
    entity_id: str
    kind: Literal["character", "place", "object"]
    visual_description: str


class Pronunciation(_Model):
    term: str
    say_as: str


class Production(_Model):
    audio_stems: bool
    aspect_ratio: Literal["16:9", "9:16"]
    resolution: Literal["1920x1080", "1080x1920"]
    fps: Literal[24, 25, 30]
    pacing: Pacing
    tts: Tts
    visual_style: VisualStyle
    subtitles: Subtitles
    music: Music
    entities: list[Entity] = Field(default_factory=list)
    pronunciation: list[Pronunciation] = Field(default_factory=list)

    @property
    def frame_size(self) -> tuple[int, int]:
        w, h = self.resolution.split("x")
        return int(w), int(h)


# ---- localization --------------------------------------------------------------
class CustomTrack(_Model):
    language: str
    voice_id: str
    status: Literal["planned", "rendered", "uploaded"]


class Localization(_Model):
    auto_dubbing: bool
    on_screen_text_policy: Literal["language_neutral", "english_ok"]
    custom_tracks: list[CustomTrack]


# ---- scenes ----------------------------------------------------------------------
class Diagram(_Model):
    template: Literal["big_number", "counter", "icon_grid", "timeline", "bar_compare"]
    params: dict[str, Any]


class Motion(_Model):
    type: MotionType
    intensity: float | None = None


class Visual(_Model):
    source: Literal["generate", "stock", "reuse", "diagram"]
    image_prompt: str | None = None
    entity_refs: list[str] = Field(default_factory=list)
    stock_query: str | None = None
    reuse_scene_id: str | None = None
    diagram: Diagram | None = None
    motion: Motion
    transition_in: Transition


class Scene(_Model):
    scene_id: str
    beat: Literal["hook", "context", "build", "turn", "climax", "payoff", "outro"]
    voiceover_text: str
    subtitle_text: str | None = None
    emphasis_words: list[str] = Field(default_factory=list)
    visual: Visual
    sfx: Literal["low_boom", "whoosh", "error_beep", "alarm_rise", "typing", "cash_register"] | None = None
    source_refs: list[str]

    @property
    def display_text(self) -> str:
        """Text shown on screen: ``subtitle_text`` or, by default, ``voiceover_text``."""
        return self.subtitle_text if self.subtitle_text is not None else self.voiceover_text


class Section(_Model):
    section_id: str
    chapter_title: str
    music_mood: MusicMood | None = None
    scenes: list[Scene]


class Short(_Model):
    short_id: str
    scene_ids: list[str]
    hook_override_text: str | None = None
    reframe: Literal["blur_background", "center_crop", "pan_scan"]
    end_card_text: str | None = None


# ---- distribution / qa (carried through untouched; out of scope for rendering) ----
class KeyFact(_Model):
    fact: str
    source_ref: str


class TelegramBrief(_Model):
    target_language: Literal["ru"]
    angle_ru: str
    key_facts: list[KeyFact]
    post_formats: list[Literal["story_post", "carousel", "quiz_poll", "longread_telegraph"]]
    image_scene_ids: list[str] = Field(default_factory=list)
    do_not_translate: list[str] = Field(default_factory=list)
    max_caption_chars: int


class Distribution(_Model):
    telegram_brief: TelegramBrief


class Qa(_Model):
    facts_verified: bool
    reviewer: str | None = None
    no_photoreal_real_people: bool
    music_licensed: bool
    originality_note: str
    notes: str | None = None


class Script(_Model):
    schema_version: Literal["1.0.0"]
    video_id: str
    channel_id: str
    niche: Literal["business_stories", "tech_disasters", "behavior_science"]
    language: Literal["en-US"]
    format: Literal["long", "short"]
    status: Literal["draft", "needs_review", "approved", "rejected"]
    created_by: str | None = None
    created_at: str | None = None
    editorial: Editorial
    seo: Seo
    thumbnail: Thumbnail
    production: Production
    localization: Localization
    sections: list[Section]
    shorts: list[Short] = Field(default_factory=list)
    distribution: Distribution
    qa: Qa

    # ---- navigation helpers -------------------------------------------------
    def scenes_in_order(self) -> list[tuple[Section, Scene]]:
        return [(sec, sc) for sec in self.sections for sc in sec.scenes]

    def scene_ids(self) -> list[str]:
        return [sc.scene_id for _, sc in self.scenes_in_order()]

    def scene_by_id(self, scene_id: str) -> Scene:
        for _, scene in self.scenes_in_order():
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(scene_id)

    def section_of(self, scene_id: str) -> Section:
        for section, scene in self.scenes_in_order():
            if scene.scene_id == scene_id:
                return section
        raise KeyError(scene_id)

    def mood_of(self, section: Section) -> str:
        return section.music_mood or self.production.music.default_mood

    def entity_map(self) -> dict[str, Entity]:
        return {e.entity_id: e for e in self.production.entities}


# ---- validation -------------------------------------------------------------------
@dataclass
class ValidationReport:
    schema_errors: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.schema_errors and not self.errors


class ContractError(Exception):
    """The script does not satisfy the contract."""


def find_contract_file(name: str) -> Path:
    """Locate a contract file: ``$PIPELINE_CONTRACT_DIR``, the project root, then the CWD."""
    candidates: list[Path] = []
    env_dir = os.environ.get("PIPELINE_CONTRACT_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / name)
    candidates.append(Path(__file__).resolve().parent.parent / name)
    candidates.append(Path.cwd() / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} not found (looked in: {', '.join(str(c.parent) for c in candidates)})")


def load_validator_module() -> ModuleType:
    """Import the supplied ``validate_script.py`` unchanged, by file path."""
    path = find_contract_file(VALIDATOR_FILE)
    spec = importlib.util.spec_from_file_location("validate_script", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_validator_cli(script_path: Path) -> int:
    """``pipeline validate``: delegate to the supplied validator's own ``main()``."""
    module = load_validator_module()
    schema = find_contract_file(SCHEMA_FILE)
    old_argv = sys.argv
    sys.argv = [VALIDATOR_FILE, str(script_path), "--schema", str(schema)]
    try:
        return int(module.main())
    finally:
        sys.argv = old_argv


def validate_document(doc: dict[str, Any]) -> ValidationReport:
    """Same checks as the validator's ``main()``, returned as data instead of printed."""
    module = load_validator_module()
    schema = json.loads(find_contract_file(SCHEMA_FILE).read_text(encoding="utf-8"))
    validator = module.Draft202012Validator(schema, format_checker=module.FormatChecker())
    report = ValidationReport()
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.path) or "<root>"
        report.schema_errors.append(f"{loc}: {err.message}")
    if report.schema_errors:
        return report
    report.errors, report.warnings = module.semantic_checks(doc)
    return report


@dataclass(frozen=True)
class LoadedScript:
    script: Script
    sha256: str
    path: Path
    report: ValidationReport


def load_script(path: Path) -> LoadedScript:
    """Read, validate (supplied validator, then pydantic) and return the script."""
    raw = path.read_bytes()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: not valid UTF-8 JSON ({exc})") from exc
    report = validate_document(doc)
    if not report.ok:
        problems = [f"schema: {e}" for e in report.schema_errors] + report.errors
        raise ContractError(f"{path.name} failed validation:\n  " + "\n  ".join(problems))
    script = Script.model_validate(doc)
    return LoadedScript(script=script, sha256=hashlib.sha256(raw).hexdigest(), path=path, report=report)
