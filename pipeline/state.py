"""The pipeline-owned state file ``state/<video_id>.json``.

Runtime facts live here (never in the script): stage statuses, per-scene audio
paths, durations, word timings, image paths, input hashes, the frame-accurate
timeline and cost accounting. Writes are atomic so a crash never corrupts it.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

STATE_VERSION = 1
# Stages whose records live in ``PipelineState.stages`` (the layout of that dict is frozen:
# the task-1 tests pin its exact keys).
STAGES: tuple[str, ...] = ("tts", "images", "render", "shorts")
# Everything ``pipeline run`` executes, in order. ``thumbnail`` keeps its stage record in
# ``PipelineState.thumbnail.stage`` — use ``PipelineState.stage_record()`` to reach any of them.
RUN_STAGES: tuple[str, ...] = ("tts", "images", "thumbnail", "render", "shorts")
StageStatus = Literal["pending", "running", "done", "failed"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(payload: Any) -> str:
    """Deterministic short hash of any JSON-serialisable structure."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:24]


class WordTiming(BaseModel):
    text: str
    start: float
    end: float


class TTSRecord(BaseModel):
    """Result of one TTS line (a scene or a Short's hook)."""

    hash: str                      # hash of everything that shaped this record
    audio_hash: str                # cache key of the synthesized audio
    audio_path: str
    duration_sec: float
    characters: int                # characters sent to TTS
    provider: str
    from_cache: bool = False
    spoken_words: list[WordTiming] = Field(default_factory=list)   # what TTS read
    words: list[WordTiming] = Field(default_factory=list)          # subtitle words, scene-relative


class ImageRecord(BaseModel):
    hash: str
    path: str
    kind: str                      # generate | stock | diagram | reuse
    provider: str | None = None
    seed: int | None = None
    from_cache: bool = False


class SceneTiming(BaseModel):
    """Position of a scene on the master timeline. Frames are authoritative."""

    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    voice_end_sec: float


class ClipRecord(BaseModel):
    hash: str
    path: str
    frames: int


class SceneState(BaseModel):
    scene_id: str
    section_id: str
    tts: TTSRecord | None = None
    image: ImageRecord | None = None
    timing: SceneTiming | None = None
    clip: ClipRecord | None = None


class StageState(BaseModel):
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class ThumbnailState(BaseModel):
    """The ``thumbnail`` stage: its status plus what it produced."""

    stage: StageState = Field(default_factory=StageState)
    image: ImageRecord | None = None       # generated background (images/thumbnail_bg.png)
    hash: str | None = None                # hash of everything that shaped final/thumbnail.jpg
    path: str | None = None
    bytes: int | None = None
    jpeg_quality: int | None = None
    info: dict[str, Any] = Field(default_factory=dict)   # text layout: lines, font size, letter height, box


class UploadSession(BaseModel):
    """A resumable upload in flight. The URL authorises that one upload — it is never logged."""

    url: str
    source_hash: str               # the file being sent (render hash) ...
    body_hash: str                 # ... and its metadata: a session is only resumed for the same pair
    size: int
    started_at: str = Field(default_factory=now_iso)


class UploadedVideo(BaseModel):
    """What is known about one video on YouTube: the long video or one Short."""

    key: str                       # "long" or the short_id
    kind: Literal["long", "short"]
    video_id: str | None = None    # YouTube video ID once the upload has finished
    uploaded_at: str | None = None
    source_hash: str | None = None           # render hash of the uploaded file
    title: str | None = None
    requested_privacy: str | None = None
    publish_at: str | None = None            # UTC, RFC 3339
    actual_privacy: str | None = None        # what videos.list reported after the upload
    upload_status: str | None = None
    verified_at: str | None = None
    thumbnail_set: bool = False
    thumbnail_hash: str | None = None
    captions_uploaded: bool = False
    captions_hash: str | None = None
    caption_id: str | None = None
    session: UploadSession | None = None
    previous_video_ids: list[str] = Field(default_factory=list)   # replaced by --force-upload, still on YouTube

    @property
    def url(self) -> str | None:
        if not self.video_id:
            return None
        return f"https://youtu.be/{self.video_id}" if self.kind == "long" else f"https://youtube.com/shorts/{self.video_id}"


class UploadState(BaseModel):
    youtube_channel_id: str | None = None
    youtube_channel_title: str | None = None
    long: UploadedVideo | None = None
    shorts: dict[str, UploadedVideo] = Field(default_factory=dict)

    def videos(self) -> list[UploadedVideo]:
        return ([self.long] if self.long else []) + list(self.shorts.values())


class Costs(BaseModel):
    """Billable usage accumulated over the life of the state file."""

    tts_characters: int = 0        # characters actually sent to the TTS API
    tts_requests: int = 0
    tts_cache_hits: int = 0
    images_generated: int = 0      # images actually requested from an image API
    images_cache_hits: int = 0


class Artifact(BaseModel):
    hash: str
    path: str
    info: dict[str, Any] = Field(default_factory=dict)


class TimelineInfo(BaseModel):
    fps: int
    total_frames: int
    duration_sec: float


class ShortState(BaseModel):
    short_id: str
    status: StageStatus = "pending"
    hook: TTSRecord | None = None
    path: str | None = None
    duration_sec: float | None = None
    frames: int | None = None
    scenes: dict[str, SceneTiming] = Field(default_factory=dict)
    clips: dict[str, ClipRecord] = Field(default_factory=dict)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)


class PipelineState(BaseModel):
    state_version: int = STATE_VERSION
    video_id: str
    script_sha256: str | None = None
    mock: bool = False
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    stages: dict[str, StageState] = Field(default_factory=lambda: {s: StageState() for s in STAGES})
    timeline: TimelineInfo | None = None
    scenes: dict[str, SceneState] = Field(default_factory=dict)
    shorts: dict[str, ShortState] = Field(default_factory=dict)
    artifacts: dict[str, Artifact] = Field(default_factory=dict)
    costs: Costs = Field(default_factory=Costs)
    thumbnail: ThumbnailState = Field(default_factory=ThumbnailState)
    upload: UploadState = Field(default_factory=UploadState)

    def stage_record(self, stage: str) -> StageState:
        """The status record of any stage in ``RUN_STAGES``."""
        if stage == "thumbnail":
            return self.thumbnail.stage
        return self.stages.setdefault(stage, StageState())

    def scene(self, scene_id: str, section_id: str) -> SceneState:
        entry = self.scenes.get(scene_id)
        if entry is None:
            entry = SceneState(scene_id=scene_id, section_id=section_id)
            self.scenes[scene_id] = entry
        entry.section_id = section_id
        return entry


class StateStore:
    """Loads and atomically saves one video's state. Thread-safe."""

    def __init__(self, state_dir: Path, video_id: str) -> None:
        self.path = state_dir / f"{video_id}.json"
        self.video_id = video_id
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> PipelineState:
        if not self.path.is_file():
            return PipelineState(video_id=self.video_id)
        state = PipelineState.model_validate_json(self.path.read_text(encoding="utf-8"))
        for stage in STAGES:
            state.stages.setdefault(stage, StageState())
        return state

    def save(self, state: PipelineState) -> None:
        with self._lock:
            state.updated_at = now_iso()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            os.replace(tmp, self.path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
