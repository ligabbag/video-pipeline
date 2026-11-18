"""Stage 2 — text-to-speech with timestamps.

One request per scene to ElevenLabs' *with-timestamps* endpoint; neighbouring
scenes are passed as ``previous_text`` / ``next_text`` for continuous intonation.
Audio + character alignment are cached by content hash, so re-runs cost nothing.

Endpoint reference (checked 2026-11): POST /v1/text-to-speech/{voice_id}/with-timestamps
-> ``{"audio_base64", "alignment": {"characters", "character_start_times_seconds",
"character_end_times_seconds"}, "normalized_alignment": {...}}``.
"""
from __future__ import annotations

import base64
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from pipeline.align import ALIGNER_VERSION, Word, apply_pronunciation, chars_to_words, words_for_display
from pipeline.audio import audio_seconds
from pipeline.config import SCENE_PAUSE_SEC, SECTION_PAUSE_SEC, Settings
from pipeline.context import RunContext
from pipeline.ffmpeg import run_ffmpeg
from pipeline.http import APIError, call_with_retries, send
from pipeline.state import SceneTiming, TimelineInfo, TTSRecord, WordTiming, atomic_write_bytes, stable_hash
from pipeline.timeline import LineSpec, lay_out


class TTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class TTSRequest:
    text: str
    voice_id: str
    model_id: str
    voice_settings: dict[str, float] = field(default_factory=dict)
    previous_text: str | None = None
    next_text: str | None = None


@dataclass(frozen=True)
class TTSResult:
    audio: bytes
    characters: list[str]
    starts: list[float]
    ends: list[float]
    billed_characters: int


class TTSProvider(ABC):
    name: str = "abstract"
    audio_ext: str = "mp3"

    @abstractmethod
    def cache_identity(self) -> dict[str, Any]:
        """Provider settings that change the audio (part of the cache key)."""

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResult:
        ...


# --------------------------------------------------------------------------- #
# mock
# --------------------------------------------------------------------------- #
class MockTTS(TTSProvider):
    """Silence of the right length + fake per-character timings derived from ``pacing.wpm``."""

    name = "mock"
    audio_ext = "mp3"

    def __init__(self, wpm: int) -> None:
        self.wpm = wpm

    def cache_identity(self) -> dict[str, Any]:
        return {"wpm": self.wpm, "v": 1}

    def synthesize(self, request: TTSRequest) -> TTSResult:
        words = max(len(request.text.split()), 1)
        duration = max(words / (self.wpm / 60.0), 0.4)
        proc = run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", f"{duration:.3f}",
                           "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "-"], capture_stdout=True)
        chars = list(request.text)
        step = duration / max(len(chars), 1)
        starts = [round(i * step, 4) for i in range(len(chars))]
        ends = [round((i + 1) * step, 4) for i in range(len(chars))]
        return TTSResult(proc.stdout, chars, starts, ends, billed_characters=0)


# --------------------------------------------------------------------------- #
# ElevenLabs
# --------------------------------------------------------------------------- #
class ElevenLabsTTS(TTSProvider):
    name = "elevenlabs"

    def __init__(self, api_key: str, *, base_url: str = "https://api.elevenlabs.io",
                 output_format: str = "mp3_44100_128", attempts: int = 6,
                 session: requests.Session | None = None, sleep: Callable[[float], None] = time.sleep,
                 log: Any = None) -> None:
        if not api_key:
            raise TTSError("ELEVENLABS_API_KEY is not set (put it into .env or the environment)")
        prefix = output_format.split("_", 1)[0]
        if prefix not in ("mp3", "wav", "opus"):
            raise TTSError(f"unsupported ELEVENLABS_OUTPUT_FORMAT '{output_format}' (use mp3_*, wav_* or opus_*)")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.output_format = output_format
        self.audio_ext = prefix
        self.attempts = attempts
        self._session = session or requests.Session()
        self._sleep = sleep
        self._log = log

    def cache_identity(self) -> dict[str, Any]:
        return {"output_format": self.output_format}

    def _post(self, request: TTSRequest, with_context: bool) -> requests.Response:
        body: dict[str, Any] = {"text": request.text, "model_id": request.model_id}
        if request.voice_settings:
            body["voice_settings"] = request.voice_settings
        if with_context:
            if request.previous_text:
                body["previous_text"] = request.previous_text
            if request.next_text:
                body["next_text"] = request.next_text
        url = f"{self.base_url}/v1/text-to-speech/{request.voice_id}/with-timestamps"
        return send(self._session, "POST", url, service="ElevenLabs",
                    params={"output_format": self.output_format},
                    headers={"xi-api-key": self._api_key, "Content-Type": "application/json",
                             "Accept": "application/json"},
                    json=body)

    def synthesize(self, request: TTSRequest) -> TTSResult:
        from pipeline.logging_setup import get_logger

        log = self._log or get_logger("tts")

        def call(with_context: bool) -> requests.Response:
            return call_with_retries(lambda: self._post(request, with_context), what="ElevenLabs TTS",
                                     log=log, attempts=self.attempts, sleep=self._sleep)

        has_context = bool(request.previous_text or request.next_text)
        try:
            response = call(True)
        except APIError as exc:
            # Some models reject request stitching; fall back to a plain request once.
            if has_context and exc.status in (400, 422) and ("previous_text" in exc.body or "next_text" in exc.body):
                log.warning("model %s rejected previous_text/next_text — retrying without them", request.model_id)
                response = call(False)
            else:
                raise TTSError(str(exc)) from None
        try:
            payload = response.json()
            audio = base64.b64decode(payload["audio_base64"])
        except (ValueError, KeyError, TypeError) as exc:
            raise TTSError(f"ElevenLabs: unexpected response shape ({type(exc).__name__})") from None
        alignment = payload.get("alignment") or payload.get("normalized_alignment") or {}
        characters = list(alignment.get("characters") or [])
        starts = [float(x) for x in alignment.get("character_start_times_seconds") or []]
        ends = [float(x) for x in alignment.get("character_end_times_seconds") or []]
        if not (len(characters) == len(starts) == len(ends)):
            characters, starts, ends = [], [], []
        try:
            billed = int(response.headers.get("character-cost", ""))
        except (TypeError, ValueError):
            billed = len(request.text)
        return TTSResult(audio, characters, starts, ends, billed_characters=billed)


def make_tts_provider(ctx: RunContext) -> TTSProvider:
    if ctx.mock:
        return MockTTS(ctx.script.production.pacing.wpm)
    settings: Settings = ctx.settings
    return ElevenLabsTTS(settings.elevenlabs_api_key or "", base_url=settings.elevenlabs_base_url,
                         output_format=settings.elevenlabs_output_format, attempts=settings.http_attempts)


# --------------------------------------------------------------------------- #
# one line (scene or hook)
# --------------------------------------------------------------------------- #
def audio_cache_key(provider: TTSProvider, request: TTSRequest) -> str:
    """Cache key per the spec: text, voice_id, model_id, voice_settings (+ provider/format)."""
    return stable_hash({"provider": provider.name, "identity": provider.cache_identity(), "text": request.text,
                        "voice_id": request.voice_id, "model_id": request.model_id,
                        "voice_settings": request.voice_settings})


def _to_timings(words: list[Word]) -> list[WordTiming]:
    return [WordTiming(text=w.text, start=round(w.start, 4), end=round(w.end, 4)) for w in words]


def synthesize_line(ctx: RunContext, provider: TTSProvider, *, label: str, request: TTSRequest,
                    voiceover_text: str, subtitle_text: str | None, out_path: Path,
                    previous: TTSRecord | None) -> TTSRecord:
    """Produce (or reuse) the audio + word timings for one line and return its state record."""
    pronunciation = [(p.term, p.say_as) for p in ctx.script.production.pronunciation]
    audio_hash = audio_cache_key(provider, request)
    record_hash = stable_hash({"audio": audio_hash, "voiceover": voiceover_text, "subtitle": subtitle_text,
                               "pronunciation": pronunciation, "aligner": ALIGNER_VERSION})
    out_path = out_path.with_suffix(f".{provider.audio_ext}")
    if previous and previous.hash == record_hash and ctx.settings.abs(previous.audio_path).is_file():
        ctx.log.debug("tts %s: up to date", label)
        return previous

    cache_audio = ctx.settings.cache_dir / "tts" / f"{audio_hash}.{provider.audio_ext}"
    cache_meta = cache_audio.with_suffix(".json")
    from_cache = False
    if not ctx.no_cache and cache_audio.is_file() and cache_meta.is_file():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        result = TTSResult(cache_audio.read_bytes(), meta["characters"], meta["starts"], meta["ends"], 0)
        from_cache = True
    else:
        result = provider.synthesize(request)
        atomic_write_bytes(cache_audio, result.audio)
        atomic_write_bytes(cache_meta, json.dumps(
            {"characters": result.characters, "starts": result.starts, "ends": result.ends},
            ensure_ascii=False).encode("utf-8"))
    with ctx.store.lock:
        if from_cache:
            ctx.state.costs.tts_cache_hits += 1
        elif provider.name != "mock":
            ctx.state.costs.tts_requests += 1
            ctx.state.costs.tts_characters += result.billed_characters

    atomic_write_bytes(out_path, result.audio)
    duration = audio_seconds(out_path)
    spoken = chars_to_words(result.characters, result.starts, result.ends)
    if not spoken:
        ctx.log.warning("tts %s: no character alignment returned — word timings are estimated", label)
    # Timings can never point past the real audio.
    spoken = [Word(w.text, min(w.start, duration), min(w.end, duration)) for w in spoken]
    words = words_for_display(spoken, voiceover_text, subtitle_text, request.text, pronunciation, duration)
    ctx.log.info("tts %s: %.2f s, %d chars%s", label, duration, len(request.text), " (cache)" if from_cache else "")
    return TTSRecord(hash=record_hash, audio_hash=audio_hash, audio_path=ctx.settings.rel(out_path),
                     duration_sec=round(duration, 6), characters=len(request.text), provider=provider.name,
                     from_cache=from_cache, spoken_words=_to_timings(spoken), words=_to_timings(words))


def build_request(ctx: RunContext, text: str, previous_text: str | None, next_text: str | None) -> TTSRequest:
    tts = ctx.script.production.tts
    settings = tts.voice_settings.model_dump(exclude_none=True) if tts.voice_settings else {}
    stitch = tts.stitching == "previous_next_text"
    return TTSRequest(text=text, voice_id=tts.voice_id, model_id=tts.model_id, voice_settings=settings,
                      previous_text=previous_text if stitch else None, next_text=next_text if stitch else None)


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def update_master_timeline(ctx: RunContext) -> None:
    """Scene start/end on the master timeline -> state (later used to read retention graphs)."""
    lines: list[LineSpec] = []
    for section in ctx.script.sections:
        for index, scene in enumerate(section.scenes):
            record = ctx.state.scenes[scene.scene_id].tts
            if record is None:
                raise TTSError(f"{scene.scene_id}: no audio yet — run the tts stage first")
            is_last_in_section = index == len(section.scenes) - 1
            lines.append(LineSpec(scene.scene_id, record.duration_sec,
                                  SECTION_PAUSE_SEC if is_last_in_section else SCENE_PAUSE_SEC))
    slots = lay_out(lines, ctx.fps)
    for slot in slots:
        ctx.state.scenes[slot.key].timing = SceneTiming(
            start_frame=slot.start_frame, end_frame=slot.end_frame, start_sec=round(slot.start_sec, 6),
            end_sec=round(slot.end_sec, 6), voice_end_sec=round(slot.voice_end_sec, 6))
    total = slots[-1].end_frame
    ctx.state.timeline = TimelineInfo(fps=ctx.fps, total_frames=total, duration_sec=round(total / ctx.fps, 6))


def run_tts_stage(ctx: RunContext) -> None:
    provider = make_tts_provider(ctx)
    pronunciation = [(p.term, p.say_as) for p in ctx.script.production.pronunciation]
    scenes = ctx.script.scenes_in_order()
    # The pronunciation dictionary touches only what TTS reads; subtitles keep the original spelling.
    tts_texts = [apply_pronunciation(scene.voiceover_text, pronunciation) for _, scene in scenes]
    ctx.log.info("TTS: %d scenes via %s", len(scenes), provider.name)
    for index, (section, scene) in enumerate(scenes):
        entry = ctx.state.scene(scene.scene_id, section.section_id)
        request = build_request(ctx, tts_texts[index],
                                tts_texts[index - 1] if index > 0 else None,
                                tts_texts[index + 1] if index + 1 < len(scenes) else None)
        entry.tts = synthesize_line(ctx, provider, label=scene.scene_id, request=request,
                                    voiceover_text=scene.voiceover_text, subtitle_text=scene.subtitle_text,
                                    out_path=ctx.paths.audio / f"{scene.scene_id}.mp3", previous=entry.tts)
        ctx.save()  # after every scene: a crash resumes from here
    update_master_timeline(ctx)
    ctx.save()
    assert ctx.state.timeline is not None
    ctx.log.info("TTS done: %.1f s of narration timeline, %d characters billed so far",
                 ctx.state.timeline.duration_sec, ctx.state.costs.tts_characters)
