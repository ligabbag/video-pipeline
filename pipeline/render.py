"""Stage 3 — FFmpeg render of the long video.

Design (why video and audio stay frame-exact):

* The master timeline is counted in frames (:mod:`pipeline.timeline`). Scene *i*
  owns exactly ``frames_i`` frames; the sum is the video length, and the audio
  stems are built to the same length in samples.
* Every scene is rendered as its own clip of exactly ``frames_i`` frames, then all
  clips are joined with the concat demuxer (stream copy, no re-encode).
* A transition never shortens anything. Classic ``xfade`` chains overlap clips
  and shift everything after them; here the outgoing picture is *extended* by the
  transition length instead (its camera move simply keeps going) and the blend
  is baked into the first frames of the incoming clip. Scene starts therefore
  stay exactly where the voice-over expects them.
* Ken Burns uses ``zoompan`` on a source upscaled ``supersample`` times in a
  4:4:4 pixel format, so the crop window moves in 1/supersample-pixel steps —
  that removes zoompan's well-known jitter (measured in tests/test_audio_render.py).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from pipeline import audio as audio_mod
from pipeline.align import Word
from pipeline.audio import MusicSegment, SfxEvent, VoicePlacement
from pipeline.captions import AssStyle, SubWord, build_cues, mark_emphasis, render_ass, render_srt, Cue
from pipeline.context import RunContext
from pipeline.contract import Scene
from pipeline.diagrams import DiagramStyle, is_animated, render_counter_sequence
from pipeline.ffmpeg import concat_list_entry, filter_escape, probe, run_ffmpeg
from pipeline.images import diagram_style
from pipeline.metadata import build_metadata
from pipeline.state import Artifact, ClipRecord, atomic_write_text, file_hash, stable_hash
from pipeline.timeline import samples_per_frame, seconds_to_frames

RENDER_VERSION = 5
TRANSITION_SEC = {"cut": 0.0, "crossfade": 0.4, "dip_to_black": 0.6, "whip": 0.2}
XFADE_NAME = {"crossfade": "fade", "dip_to_black": "fadeblack", "whip": "slideleft"}
DEFAULT_INTENSITY = 0.05
MAX_SUPERSAMPLED_WIDTH = 7680
# RGB stills are converted to BT.709 (the HD standard) and the stream is tagged accordingly;
# FFmpeg's default for RGB->YUV is BT.601, which players would show with shifted colours.
COLOR_MATRIX = "out_color_matrix=bt709:out_range=tv"
COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]


class RenderError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# clip description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Shot:
    """A still picture with a camera move spread over ``life_frames`` frames."""

    image: Path
    motion: str = "static"
    intensity: float = 0.0
    life_frames: int = 1


@dataclass(frozen=True)
class CounterAnim:
    params: dict[str, Any]
    style: DiagramStyle
    count_frames: int
    size: tuple[int, int] = (1920, 1080)   # frames are drawn at the master size, then scaled by the layout


@dataclass(frozen=True)
class ClipSpec:
    key: str
    out_path: Path
    frames: int
    fps: int
    shot: Shot
    transition: str = "cut"
    transition_frames: int = 0
    prev_shot: Shot | None = None
    prev_offset: int = 0                 # frame of prev_shot's life visible on this clip's first frame
    counter: CounterAnim | None = None
    layout: str = "landscape"            # landscape | blur_background | center_crop | pan_scan
    out_size: tuple[int, int] = (1920, 1080)
    ass_file: Path | None = None
    ass_offset_frames: int = 0
    fonts_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


def transition_frames(name: str, fps: int, clip_frames: int) -> int:
    """Length of a transition in frames, never more than half of the incoming clip."""
    wanted = seconds_to_frames(TRANSITION_SEC.get(name, 0.0), fps)
    return max(0, min(wanted, clip_frames // 2))


# --------------------------------------------------------------------------- #
# filter graph
# --------------------------------------------------------------------------- #
def motion_expressions(shot: Shot, offset: int) -> tuple[str, str, str]:
    """zoompan ``z``/``x``/``y`` expressions for output frame ``on`` (+ ``offset`` into the shot's life)."""
    amount = shot.intensity if shot.intensity > 0 else DEFAULT_INTENSITY
    denom = max(shot.life_frames - 1, 1)
    p = f"min(1,(on+{offset})/{denom})"
    cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    if shot.motion == "zoom_in":
        return f"1+{amount:.5f}*{p}", cx, cy
    if shot.motion == "zoom_out":
        return f"1+{amount:.5f}*(1-{p})", cx, cy
    z = f"{1 + amount:.5f}"  # pans: zoomed in just enough to travel `amount` of the frame
    if shot.motion == "pan_right":
        return z, f"(iw-iw/zoom)*{p}", cy
    if shot.motion == "pan_left":
        return z, f"(iw-iw/zoom)*(1-{p})", cy
    if shot.motion == "pan_down":
        return z, cx, f"(ih-ih/zoom)*{p}"
    if shot.motion == "pan_up":
        return z, cx, f"(ih-ih/zoom)*(1-{p})"
    return "1", "0", "0"


def shot_chain(shot: Shot, offset: int, frames: int, fps: int, inner: tuple[int, int], supersample: int) -> str:
    """Filter chain turning one still image into ``frames`` frames of ``inner`` size."""
    w, h = inner
    # zoompan already emits exactly `frames` frames at `fps`; only normalise format and timestamps.
    tail = f"format=yuv420p,settb=1/{fps},setpts=N"
    if shot.motion == "static":
        return (f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos:{COLOR_MATRIX},crop={w}:{h},"
                f"zoompan=z=1:x=0:y=0:d={frames}:s={w}x{h}:fps={fps},{tail}")
    ss = max(1, min(supersample, MAX_SUPERSAMPLED_WIDTH // w))
    z, x, y = motion_expressions(shot, offset)
    # yuv444p on purpose: zoompan snaps the crop origin to the chroma grid, so with 4:2:0 input it
    # moves in 2-pixel steps. 4:4:4 + supersampling gives steps of 1/ss output pixel.
    return (f"scale={w * ss}:{h * ss}:force_original_aspect_ratio=increase:flags=lanczos:{COLOR_MATRIX},"
            f"crop={w * ss}:{h * ss},format=yuv444p,zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={w}x{h}:fps={fps},{tail}")


def inner_size(layout: str, out_size: tuple[int, int]) -> tuple[int, int]:
    """Size of the 16:9 picture rendered before the layout step."""
    ow, oh = out_size
    if layout == "landscape":
        return out_size
    if layout == "blur_background":
        return ow, int(round(ow * 9 / 16 / 2)) * 2
    return int(round(oh * 16 / 9 / 2)) * 2, oh  # center_crop / pan_scan: fill the height


def layout_chain(layout: str, out_size: tuple[int, int], frames: int) -> list[str]:
    """Filters from ``[v0]`` (inner picture) to ``[v1]`` (final frame, before subtitles)."""
    ow, oh = out_size
    if layout == "landscape":
        return ["[v0]null[v1]"]
    if layout == "blur_background":
        sw, sh = ow // 4, oh // 4
        return [
            "[v0]split=2[fg][bg0]",
            f"[bg0]scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},gblur=sigma=9,"
            f"scale={ow}:{oh}:flags=bicubic,eq=brightness=-0.10:saturation=0.85[bg]",
            "[bg][fg]overlay=(W-w)/2:(H-h)/2:format=auto,format=yuv420p[v1]",
        ]
    if layout == "center_crop":
        return [f"[v0]crop={ow}:{oh}[v1]"]
    if layout == "pan_scan":
        return [f"[v0]crop={ow}:{oh}:x='(iw-{ow})*n/{max(frames - 1, 1)}':y=0[v1]"]
    raise RenderError(f"unknown layout '{layout}'")


def encoder_args(hw: bool) -> list[str]:
    """Video encoder options. ``--hw`` uses VideoToolbox on macOS, otherwise libx264 CRF 20."""
    if hw and platform.system() == "Darwin":
        quality = ["-q:v", "62"] if platform.machine() == "arm64" else ["-b:v", "10M"]
        return ["-c:v", "h264_videotoolbox", *quality, "-allow_sw", "1", "-pix_fmt", "yuv420p", *COLOR_TAGS]
    return ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", *COLOR_TAGS]


def build_clip_command(spec: ClipSpec, supersample: int, hw: bool, counter_dir: Path | None,
                       tmp_out: Path) -> tuple[list[str], Path | None]:
    """FFmpeg arguments for one clip, and the working directory to run in (for the ASS file)."""
    fps = spec.fps
    inner = inner_size(spec.layout, spec.out_size)
    args: list[str] = []
    graph: list[str] = []

    if spec.counter is not None and counter_dir is not None:
        args += ["-framerate", str(fps), "-start_number", "0", "-i", str(counter_dir / "%05d.png")]
        graph.append(f"[0:v]scale={inner[0]}:{inner[1]}:flags=lanczos:{COLOR_MATRIX},fps={fps},format=yuv420p,"
                     f"settb=1/{fps},setpts=N[cur]")
    else:
        args += ["-i", str(spec.shot.image)]
        graph.append(f"[0:v]{shot_chain(spec.shot, 0, spec.frames, fps, inner, supersample)}[cur]")

    t = spec.transition_frames
    if t > 0 and spec.prev_shot is not None:
        args += ["-i", str(spec.prev_shot.image)]
        graph.append(f"[1:v]{shot_chain(spec.prev_shot, spec.prev_offset, t, fps, inner, supersample)}[prev]")
        graph.append(f"[prev][cur]xfade=transition={XFADE_NAME[spec.transition]}:duration={t / fps:.6f}:offset=0,"
                     f"format=yuv420p[v0]")
    elif t > 0:  # first clip of a timeline: fade up from black
        graph.append(f"[cur]fade=t=in:st=0:d={t / fps:.6f}[v0]")
    else:
        graph.append("[cur]null[v0]")

    graph += layout_chain(spec.layout, spec.out_size, spec.frames)

    cwd: Path | None = None
    if spec.ass_file is not None:
        cwd = spec.ass_file.parent
        fonts = ""
        if spec.fonts_dir is not None and spec.fonts_dir.is_dir() and any(spec.fonts_dir.iterdir()):
            fonts = f":fontsdir={filter_escape(spec.fonts_dir.resolve())}"
        k = spec.ass_offset_frames
        graph.append(f"[v1]settb=1/{fps},setpts=PTS+{k},subtitles=filename={filter_escape(spec.ass_file.name)}{fonts},"
                     f"setpts=PTS-{k}[v]")
    else:
        graph.append("[v1]null[v]")

    args += ["-filter_complex", ";".join(graph), "-map", "[v]", "-frames:v", str(spec.frames), "-an",
             *encoder_args(hw), str(tmp_out.resolve())]
    return args, cwd


# --------------------------------------------------------------------------- #
# hashing + execution
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4096)
def _file_hash_cached(path: str, mtime_ns: int, size: int) -> str:
    return file_hash(Path(path))


def content_hash(path: Path) -> str:
    stat = path.stat()
    return _file_hash_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _shot_key(shot: Shot | None) -> Any:
    if shot is None:
        return None
    return [content_hash(shot.image), shot.motion, round(shot.intensity, 5), shot.life_frames]


def clip_hash(spec: ClipSpec, supersample: int, hw: bool) -> str:
    return stable_hash({
        "v": RENDER_VERSION, "frames": spec.frames, "fps": spec.fps, "shot": _shot_key(spec.shot),
        "transition": [spec.transition, spec.transition_frames],
        "prev": [_shot_key(spec.prev_shot), spec.prev_offset] if spec.transition_frames else None,
        "counter": [spec.counter.params, spec.counter.style.__dict__, spec.counter.count_frames] if spec.counter else None,
        "layout": spec.layout, "size": spec.out_size, "ss": supersample, "encoder": encoder_args(hw),
        "ass": [content_hash(spec.ass_file), spec.ass_offset_frames] if spec.ass_file else None,
    })


def render_clip(spec: ClipSpec, supersample: int, hw: bool, work_dir: Path) -> int:
    """Render one clip and verify it has exactly ``spec.frames`` frames."""
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = spec.out_path.with_name(spec.out_path.stem + ".part.mp4")
    counter_dir: Path | None = None
    try:
        if spec.counter is not None:
            counter_dir = work_dir / "frames" / spec.out_path.parent.name / spec.key
            shutil.rmtree(counter_dir, ignore_errors=True)
            render_counter_sequence(spec.counter.params, spec.counter.style, spec.counter.size,
                                    total_frames=spec.frames, count_frames=spec.counter.count_frames,
                                    out_dir=counter_dir)
        args, cwd = build_clip_command(spec, supersample, hw, counter_dir, tmp_out)
        run_ffmpeg(args, cwd=cwd)
        frames = probe(tmp_out).video_frames
        if frames != spec.frames:
            raise RenderError(f"clip {spec.key}: rendered {frames} frames, expected {spec.frames}")
        os.replace(tmp_out, spec.out_path)
        return spec.frames
    finally:
        tmp_out.unlink(missing_ok=True)
        if counter_dir is not None:
            shutil.rmtree(counter_dir, ignore_errors=True)


def render_clips(ctx: RunContext, specs: Sequence[ClipSpec], existing: dict[str, ClipRecord | None],
                 on_done: Any) -> dict[str, ClipRecord]:
    """Render what is missing or stale (in parallel) and return a record per clip key."""
    supersample = ctx.settings.supersample
    records: dict[str, ClipRecord] = {}
    todo: list[tuple[ClipSpec, str]] = []
    for spec in specs:
        digest = clip_hash(spec, supersample, ctx.hw)
        old = existing.get(spec.key)
        if old and old.hash == digest and spec.out_path.is_file():
            records[spec.key] = old
        else:
            todo.append((spec, digest))
    ctx.log.info("clips: %d to render, %d up to date", len(todo), len(records))
    if not todo:
        return records

    def work(spec: ClipSpec, digest: str) -> ClipRecord:
        frames = render_clip(spec, supersample, ctx.hw, ctx.paths.work)
        return ClipRecord(hash=digest, path=ctx.settings.rel(spec.out_path), frames=frames)

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=ctx.settings.render_workers) as pool:
        futures = {pool.submit(work, spec, digest): spec for spec, digest in todo}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001 — finish the others, then fail the stage
                errors.append(f"{spec.key}: {exc}")
                ctx.log.error("clip %s failed: %s", spec.key, str(exc)[:800])
                continue
            records[spec.key] = record
            ctx.log.info("clip %s: %d frames", spec.key, record.frames)
            with ctx.store.lock:
                on_done(spec.key, record)
                ctx.save()
    if errors:
        raise RenderError(f"{len(errors)} clip(s) failed; first: {errors[0][:600]}")
    return records


def concat_and_mux(clips: Sequence[Path], audio: Path, out: Path, list_file: Path) -> None:
    """Join clips without re-encoding and add the final mix as AAC 192k, moov atom first."""
    list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text("".join(concat_list_entry(c) for c in clips), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp])
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def verify_sync(path: Path, expected_frames: int, fps: int) -> dict[str, Any]:
    """Frame count must match the timeline and A/V durations may differ by at most one frame."""
    info = probe(path)
    if info.video_frames != expected_frames:
        raise RenderError(f"{path.name}: {info.video_frames} video frames, timeline has {expected_frames}")
    v_dur = info.video_duration if info.video_duration is not None else expected_frames / fps
    a_dur = info.audio_duration if info.audio_duration is not None else 0.0
    drift = abs(v_dur - a_dur)
    if drift > 1.0 / fps + 1e-6:
        raise RenderError(f"{path.name}: audio/video durations differ by {drift * 1000:.1f} ms (> 1 frame)")
    return {"frames": info.video_frames, "video_sec": round(v_dur, 4), "audio_sec": round(a_dur, 4),
            "drift_ms": round(drift * 1000, 2)}


# --------------------------------------------------------------------------- #
# shared builders (also used by shorts)
# --------------------------------------------------------------------------- #
def scene_shot(ctx: RunContext, scene: Scene, life_frames: int) -> Shot:
    record = ctx.state.scenes[scene.scene_id].image
    if record is None:
        raise RenderError(f"{scene.scene_id}: no image yet — run the images stage first")
    path = ctx.settings.abs(record.path)
    if not path.is_file():
        raise RenderError(f"{scene.scene_id}: image file is missing ({record.path}) — re-run the images stage")
    motion = scene.visual.motion
    intensity = float(motion.intensity) if motion.intensity is not None else DEFAULT_INTENSITY
    return Shot(path, motion.type, 0.0 if motion.type == "static" else intensity, max(life_frames, 1))


def scene_counter(ctx: RunContext, scene: Scene, count_frames: int) -> CounterAnim | None:
    diagram = scene.visual.diagram
    if scene.visual.source != "diagram" or diagram is None or not is_animated(diagram.template):
        return None
    return CounterAnim(dict(diagram.params), diagram_style(ctx), max(count_frames, 2), ctx.frame_size)


def scene_subwords(ctx: RunContext, scene: Scene, offset_sec: float) -> list[SubWord]:
    """A scene's subtitle words on an absolute timeline, with emphasis flags."""
    record = ctx.state.scenes[scene.scene_id].tts
    if record is None:
        raise RenderError(f"{scene.scene_id}: no audio yet — run the tts stage first")
    flags = mark_emphasis([w.text for w in record.words], scene.emphasis_words)
    return [SubWord(w.text, offset_sec + w.start, offset_sec + w.end, flag) for w, flag in zip(record.words, flags)]


def ass_style(ctx: RunContext) -> AssStyle:
    subs = ctx.script.production.subtitles
    return AssStyle(font=subs.font, base_color=subs.base_color or "#FFFFFF", highlight_color=subs.highlight_color,
                    position=subs.position, dynamic=subs.style == "yellow_dynamic",
                    max_words_per_chunk=subs.max_words_per_chunk)


def resolve_sfx(ctx: RunContext, name: str) -> Path | None:
    found = audio_mod.find_sfx(ctx.settings.assets_dir, name)
    if found is None and ctx.mock:
        return audio_mod.synth_sfx(ctx.paths.work / "mock_assets" / f"sfx_{name}.wav", name)
    if found is None:
        ctx.log.warning("SFX '%s' not found (expected assets/sfx/%s.wav) — skipped", name, name)
    return found


def resolve_music(ctx: RunContext, mood: str) -> Path | None:
    found = audio_mod.pick_music_track(ctx.settings.assets_dir, mood, f"{ctx.script.video_id}:{mood}")
    if found is None and ctx.mock:
        return audio_mod.synth_music(ctx.paths.work / "mock_assets" / f"music_{mood}.wav", mood)
    if found is None:
        ctx.log.warning("no music in assets/music/%s/ — that part of the video will have no music", mood)
    return found


def stems_hash(ctx: RunContext, voice: Sequence[VoicePlacement], sfx: Sequence[SfxEvent],
               music: Sequence[MusicSegment], total_samples: int) -> str:
    return stable_hash({
        "v": audio_mod.AUDIO_VERSION, "total": total_samples,
        "voice": [[content_hash(v.path), v.start_sample] for v in voice],
        "sfx": [[content_hash(s.path), s.start_sample] for s in sfx],
        "music": [[m.mood, m.start_sample, m.end_sample, content_hash(m.track) if m.track else None] for m in music],
        "ducking": ctx.script.production.music.ducking_db,
        "gains": [ctx.settings.music_gain_db, ctx.settings.sfx_gain_db, ctx.settings.target_lufs,
                  ctx.settings.true_peak_db],
    })


def build_audio(ctx: RunContext, voice: Sequence[VoicePlacement], sfx: Sequence[SfxEvent],
                music: Sequence[MusicSegment], total_samples: int, out_dir: Path,
                previous: Artifact | None) -> Artifact:
    """Stems + mix into ``out_dir`` unless an identical build is already there."""
    digest = stems_hash(ctx, voice, sfx, music, total_samples)
    names = ("voice.wav", "music.wav", "sfx.wav", "mix.wav")
    if previous and previous.hash == digest and all((out_dir / n).is_file() for n in names):
        ctx.log.info("audio: stems and mix up to date")
        return previous
    report = audio_mod.build_all(
        voice=voice, sfx=sfx, music=music, total_samples=total_samples, out_dir=out_dir,
        work_dir=ctx.paths.work / "audio", ducking_db=ctx.script.production.music.ducking_db,
        music_gain_db=ctx.settings.music_gain_db, sfx_gain_db=ctx.settings.sfx_gain_db,
        target_lufs=ctx.settings.target_lufs, true_peak_db=ctx.settings.true_peak_db, log=ctx.log)
    return Artifact(hash=digest, path=ctx.settings.rel(out_dir), info=dict(report))


# --------------------------------------------------------------------------- #
# long video
# --------------------------------------------------------------------------- #
def require_inputs(ctx: RunContext) -> None:
    missing_tts = [sid for sid in ctx.script.scene_ids()
                   if sid not in ctx.state.scenes or ctx.state.scenes[sid].tts is None
                   or ctx.state.scenes[sid].timing is None]
    missing_img = [sid for sid in ctx.script.scene_ids()
                   if sid not in ctx.state.scenes or ctx.state.scenes[sid].image is None]
    if missing_tts:
        raise RenderError(f"no audio/timing for {', '.join(missing_tts[:5])}… — run the tts stage first")
    if missing_img:
        raise RenderError(f"no image for {', '.join(missing_img[:5])}… — run the images stage first")
    if ctx.state.timeline is None or ctx.state.timeline.fps != ctx.fps:
        raise RenderError("timeline is missing or was built for another fps — re-run the tts stage")


def master_srt_cues(ctx: RunContext) -> list[Cue]:
    cues: list[Cue] = []
    for _, scene in ctx.script.scenes_in_order():
        entry = ctx.state.scenes[scene.scene_id]
        assert entry.tts is not None and entry.timing is not None
        words = [Word(w.text, entry.timing.start_sec + w.start, entry.timing.start_sec + w.end) for w in entry.tts.words]
        cues.extend(build_cues(words, limit=entry.timing.end_sec))
    fixed: list[Cue] = []
    for index, cue in enumerate(cues):  # no overlaps across scene boundaries
        end = cue.end
        if index + 1 < len(cues) and end > cues[index + 1].start - 0.04:
            end = max(cue.start + 0.05, cues[index + 1].start - 0.04)
        fixed.append(Cue(cue.start, end, cue.lines))
    return fixed


def build_long_specs(ctx: RunContext, ass_file: Path | None) -> list[ClipSpec]:
    scenes = [scene for _, scene in ctx.script.scenes_in_order()]
    fps = ctx.fps
    frames = []
    for scene in scenes:
        timing = ctx.state.scenes[scene.scene_id].timing
        assert timing is not None
        frames.append(timing.end_frame - timing.start_frame)
    trans = [transition_frames(scene.visual.transition_in, fps, n) for scene, n in zip(scenes, frames)]
    shots = [scene_shot(ctx, scene, frames[i] + (trans[i + 1] if i + 1 < len(scenes) else 0))
             for i, scene in enumerate(scenes)]
    fonts_dir = ctx.settings.assets_dir / "fonts"
    specs: list[ClipSpec] = []
    for i, scene in enumerate(scenes):
        entry = ctx.state.scenes[scene.scene_id]
        assert entry.timing is not None and entry.tts is not None
        specs.append(ClipSpec(
            key=scene.scene_id, out_path=ctx.paths.clips / f"{scene.scene_id}.mp4", frames=frames[i], fps=fps,
            shot=shots[i], transition=scene.visual.transition_in, transition_frames=trans[i],
            prev_shot=shots[i - 1] if i > 0 else None, prev_offset=frames[i - 1] if i > 0 else 0,
            counter=scene_counter(ctx, scene, seconds_to_frames(entry.tts.duration_sec, fps)),
            layout="landscape", out_size=ctx.frame_size, ass_file=ass_file,
            ass_offset_frames=entry.timing.start_frame, fonts_dir=fonts_dir))
    return specs


def build_long_audio_plan(ctx: RunContext) -> tuple[list[VoicePlacement], list[SfxEvent], list[MusicSegment], int]:
    spf = samples_per_frame(ctx.fps)
    assert ctx.state.timeline is not None
    total_samples = ctx.state.timeline.total_frames * spf
    voice: list[VoicePlacement] = []
    sfx: list[SfxEvent] = []
    music: list[MusicSegment] = []
    for section in ctx.script.sections:
        first = ctx.state.scenes[section.scenes[0].scene_id].timing
        last = ctx.state.scenes[section.scenes[-1].scene_id].timing
        assert first is not None and last is not None
        mood = ctx.script.mood_of(section)
        if music and music[-1].mood == mood:  # same mood continues: one uninterrupted segment
            music[-1] = MusicSegment(mood, music[-1].start_sample, last.end_frame * spf, music[-1].track)
        else:
            music.append(MusicSegment(mood, first.start_frame * spf, last.end_frame * spf, resolve_music(ctx, mood)))
        for scene in section.scenes:
            entry = ctx.state.scenes[scene.scene_id]
            assert entry.tts is not None and entry.timing is not None
            voice.append(VoicePlacement(ctx.settings.abs(entry.tts.audio_path), entry.timing.start_frame * spf))
            if scene.sfx:
                path = resolve_sfx(ctx, scene.sfx)
                if path is not None:
                    sfx.append(SfxEvent(path, entry.timing.start_frame * spf))
    return voice, sfx, music, total_samples


def run_render_stage(ctx: RunContext) -> None:
    require_inputs(ctx)
    assert ctx.state.timeline is not None
    timeline = ctx.state.timeline
    fps = ctx.fps
    ctx.log.info("render: %d scenes, %d frames (%.1f s) at %d fps", len(ctx.script.scene_ids()),
                 timeline.total_frames, timeline.duration_sec, fps)

    # captions: always exported; burned into the long video only on request
    srt_path = ctx.paths.captions / f"{ctx.script.language.split('-')[0]}.srt"
    atomic_write_text(srt_path, render_srt(master_srt_cues(ctx)))
    ctx.log.info("captions: %s", ctx.settings.rel(srt_path))

    ass_file: Path | None = None
    if ctx.script.production.subtitles.burn_in.long:
        lines = []
        for _, scene in ctx.script.scenes_in_order():
            timing = ctx.state.scenes[scene.scene_id].timing
            assert timing is not None
            lines.append(scene_subwords(ctx, scene, timing.start_sec))
        ass_file = ctx.paths.work / "long.ass"
        atomic_write_text(ass_file, render_ass(lines, ass_style(ctx), ctx.frame_size, total_sec=timeline.duration_sec))
        if ctx.script.localization.custom_tracks:
            ctx.log.warning("burn_in.long=true with dubbed tracks planned: subtitles will be burned in English")

    # audio: stems + mix
    voice, sfx, music, total_samples = build_long_audio_plan(ctx)
    stems = build_audio(ctx, voice, sfx, music, total_samples, ctx.paths.stems, ctx.state.artifacts.get("stems"))
    ctx.state.artifacts["stems"] = stems
    ctx.save()

    # video: one clip per scene, then a lossless join
    specs = build_long_specs(ctx, ass_file)
    existing = {sid: ctx.state.scenes[sid].clip for sid in ctx.script.scene_ids()}

    def remember(key: str, record: ClipRecord) -> None:
        ctx.state.scenes[key].clip = record

    records = render_clips(ctx, specs, existing, remember)
    out = ctx.paths.final / "long.mp4"
    final_hash = stable_hash({"v": RENDER_VERSION, "clips": [records[s.key].hash for s in specs], "stems": stems.hash})
    previous = ctx.state.artifacts.get("long")
    if previous and previous.hash == final_hash and out.is_file():
        ctx.log.info("final: long.mp4 up to date")
    else:
        concat_and_mux([s.out_path for s in specs], ctx.paths.stems / "mix.wav", out, ctx.paths.work / "long.concat.txt")
        info = verify_sync(out, timeline.total_frames, fps)
        ctx.state.artifacts["long"] = Artifact(hash=final_hash, path=ctx.settings.rel(out), info=info)
        ctx.log.info("final: %s — %d frames, video %.3f s, audio %.3f s (drift %.1f ms)", ctx.settings.rel(out),
                     info["frames"], info["video_sec"], info["audio_sec"], info["drift_ms"])

    # metadata.json
    section_starts = []
    for section in ctx.script.sections:
        timing = ctx.state.scenes[section.scenes[0].scene_id].timing
        assert timing is not None
        section_starts.append((section.chapter_title, timing.start_sec))
    metadata = build_metadata(ctx.script, section_starts, timeline.duration_sec)
    if len(metadata["chapters"]) < 3:
        ctx.log.warning("only %d chapter(s) after merging short ones — YouTube needs at least 3 to show chapters",
                        len(metadata["chapters"]))
    atomic_write_text(ctx.paths.metadata, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    ctx.state.artifacts["metadata"] = Artifact(hash=stable_hash(metadata), path=ctx.settings.rel(ctx.paths.metadata))
    ctx.save()
