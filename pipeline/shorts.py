"""Stage 6 — vertical Shorts (9:16) cut from existing scenes.

Each Short has its own small timeline: optional hook line (separate TTS) first,
then the listed scenes back to back. Pictures are re-rendered from the scene
images straight into the vertical frame (one encode, no generation loss), with
burned-in word-by-word subtitles (ASS).
"""
from __future__ import annotations

from pathlib import Path

from pipeline.align import apply_pronunciation
from pipeline.audio import MusicSegment, SfxEvent, VoicePlacement
from pipeline.captions import EndCard, SubWord, render_ass
from pipeline.config import SCENE_PAUSE_SEC
from pipeline.context import RunContext
from pipeline.contract import Short
from pipeline.render import (ClipSpec, RenderError, ass_style, build_audio, concat_and_mux, render_clips,
                             resolve_music, resolve_sfx, scene_counter, scene_shot, scene_subwords,
                             transition_frames, verify_sync, RENDER_VERSION, require_inputs)
from pipeline.state import Artifact, ClipRecord, SceneTiming, ShortState, atomic_write_text, stable_hash
from pipeline.timeline import LineSpec, lay_out, samples_per_frame, seconds_to_frames
from pipeline.tts import TTSProvider, build_request, make_tts_provider, synthesize_line

END_CARD_SEC = 1.5
TAIL_SEC = 0.4
SHORT_WARN_SEC = 60.0
HOOK_KEY = "hook"


def short_frame_size(master: tuple[int, int]) -> tuple[int, int]:
    """Vertical frame for a 16:9 master: 1920x1080 -> 1080x1920."""
    w, h = master
    return (h, w) if w > h else (w, h)


def render_short(ctx: RunContext, short: Short, provider: TTSProvider | None) -> ShortState:
    st = ctx.state.shorts.get(short.short_id) or ShortState(short_id=short.short_id)
    ctx.state.shorts[short.short_id] = st
    st.status = "running"
    ctx.save()
    fps = ctx.fps
    spf = samples_per_frame(fps)
    scenes = [ctx.script.scene_by_id(scene_id) for scene_id in short.scene_ids]
    pronunciation = [(p.term, p.say_as) for p in ctx.script.production.pronunciation]

    # 1. hook line: its own TTS request, spoken before the first scene
    if short.hook_override_text:
        if provider is None:
            raise RenderError("hook line needs a TTS provider")
        first_text = apply_pronunciation(scenes[0].voiceover_text, pronunciation)
        request = build_request(ctx, apply_pronunciation(short.hook_override_text, pronunciation), None, first_text)
        st.hook = synthesize_line(ctx, provider, label=f"hook {short.short_id}", request=request,
                                  voiceover_text=short.hook_override_text, subtitle_text=None,
                                  out_path=ctx.paths.audio / f"hook_{short.short_id}.mp3", previous=st.hook)
        ctx.save()
    else:
        st.hook = None

    # 2. timeline of the short
    tail = END_CARD_SEC if short.end_card_text else TAIL_SEC
    lines: list[LineSpec] = []
    if st.hook is not None:
        lines.append(LineSpec(HOOK_KEY, st.hook.duration_sec, SCENE_PAUSE_SEC))
    for index, scene in enumerate(scenes):
        record = ctx.state.scenes[scene.scene_id].tts
        assert record is not None
        lines.append(LineSpec(f"{index:02d}_{scene.scene_id}", record.duration_sec,
                              tail if index == len(scenes) - 1 else SCENE_PAUSE_SEC))
    slots = lay_out(lines, fps)
    total_frames = slots[-1].end_frame
    total_sec = total_frames / fps
    scene_slots = [s for s in slots if s.key != HOOK_KEY]
    st.scenes = {slot.key: SceneTiming(start_frame=slot.start_frame, end_frame=slot.end_frame,
                                       start_sec=round(slot.start_sec, 6), end_sec=round(slot.end_sec, 6),
                                       voice_end_sec=round(slot.voice_end_sec, 6)) for slot in slots}
    if total_sec > SHORT_WARN_SEC:
        ctx.log.warning("short %s is %.1f s long — over the %d s Shorts limit", short.short_id, total_sec,
                        int(SHORT_WARN_SEC))

    work = ctx.paths.work / "shorts" / short.short_id
    frame = short_frame_size(ctx.frame_size)

    # 3. burned-in subtitles (+ end card)
    ass_file: Path | None = None
    burn = ctx.script.production.subtitles.burn_in.shorts
    if burn or short.end_card_text:
        sub_lines: list[list[SubWord]] = []
        if burn:
            if st.hook is not None:
                sub_lines.append([SubWord(w.text, w.start, w.end) for w in st.hook.words])
            for scene, slot in zip(scenes, scene_slots):
                sub_lines.append(scene_subwords(ctx, scene, slot.start_sec))
        end_card = None
        if short.end_card_text:
            end_card = EndCard(short.end_card_text, min(scene_slots[-1].voice_end_sec + 0.1, total_sec), total_sec)
        ass_file = work / "subs.ass"
        atomic_write_text(ass_file, render_ass(sub_lines, ass_style(ctx), frame, end_card, total_sec))

    # 4. clips: the first one also covers the hook
    clip_frames: list[int] = []
    clip_starts: list[int] = []
    for index, slot in enumerate(scene_slots):
        start = 0 if index == 0 else slot.start_frame
        clip_starts.append(start)
        clip_frames.append(slot.end_frame - start)
    trans = [0] + [transition_frames(scene.visual.transition_in, fps, n)
                   for scene, n in zip(scenes[1:], clip_frames[1:])]
    shots = [scene_shot(ctx, scene, clip_frames[i] + (trans[i + 1] if i + 1 < len(scenes) else 0))
             for i, scene in enumerate(scenes)]
    clips_dir = ctx.paths.clips / "shorts" / short.short_id
    specs: list[ClipSpec] = []
    for i, (scene, slot) in enumerate(zip(scenes, scene_slots)):
        specs.append(ClipSpec(
            key=slot.key, out_path=clips_dir / f"{slot.key}.mp4", frames=clip_frames[i], fps=fps, shot=shots[i],
            transition=scene.visual.transition_in if i > 0 else "cut", transition_frames=trans[i],
            prev_shot=shots[i - 1] if i > 0 else None, prev_offset=clip_frames[i - 1] if i > 0 else 0,
            counter=scene_counter(ctx, scene, clip_frames[i] - seconds_to_frames(SCENE_PAUSE_SEC, fps)),
            layout=short.reframe, out_size=frame, ass_file=ass_file, ass_offset_frames=clip_starts[i],
            fonts_dir=ctx.settings.assets_dir / "fonts"))

    def remember(key: str, record: ClipRecord) -> None:
        st.clips[key] = record

    records = render_clips(ctx, specs, dict(st.clips), remember)
    st.clips = {spec.key: records[spec.key] for spec in specs}

    # 5. audio: voice (hook + scenes), one music bed, the scenes' SFX
    voice: list[VoicePlacement] = []
    sfx: list[SfxEvent] = []
    if st.hook is not None:
        voice.append(VoicePlacement(ctx.settings.abs(st.hook.audio_path), 0))
    for scene, slot in zip(scenes, scene_slots):
        record = ctx.state.scenes[scene.scene_id].tts
        assert record is not None
        voice.append(VoicePlacement(ctx.settings.abs(record.audio_path), slot.start_frame * spf))
        if scene.sfx:
            path = resolve_sfx(ctx, scene.sfx)
            if path is not None:
                sfx.append(SfxEvent(path, slot.start_frame * spf))
    mood = ctx.script.mood_of(ctx.script.section_of(scenes[0].scene_id))
    music = [MusicSegment(mood, 0, total_frames * spf, resolve_music(ctx, mood))]
    stems = build_audio(ctx, voice, sfx, music, total_frames * spf, work / "audio", st.artifacts.get("stems"))
    st.artifacts["stems"] = stems

    # 6. join + mux
    out = ctx.paths.shorts / f"{short.short_id}.mp4"
    final_hash = stable_hash({"v": RENDER_VERSION, "clips": [records[s.key].hash for s in specs], "stems": stems.hash})
    previous = st.artifacts.get("final")
    if previous and previous.hash == final_hash and out.is_file():
        ctx.log.info("short %s: up to date", short.short_id)
    else:
        concat_and_mux([s.out_path for s in specs], work / "audio" / "mix.wav", out, work / "concat.txt")
        info = verify_sync(out, total_frames, fps)
        st.artifacts["final"] = Artifact(hash=final_hash, path=ctx.settings.rel(out), info=info)
        ctx.log.info("short %s: %s — %.1f s, %d frames (drift %.1f ms)", short.short_id, ctx.settings.rel(out),
                     total_sec, total_frames, info["drift_ms"])
    st.path = ctx.settings.rel(out)
    st.frames = total_frames
    st.duration_sec = round(total_sec, 4)
    st.status = "done"
    ctx.save()
    return st


def run_shorts_stage(ctx: RunContext) -> None:
    shorts = ctx.script.shorts
    if not shorts:
        ctx.log.info("shorts: none defined in the script")
        return
    require_inputs(ctx)
    wanted = {s.short_id for s in shorts}
    for stale in [key for key in ctx.state.shorts if key not in wanted]:
        del ctx.state.shorts[stale]
    provider = make_tts_provider(ctx) if any(s.hook_override_text for s in shorts) else None
    failures: list[str] = []
    for short in shorts:
        ctx.log.info("short %s: %d scenes, reframe=%s", short.short_id, len(short.scene_ids), short.reframe)
        try:
            render_short(ctx, short, provider)
        except Exception as exc:  # noqa: BLE001 — one broken Short must not block the others
            failures.append(f"{short.short_id}: {exc}")
            ctx.log.error("short %s failed: %s", short.short_id, str(exc)[:800])
            if short.short_id in ctx.state.shorts:
                ctx.state.shorts[short.short_id].status = "failed"
            ctx.save()
    if failures:
        raise RenderError(f"{len(failures)} short(s) failed; first: {failures[0][:600]}")
