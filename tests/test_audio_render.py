"""Audio mixing (ducking, loudness, exact lengths) and frame-exact video rendering."""
from __future__ import annotations

import array
import logging
import math
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pipeline import audio
from pipeline.audio import MusicSegment, SfxEvent, VoicePlacement
from pipeline.ffmpeg import filter_escape, probe, run_ffmpeg
from pipeline.render import (ClipSpec, Shot, concat_and_mux, inner_size, motion_expressions, render_clip,
                             transition_frames, verify_sync, RenderError)

RATE = audio.AUDIO_RATE
LOG = logging.getLogger("test")


def rms_db(pcm: bytes, start: float, end: float) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[int(start * RATE) * 4:int(end * RATE) * 4])
    mean = sum(s * s for s in samples) / max(len(samples), 1)
    return 10 * math.log10(mean / (32768.0 ** 2) + 1e-12)


def tone(path: Path, freq: float, seconds: float, volume_db: float, extra: str = "") -> Path:
    af = f"volume={volume_db}dB" + (f",{extra}" if extra else "")
    run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={RATE}:duration={seconds}", "-af", af,
                "-ac", "2", "-c:a", "pcm_s16le", path])
    return path


# ---- audio ------------------------------------------------------------------------
@pytest.mark.parametrize("ducking_db", [-18.0, -9.0])
def test_music_is_ducked_by_about_ducking_db_under_voice(tmp_path: Path, ducking_db: float) -> None:
    voice = tone(tmp_path / "voice.wav", 1000, 5, -20, "afade=t=out:st=2:d=0.01,volume=enable='gt(t,2.02)':volume=0")
    music = tone(tmp_path / "music.wav", 220, 5, -20)
    out = tmp_path / "ducked.wav"
    run_ffmpeg(["-i", music, "-i", voice, "-filter_complex", f"[0:a][1:a]{audio.ducking_filter(ducking_db)}[o]",
                "-map", "[o]", "-c:a", "pcm_s16le", out])
    pcm = audio.decode_pcm(out)
    under_voice, alone = rms_db(pcm, 0.8, 1.9), rms_db(pcm, 4.0, 4.9)
    original = rms_db(audio.decode_pcm(music), 4.0, 4.9)
    assert alone == pytest.approx(original, abs=1.0)          # back to full level once the voice is gone
    assert under_voice - alone == pytest.approx(ducking_db, abs=2.5)


def test_stems_are_sample_exact_and_mix_hits_loudness_target(tmp_path: Path) -> None:
    speech = tmp_path / "speech.wav"
    run_ffmpeg(["-f", "lavfi", "-i", f"anoisesrc=color=pink:sample_rate={RATE}:duration=4:seed=7", "-af",
                "volume=-12dB,tremolo=f=4:d=0.9", "-ac", "2", "-c:a", "pcm_s16le", speech])
    track = audio.synth_music(tmp_path / "music_src.wav", "tense_minimal", seconds=3)   # shorter than needed: loops
    beep = audio.synth_sfx(tmp_path / "beep.wav", "low_boom")
    fps, frames = 30, 361
    total = frames * (RATE // fps)
    placements = [VoicePlacement(speech, 0), VoicePlacement(speech, 5 * RATE), VoicePlacement(speech, 9 * RATE)]
    segments = [MusicSegment("tense_minimal", 0, 6 * RATE, track), MusicSegment("dread_drone", 6 * RATE, total, track)]
    report = audio.build_all(voice=placements, sfx=[SfxEvent(beep, RATE), SfxEvent(beep, 11 * RATE)], music=segments,
                             total_samples=total, out_dir=tmp_path / "stems", work_dir=tmp_path / "work",
                             ducking_db=-18, music_gain_db=-6, sfx_gain_db=-3, target_lufs=-14, true_peak_db=-1,
                             log=LOG)
    for name in ("voice.wav", "music.wav", "sfx.wav", "mix.wav"):
        assert audio.wav_samples(tmp_path / "stems" / name) == total
    assert report["integrated_lufs"] == pytest.approx(-14.0, abs=0.7)
    assert report["true_peak_db"] <= -0.8
    voice_pcm = audio.decode_pcm(tmp_path / "stems" / "voice.wav")
    assert rms_db(voice_pcm, 4.2, 4.9) < -80 and rms_db(voice_pcm, 5.1, 5.9) > -40       # placed exactly at 5 s
    music_pcm = audio.decode_pcm(tmp_path / "stems" / "music.wav")
    assert rms_db(music_pcm, 10.0, 11.0) > -50                                           # looped to the end
    sfx_pcm = audio.decode_pcm(tmp_path / "stems" / "sfx.wav")
    assert rms_db(sfx_pcm, 0.5, 0.95) < -80 and rms_db(sfx_pcm, 1.0, 1.3) > -40


def test_silent_mix_is_not_normalised(tmp_path: Path) -> None:
    total = RATE * 2
    for name in ("v.wav", "m.wav", "s.wav"):
        audio.write_silence(tmp_path / name, total)
    report = audio.mix_stems(tmp_path / "v.wav", tmp_path / "m.wav", tmp_path / "s.wav", tmp_path / "mix.wav",
                             ducking_db=-18, total_samples=total, work_dir=tmp_path / "w")
    assert report["silent"] is True and audio.wav_samples(tmp_path / "mix.wav") == total


def test_asset_lookup(tmp_path: Path) -> None:
    (tmp_path / "music" / "dread_drone").mkdir(parents=True)
    (tmp_path / "sfx").mkdir()
    assert audio.pick_music_track(tmp_path, "dread_drone", "k") is None
    for name in ("b.mp3", "a.wav", "notes.txt"):
        (tmp_path / "music" / "dread_drone" / name).write_bytes(b"x")
    picked = audio.pick_music_track(tmp_path, "dread_drone", "video:dread_drone")
    assert picked is not None and picked.name in ("a.wav", "b.mp3")
    assert picked == audio.pick_music_track(tmp_path, "dread_drone", "video:dread_drone")   # deterministic
    (tmp_path / "sfx" / "whoosh.wav").write_bytes(b"x")
    assert audio.find_sfx(tmp_path, "whoosh") == tmp_path / "sfx" / "whoosh.wav"
    assert audio.find_sfx(tmp_path, "typing") is None


# ---- video ------------------------------------------------------------------------
SIZE = (640, 360)


def solid(path: Path, color: str) -> Path:
    Image.new("RGB", (800, 450), color).save(path)
    return path


def frame_rgb(video: Path, index: int) -> tuple[int, int, int]:
    """Average colour of one frame, decoded the way a player decodes HD video (BT.709, limited range).

    The YUV -> RGB step is done here on purpose: FFmpeg's implicit conversion assumes BT.601.
    """
    proc = run_ffmpeg(["-i", video, "-vf", f"select='eq(n,{index})',scale=1:1,format=yuv444p", "-frames:v", "1",
                       "-f", "rawvideo", "-"], capture_stdout=True)
    y, cb, cr = ((proc.stdout[0] - 16) / 219, (proc.stdout[1] - 128) / 224, (proc.stdout[2] - 128) / 224)
    rgb = (y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb)
    r, g, b = (max(0, min(255, round(v * 255))) for v in rgb)
    return r, g, b


def test_transition_lengths() -> None:
    assert transition_frames("cut", 30, 150) == 0
    assert transition_frames("crossfade", 30, 150) == 12
    assert transition_frames("dip_to_black", 25, 150) == 15
    assert transition_frames("whip", 24, 150) == 5
    assert transition_frames("crossfade", 30, 10) == 5                  # never more than half the clip
    assert inner_size("blur_background", (1080, 1920)) == (1080, 608)
    assert inner_size("center_crop", (1080, 1920)) == (3414, 1920)


@pytest.mark.parametrize("fps", [30, 24])
def test_transitions_do_not_shift_sync(tmp_path: Path, fps: int) -> None:
    """Clips keep their exact frame counts with xfade baked in; the join matches the audio to the frame."""
    red, green, blue = (solid(tmp_path / f"{n}.png", c) for n, c in (("r", "#ff0000"), ("g", "#00ff00"), ("b", "#0000ff")))
    frames = [40, 35, 31]
    t2, t3 = transition_frames("crossfade", fps, frames[1]), transition_frames("whip", fps, frames[2])
    shots = [Shot(red, "zoom_in", 0.08, frames[0] + t2), Shot(green, "pan_left", 0.1, frames[1] + t3),
             Shot(blue, "static", 0.0, frames[2])]
    specs = [
        ClipSpec("a", tmp_path / "a.mp4", frames[0], fps, shots[0], "dip_to_black", 6, out_size=SIZE),
        ClipSpec("b", tmp_path / "b.mp4", frames[1], fps, shots[1], "crossfade", t2, shots[0], frames[0], out_size=SIZE),
        ClipSpec("c", tmp_path / "c.mp4", frames[2], fps, shots[2], "whip", t3, shots[1], frames[1], out_size=SIZE),
    ]
    for spec in specs:
        assert render_clip(spec, 2, False, tmp_path / "work") == spec.frames
        assert probe(spec.out_path).video_frames == spec.frames

    total = sum(frames)
    mix = tmp_path / "mix.wav"
    audio.write_silence(mix, total * (RATE // fps))
    out = tmp_path / "joined.mp4"
    concat_and_mux([s.out_path for s in specs], mix, out, tmp_path / "list.txt")
    info = verify_sync(out, total, fps)
    assert info["frames"] == total
    assert abs(info["video_sec"] - info["audio_sec"]) <= 1 / fps          # at most one frame apart
    assert info["video_sec"] == pytest.approx(total / fps, abs=1e-3)

    r0 = frame_rgb(out, 0)
    assert max(r0) < 40                                                   # fades up from black
    r, g, b = frame_rgb(out, frames[0] - 1)
    assert r > 200 and g < 60                                             # last frame of scene 1: pure red
    r, g, b = frame_rgb(out, frames[0] + t2 // 2)
    assert r > 50 and g > 50                                              # mid-crossfade: both pictures
    r, g, b = frame_rgb(out, frames[0] + t2)
    assert g > 200 and r < 60                                             # scene 2 fully in right after
    r, g, b = frame_rgb(out, frames[0] + frames[1] + t3)
    assert b > 200 and g < 60
    with pytest.raises(RenderError):
        verify_sync(out, total + 1, fps)


def test_ken_burns_moves_smoothly_without_jitter(tmp_path: Path) -> None:
    """The picture must travel in even sub-pixel steps (zoompan alone jumps in whole pixels)."""
    image = Image.new("RGB", (800, 450), "black")
    ImageDraw.Draw(image).rectangle([395, 0, 405, 450], fill="white")
    src = tmp_path / "bar.png"
    image.save(src)
    frames, fps = 50, 25
    spec = ClipSpec("kb", tmp_path / "kb.mp4", frames, fps, Shot(src, "pan_right", 0.1, frames), out_size=SIZE)
    render_clip(spec, 4, False, tmp_path / "work")
    proc = run_ffmpeg(["-i", spec.out_path, "-vf", "crop=640:4:0:178,format=gray", "-f", "rawvideo", "-"],
                      capture_stdout=True)
    row = 640 * 4
    centroids = []
    for index in range(frames):
        data = proc.stdout[index * row:(index + 1) * row]
        weights = [sum(data[x + 640 * y] for y in range(4)) for x in range(640)]
        centroids.append(sum(x * w for x, w in enumerate(weights)) / sum(weights))
    steps = [b - a for a, b in zip(centroids, centroids[1:])]
    mean = sum(steps) / len(steps)
    travel = centroids[0] - centroids[-1]
    assert travel == pytest.approx(0.1 * 640, abs=2.5)        # intensity = fraction of the frame travelled
    assert mean < 0                                           # camera pans right -> picture moves left
    # Even motion: with 4x supersampling the window moves in ~0.28 px steps, so no frame-to-frame
    # step may differ from the average by more than a fraction of a pixel (plain zoompan: ~1 px).
    assert max(abs(s - mean) for s in steps) < 0.4
    z, x, y = motion_expressions(Shot(src, "zoom_out", 0.2, 100), 0)
    assert z.startswith("1+0.20000*(1-") and "zoom" in x and "zoom" in y


@pytest.mark.parametrize("layout", ["blur_background", "center_crop", "pan_scan"])
def test_vertical_layouts_and_burned_subtitles(tmp_path: Path, layout: str) -> None:
    from pipeline.captions import AssStyle, SubWord, render_ass

    src = solid(tmp_path / "pic.png", "#2a6f97")
    workdir = tmp_path / "dir with ' quote, [brackets]"
    workdir.mkdir()
    ass = workdir / "subs.ass"
    style = AssStyle("DejaVu Sans", "#FFFFFF", "#FFD400", "center", True, 3)
    ass.write_text(render_ass([[SubWord("HELLO", 10.0, 10.5), SubWord("WORLD", 10.5, 11.0)]], style, (1080, 1920)),
                   encoding="utf-8")
    fps, frames = 30, 30
    plain = ClipSpec("p", workdir / "plain.mp4", frames, fps, Shot(src, "zoom_in", 0.1, frames), layout=layout,
                     out_size=(1080, 1920))
    subbed = ClipSpec("s", workdir / "subbed.mp4", frames, fps, Shot(src, "zoom_in", 0.1, frames), layout=layout,
                      out_size=(1080, 1920), ass_file=ass, ass_offset_frames=10 * fps, fonts_dir=workdir)
    for spec in (plain, subbed):
        render_clip(spec, 1, False, tmp_path / "work")
        info = probe(spec.out_path)
        assert (info.width, info.height, info.video_frames) == (1080, 1920, frames)

    def centre_brightness(path: Path) -> float:
        proc = run_ffmpeg(["-i", path, "-vf", "select='eq(n,5)',crop=700:160:190:880,format=gray", "-frames:v", "1",
                           "-f", "rawvideo", "-"], capture_stdout=True)
        return max(proc.stdout)

    # the clip starts 10 s into the timeline: the subtitle is visible only if the offset is applied
    assert centre_brightness(subbed.out_path) > 220 > centre_brightness(plain.out_path)
    assert filter_escape("a:b'c,d") == "a\\\\:b\\\\\\'c\\,d"


@pytest.mark.parametrize("motion", ["static", "zoom_in"])
def test_colours_survive_the_render_as_bt709(tmp_path: Path, motion: str) -> None:
    """highlight_color on screen must be the colour from the script: BT.709 conversion + tags."""
    src = solid(tmp_path / "yellow.png", "#FFD400")
    spec = ClipSpec("c", tmp_path / "c.mp4", 12, 30, Shot(src, motion, 0.1, 12), out_size=SIZE)
    render_clip(spec, 2, False, tmp_path / "work")
    tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=color_space,color_primaries,color_transfer,pix_fmt", "-of", "csv=p=0",
                           str(spec.out_path)], capture_output=True, text=True, check=True).stdout.strip()
    assert tags == "yuv420p,bt709,bt709,bt709"
    r, g, b = frame_rgb(spec.out_path, 6)
    assert abs(r - 255) <= 6 and abs(g - 212) <= 6 and b <= 8
