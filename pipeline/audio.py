"""Audio: separate stems (voice / music / sfx) and the loudness-normalised final mix.

All stems are 48 kHz stereo 16-bit WAV of exactly the master length, so a dubbed
track later needs only a new ``voice.wav`` — ``music.wav`` and ``sfx.wav`` are
reused as they are and :func:`mix_stems` produces that language's mix.

The music stem is stored *un-ducked*: ducking is driven by whichever voice stem
is mixed in, so it follows each language's own speech.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import AUDIO_CHANNELS, AUDIO_RATE
from pipeline.ffmpeg import FFmpegError, run_ffmpeg, run_ffmpeg_stderr

AUDIO_VERSION = 3
BYTES_PER_FRAME = 2 * AUDIO_CHANNELS  # s16 stereo
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aiff", ".aif")
MUSIC_CROSSFADE_SEC = 1.0
_PCM_ARGS = ["-vn", "-ac", str(AUDIO_CHANNELS), "-ar", str(AUDIO_RATE)]
_FMT = f"aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"


@dataclass(frozen=True)
class VoicePlacement:
    path: Path
    start_sample: int


@dataclass(frozen=True)
class SfxEvent:
    path: Path
    start_sample: int


@dataclass(frozen=True)
class MusicSegment:
    mood: str
    start_sample: int
    end_sample: int
    track: Path | None


# --------------------------------------------------------------------------- #
# PCM helpers
# --------------------------------------------------------------------------- #
def decode_pcm(path: Path) -> bytes:
    """Decode any audio file to raw s16le / 48 kHz / stereo."""
    proc = run_ffmpeg(["-i", path, *_PCM_ARGS, "-f", "s16le", "-acodec", "pcm_s16le", "-"], capture_stdout=True)
    return proc.stdout


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_FRAME / AUDIO_RATE


def audio_seconds(path: Path) -> float:
    """Exact decoded duration of an audio file."""
    return pcm_seconds(decode_pcm(path))


def write_wav(path: Path, pcm: bytes | bytearray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(AUDIO_CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(AUDIO_RATE)
        wav.writeframes(bytes(pcm))
    tmp.replace(path)


def write_silence(path: Path, total_samples: int) -> None:
    write_wav(path, bytes(total_samples * BYTES_PER_FRAME))


def wav_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes()


# --------------------------------------------------------------------------- #
# asset lookup (+ synthetic placeholders for --mock)
# --------------------------------------------------------------------------- #
def list_music_tracks(assets_dir: Path, mood: str) -> list[Path]:
    folder = assets_dir / "music" / mood
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS and not p.name.startswith("."))


def pick_music_track(assets_dir: Path, mood: str, key: str) -> Path | None:
    """Deterministic choice of a track for ``mood`` (stable for a given ``key``)."""
    tracks = list_music_tracks(assets_dir, mood)
    if not tracks:
        return None
    index = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(tracks)
    return tracks[index]


def find_sfx(assets_dir: Path, name: str) -> Path | None:
    folder = assets_dir / "sfx"
    for ext in AUDIO_EXTENSIONS:  # .wav first, as the spec says
        candidate = folder / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None


_MOCK_MOOD_HZ = {"tense_minimal": 110.0, "curious_pulse": 146.8, "dread_drone": 82.4,
                 "uplifting_resolve": 196.0, "reflective_piano": 130.8}


def synth_music(path: Path, mood: str, seconds: float = 24.0) -> Path:
    """Mock-mode placeholder music: a quiet two-note drone (never written into assets/)."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    base = _MOCK_MOOD_HZ.get(mood, 110.0)
    run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={base}:sample_rate={AUDIO_RATE}:duration={seconds}",
                "-f", "lavfi", "-i", f"sine=frequency={base * 1.5:.2f}:sample_rate={AUDIO_RATE}:duration={seconds}",
                "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0,tremolo=f=0.4:d=0.5,volume=-14dB",
                *_PCM_ARGS, "-c:a", "pcm_s16le", path])
    return path


def synth_sfx(path: Path, name: str) -> Path:
    """Mock-mode placeholder sound effect: a short decaying beep."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    freq = 300 + (int(hashlib.sha256(name.encode()).hexdigest(), 16) % 8) * 110
    run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={AUDIO_RATE}:duration=0.45",
                "-af", "afade=t=out:st=0.05:d=0.4,volume=-10dB", *_PCM_ARGS, "-c:a", "pcm_s16le", path])
    return path


# --------------------------------------------------------------------------- #
# stems
# --------------------------------------------------------------------------- #
def build_voice_stem(placements: Sequence[VoicePlacement], total_samples: int, out: Path) -> None:
    """Sample-exact voice stem: each line pasted at its frame-aligned offset."""
    buf = bytearray(total_samples * BYTES_PER_FRAME)
    for placement in placements:
        pcm = decode_pcm(placement.path)
        begin = placement.start_sample * BYTES_PER_FRAME
        end = min(len(buf), begin + len(pcm))
        if begin < end:
            buf[begin:end] = pcm[: end - begin]
    write_wav(out, buf)


def _exact_length(total_samples: int) -> str:
    return f"apad=whole_len={total_samples},atrim=end_sample={total_samples}"


def build_sfx_stem(events: Sequence[SfxEvent], total_samples: int, out: Path, gain_db: float = 0.0) -> None:
    if not events:
        write_silence(out, total_samples)
        return
    args: list[str | Path] = []
    chains: list[str] = []
    for index, event in enumerate(events):
        args += ["-i", event.path]
        chains.append(f"[{index}:a]{_FMT},adelay=delays={event.start_sample}S:all=1[s{index}]")
    labels = "".join(f"[s{i}]" for i in range(len(events)))
    joiner = f"amix=inputs={len(events)}:normalize=0:dropout_transition=0," if len(events) > 1 else ""
    chains.append(f"{labels}{joiner}volume={gain_db}dB,{_exact_length(total_samples)}[out]")
    tmp = out.with_name(out.stem + ".tmp.wav")
    run_ffmpeg([*args, "-filter_complex", ";".join(chains), "-map", "[out]", "-c:a", "pcm_s16le",
                "-ar", str(AUDIO_RATE), tmp])
    tmp.replace(out)


def build_music_stem(segments: Sequence[MusicSegment], total_samples: int, out: Path, gain_db: float = -6.0) -> None:
    """Music bed: one looped track per segment, cross-faded where the mood changes."""
    active = [s for s in segments if s.track is not None and s.end_sample > s.start_sample]
    if not active:
        write_silence(out, total_samples)
        return
    args: list[str | Path] = []
    chains: list[str] = []
    for index, seg in enumerate(active):
        is_last = seg is active[-1]
        body = (seg.end_sample - seg.start_sample) / AUDIO_RATE
        tail = 0.0 if is_last else MUSIC_CROSSFADE_SEC
        length = body + tail
        fade_in = min(0.5 if index == 0 else MUSIC_CROSSFADE_SEC, length / 2)
        fade_out = min(1.5 if is_last else MUSIC_CROSSFADE_SEC, length / 2)
        args += ["-stream_loop", "-1", "-t", f"{length + 0.25:.3f}", "-i", seg.track]  # type: ignore[list-item]
        chains.append(
            f"[{index}:a]{_FMT},atrim=duration={length:.4f},asetpts=PTS-STARTPTS,"
            f"afade=t=in:st=0:d={fade_in:.3f},afade=t=out:st={length - fade_out:.4f}:d={fade_out:.3f},"
            f"adelay=delays={seg.start_sample}S:all=1[m{index}]")
    labels = "".join(f"[m{i}]" for i in range(len(active)))
    joiner = f"amix=inputs={len(active)}:normalize=0:dropout_transition=0," if len(active) > 1 else ""
    chains.append(f"{labels}{joiner}volume={gain_db}dB,{_exact_length(total_samples)}[out]")
    tmp = out.with_name(out.stem + ".tmp.wav")
    run_ffmpeg([*args, "-filter_complex", ";".join(chains), "-map", "[out]", "-c:a", "pcm_s16le",
                "-ar", str(AUDIO_RATE), tmp])
    tmp.replace(out)


# --------------------------------------------------------------------------- #
# mix
# --------------------------------------------------------------------------- #
def ducking_filter(ducking_db: float) -> str:
    """``sidechaincompress`` settings that pull the music down by about ``ducking_db`` under speech.

    A hard, low-threshold compressor would duck by far more than asked, so its
    output is blended with the dry signal: ``mix`` leaves a floor of exactly
    ``10^(ducking_db/20)`` whenever the voice is present, regardless of how loud
    the voice is. Slow release keeps the bed down between words.
    """
    floor = 10 ** (min(ducking_db, 0.0) / 20.0)
    mix = max(0.0, min(1.0, 1.0 - floor))
    return (f"sidechaincompress=threshold=0.003:ratio=20:attack=15:release=400:knee=2:"
            f"level_sc=16:detection=rms:mix={mix:.4f}")


def measure_loudness(path: Path, target_lufs: float = -14.0, true_peak_db: float = -1.0) -> dict[str, float]:
    """EBU R128 measurement via ``loudnorm`` (integrated LUFS, true peak, LRA, threshold, offset)."""
    stderr = run_ffmpeg_stderr(["-i", path, "-af",
                                f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA=11:print_format=json",
                                "-f", "null", "-"])
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, flags=re.S)
    if not blocks:
        raise FFmpegError("loudnorm did not report measurements")
    raw: dict[str, Any] = json.loads(blocks[-1])
    out: dict[str, float] = {}
    for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        try:
            out[key] = float(raw[key])
        except (KeyError, ValueError):
            out[key] = float("-inf")
    return out


def mix_stems(voice: Path, music: Path, sfx: Path, out: Path, *, ducking_db: float, total_samples: int,
              work_dir: Path, target_lufs: float = -14.0, true_peak_db: float = -1.0,
              log: logging.Logger | None = None) -> dict[str, float | bool]:
    """voice + ducked music + sfx -> ``out``, normalised to ``target_lufs`` / ``true_peak_db``.

    Two-pass ``loudnorm`` (measure, then apply with the measured values, linear when possible).
    Reusable for any language: pass another voice stem with the same music/sfx stems.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    premix = work_dir / f"{out.stem}.premix.wav"
    duck = ducking_filter(ducking_db) if ducking_db < 0 else "anull"
    if ducking_db < 0:
        graph = (f"[0:a]{_FMT},asplit=2[v][sc];[1:a]{_FMT}[m0];[m0][sc]{duck}[m];[2:a]{_FMT}[s];"
                 f"[v][m][s]amix=inputs=3:normalize=0:dropout_transition=0[out]")
    else:
        graph = (f"[0:a]{_FMT}[v];[1:a]{_FMT}[m];[2:a]{_FMT}[s];"
                 f"[v][m][s]amix=inputs=3:normalize=0:dropout_transition=0[out]")
    run_ffmpeg(["-i", voice, "-i", music, "-i", sfx, "-filter_complex", graph, "-map", "[out]",
                "-c:a", "pcm_f32le", "-ar", str(AUDIO_RATE), premix])

    measured = measure_loudness(premix, target_lufs, true_peak_db)
    report: dict[str, float | bool] = {"premix_lufs": measured["input_i"], "premix_true_peak_db": measured["input_tp"]}
    tmp = out.with_name(out.stem + ".tmp.wav")
    if measured["input_i"] < -70.0:  # digital silence (e.g. mock voice, no music): nothing to normalise
        run_ffmpeg(["-i", premix, "-af", _exact_length(total_samples), "-c:a", "pcm_s16le", "-ar", str(AUDIO_RATE), tmp])
        tmp.replace(out)
        report.update({"silent": True, "integrated_lufs": measured["input_i"], "true_peak_db": measured["input_tp"]})
        if log:
            log.warning("mix is silent (%.1f LUFS) — loudness normalisation skipped", measured["input_i"])
    else:
        lra = min(50.0, max(11.0, measured["input_lra"] + 1.0))
        loudnorm = (f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra:.1f}:"
                    f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
                    f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
                    f"offset={measured['target_offset']}:linear=true:print_format=summary")
        # In dynamic mode loudnorm outputs 192 kHz: resample *before* trimming to the exact sample count.
        run_ffmpeg(["-i", premix, "-af", f"{loudnorm},aresample={AUDIO_RATE},{_exact_length(total_samples)}",
                    "-c:a", "pcm_s16le", "-ar", str(AUDIO_RATE), tmp])
        tmp.replace(out)
        final = measure_loudness(out, target_lufs, true_peak_db)
        report.update({"silent": False, "integrated_lufs": final["input_i"], "true_peak_db": final["input_tp"]})
        if log:
            log.info("mix loudness: %.1f LUFS integrated, true peak %.1f dBTP (target %.1f / %.1f)",
                     final["input_i"], final["input_tp"], target_lufs, true_peak_db)
            if abs(final["input_i"] - target_lufs) > 1.0:
                log.warning("mix loudness is %.1f LUFS, more than 1 LU away from the target", final["input_i"])
    premix.unlink(missing_ok=True)
    return report


def build_all(*, voice: Sequence[VoicePlacement], sfx: Sequence[SfxEvent], music: Sequence[MusicSegment],
              total_samples: int, out_dir: Path, work_dir: Path, ducking_db: float, music_gain_db: float,
              sfx_gain_db: float, target_lufs: float, true_peak_db: float,
              log: logging.Logger) -> dict[str, float | bool]:
    """Write ``voice.wav``, ``music.wav``, ``sfx.wav`` and the final ``mix.wav`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_path, music_path, sfx_path = out_dir / "voice.wav", out_dir / "music.wav", out_dir / "sfx.wav"
    build_voice_stem(voice, total_samples, voice_path)
    build_music_stem(music, total_samples, music_path, music_gain_db)
    build_sfx_stem(sfx, total_samples, sfx_path, sfx_gain_db)
    for stem in (voice_path, music_path, sfx_path):
        if wav_samples(stem) != total_samples:
            raise FFmpegError(f"{stem.name}: {wav_samples(stem)} samples, expected {total_samples}")
    report = mix_stems(voice_path, music_path, sfx_path, out_dir / "mix.wav", ducking_db=ducking_db,
                       total_samples=total_samples, work_dir=work_dir, target_lufs=target_lufs,
                       true_peak_db=true_peak_db, log=log)
    if wav_samples(out_dir / "mix.wav") != total_samples:
        raise FFmpegError("mix.wav length differs from the master timeline")
    return report


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
