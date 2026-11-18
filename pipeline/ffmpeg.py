"""Thin, typed wrappers around the ``ffmpeg`` / ``ffprobe`` binaries (via subprocess)."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class FFmpegError(RuntimeError):
    """FFmpeg exited with a non-zero status."""


def ensure_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FFmpegError(f"'{binary}' not found in PATH — install FFmpeg (see README.md)")


def run_ffmpeg(args: Sequence[str | Path], *, cwd: Path | None = None, capture_stdout: bool = False,
               stdin: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run ``ffmpeg -y <args>``; raise :class:`FFmpegError` with the stderr tail on failure."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *[str(a) for a in args]]
    if stdin is not None:
        cmd.remove("-nostdin")
    proc = subprocess.run(cmd, cwd=cwd, input=stdin, stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-1500:]
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {tail}\ncommand: {' '.join(cmd[:60])}")
    return proc


def run_ffmpeg_stderr(args: Sequence[str | Path]) -> str:
    """Run ffmpeg at ``info`` log level and return stderr (for filters that print reports)."""
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", *[str(a) for a in args]]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    text = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}): {text.strip()[-1500:]}")
    return text


@dataclass(frozen=True)
class MediaInfo:
    video_frames: int | None
    video_duration: float | None
    audio_duration: float | None
    fps: float | None
    width: int | None
    height: int | None


def probe(path: Path) -> MediaInfo:
    """Exact packet count and stream durations of a media file."""
    cmd = ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
           "stream=codec_type,nb_read_packets,duration,r_frame_rate,width,height", "-of", "json", str(path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {proc.stderr.decode('utf-8', 'replace').strip()[-500:]}")
    streams = json.loads(proc.stdout.decode("utf-8")).get("streams", [])
    frames = v_dur = a_dur = fps = None
    width = height = None
    for stream in streams:
        duration = float(stream["duration"]) if stream.get("duration") not in (None, "N/A") else None
        if stream.get("codec_type") == "video" and frames is None:
            frames = int(stream["nb_read_packets"]) if stream.get("nb_read_packets") else None
            v_dur = duration
            num, _, den = str(stream.get("r_frame_rate", "0/1")).partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else None
            width, height = stream.get("width"), stream.get("height")
        elif stream.get("codec_type") == "audio" and a_dur is None:
            a_dur = duration
    return MediaInfo(frames, v_dur, a_dur, fps, width, height)


def filter_escape(value: str | Path) -> str:
    """Escape a value (e.g. a path) for use as a filter option inside ``-filter_complex``.

    Two parsing levels apply: the filter's option parser, then the filtergraph parser.
    """
    text = str(value).replace("\\", "/")
    text = text.replace(":", "\\:").replace("'", "\\'")              # option level
    out: list[str] = []
    for ch in text:                                                  # filtergraph level
        out.append("\\" + ch if ch in "\\'[],;" else ch)
    return "".join(out)


def concat_list_entry(path: Path) -> str:
    """One ``file`` line for the concat demuxer."""
    escaped = str(path.resolve()).replace("'", "'\\''")
    return f"file '{escaped}'\n"
