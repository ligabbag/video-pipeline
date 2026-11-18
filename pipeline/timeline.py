"""Frame-accurate timelines.

Every line of voice-over starts exactly on a video frame boundary; a scene lasts
its audio plus a pause, rounded *up* to whole frames. Because audio stems are
assembled at ``frame * samples_per_frame`` sample offsets (48 kHz divides evenly
by 24/25/30 fps), video and audio cannot drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from pipeline.config import AUDIO_RATE


@dataclass(frozen=True)
class LineSpec:
    """One spoken line to be placed on a timeline."""

    key: str
    audio_sec: float
    pause_sec: float


@dataclass(frozen=True)
class Slot:
    key: str
    start_frame: int
    end_frame: int
    audio_sec: float
    fps: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_sec(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_sec(self) -> float:
        return self.end_frame / self.fps

    @property
    def voice_end_sec(self) -> float:
        return self.start_sec + self.audio_sec

    @property
    def start_sample(self) -> int:
        return self.start_frame * samples_per_frame(self.fps)


def samples_per_frame(fps: int) -> int:
    if AUDIO_RATE % fps:
        raise ValueError(f"{AUDIO_RATE} Hz is not divisible by {fps} fps")
    return AUDIO_RATE // fps


def lay_out(lines: Sequence[LineSpec], fps: int, start_frame: int = 0) -> list[Slot]:
    """Place lines back to back. Each slot = audio + pause, rounded up to whole frames."""
    slots: list[Slot] = []
    cursor = start_frame
    for line in lines:
        end_exact = cursor / fps + max(line.audio_sec, 0.0) + max(line.pause_sec, 0.0)
        end_frame = max(cursor + 1, math.ceil(end_exact * fps - 1e-6))
        slots.append(Slot(line.key, cursor, end_frame, line.audio_sec, fps))
        cursor = end_frame
    return slots


def seconds_to_frames(seconds: float, fps: int) -> int:
    return max(0, round(seconds * fps))
