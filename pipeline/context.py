"""Everything one pipeline run carries around."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import OutputPaths, Settings
from pipeline.contract import Script
from pipeline.state import PipelineState, StateStore


@dataclass
class RunContext:
    settings: Settings
    script: Script
    script_path: Path
    state: PipelineState
    store: StateStore
    paths: OutputPaths
    log: logging.Logger
    mock: bool = False
    force: bool = False
    no_cache: bool = False
    hw: bool = False

    def save(self) -> None:
        self.store.save(self.state)

    @property
    def fps(self) -> int:
        return int(self.script.production.fps)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.script.production.frame_size
