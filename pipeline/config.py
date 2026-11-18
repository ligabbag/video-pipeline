"""Runtime settings: project paths, tunables and secrets.

Secrets come only from environment variables or a ``.env`` file in the project
root. They are never written to logs or to the state file (see ``logging_setup``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# Fixed timeline constants (task spec, stage 2).
SCENE_PAUSE_SEC = 0.15
SECTION_PAUSE_SEC = 0.6

AUDIO_RATE = 48_000
AUDIO_CHANNELS = 2

SECRET_ENV_NAMES = ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "BFL_API_KEY")


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``.env`` parser: ``KEY=value`` lines, ``#`` comments, optional quotes."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


CHANNEL_PIN_PREFIX = "YOUTUBE_CHANNEL_ID__"


def channel_pin_key(channel_id: str) -> str:
    """``yt_tech-disasters.en`` -> ``YT_TECH_DISASTERS_EN`` (the suffix of the pin's variable name)."""
    return "".join(ch if ch.isalnum() else "_" for ch in channel_id).upper()


def _default_workers() -> int:
    cpus = os.cpu_count() or 2
    return max(1, min(4, cpus // 2))


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to know about its environment."""

    root: Path
    elevenlabs_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    bfl_api_key: str | None = field(default=None, repr=False)

    elevenlabs_base_url: str = "https://api.elevenlabs.io"
    elevenlabs_output_format: str = "mp3_44100_128"
    openai_base_url: str = "https://api.openai.com"
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str | None = None  # None -> chosen from the model's capabilities
    openai_image_quality: str = "medium"
    bfl_base_url: str = "https://api.bfl.ai"
    bfl_model: str = "flux-2-pro"
    bfl_image_size: str = "2560x1440"
    image_provider_override: str | None = None

    supersample: int = 4          # Ken Burns oversampling factor (anti-jitter)
    render_workers: int = 2       # parallel FFmpeg scene renders
    image_workers: int = 4        # parallel image API requests (spec: at most 4)
    http_attempts: int = 6        # tries per API call on 429/5xx
    music_gain_db: float = -6.0   # level of the music bed before ducking
    sfx_gain_db: float = -3.0
    target_lufs: float = -14.0
    true_peak_db: float = -1.0

    # ---- YouTube upload (paths may be absolute or relative to the project root) ----
    youtube_secrets_dir: str = "secrets"                 # token_<channel_id>.json files live here
    youtube_client_secret_file: str | None = None        # default: <secrets dir>/client_secret.json
    youtube_oauth_scopes: tuple[str, ...] = ()           # empty -> youtube_auth.DEFAULT_SCOPES
    youtube_oauth_port: int = 0                          # local redirect port for the first login (0 = any free)
    youtube_oauth_timeout_sec: int = 600                 # give up waiting for the browser login after this long
    youtube_upload_chunk_mb: int = 8                     # resumable upload chunk (progress + resume granularity)
    youtube_channel_pins: tuple[tuple[str, str], ...] = ()   # (script channel_id, expected YouTube channel ID)

    # ---- paths -----------------------------------------------------------
    @property
    def output_root(self) -> Path:
        return self.root / "output"

    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def secrets_dir(self) -> Path:
        return self.abs(self.youtube_secrets_dir)

    @property
    def client_secret_path(self) -> Path:
        if self.youtube_client_secret_file:
            return self.abs(self.youtube_client_secret_file)
        return self.secrets_dir / "client_secret.json"

    def pinned_channel(self, channel_id: str) -> str | None:
        """Expected YouTube channel ID for a script ``channel_id`` (``YOUTUBE_CHANNEL_ID__<ID>`` in ``.env``)."""
        return dict(self.youtube_channel_pins).get(channel_pin_key(channel_id))

    def secrets(self) -> list[str]:
        """Secret values to be redacted from every log line."""
        return [s for s in (self.elevenlabs_api_key, self.openai_api_key, self.bfl_api_key) if s]

    def rel(self, path: Path) -> str:
        """Path relative to the project root (how paths are stored in the state file)."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def abs(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return p if p.is_absolute() else self.root / p

    # ---- construction ----------------------------------------------------
    @classmethod
    def load(cls, root: Path | None = None, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from ``<root>/.env`` overlaid by real environment variables."""
        base_env = dict(os.environ if env is None else env)
        root_path = Path(root or base_env.get("PIPELINE_ROOT") or Path.cwd()).resolve()
        merged = parse_env_file(root_path / ".env")
        merged.update(base_env)  # real environment wins over .env

        def get(name: str) -> str | None:
            value = merged.get(name, "").strip()
            return value or None

        def get_int(name: str, default: int, lo: int, hi: int) -> int:
            raw = get(name)
            try:
                return max(lo, min(hi, int(raw))) if raw else default
            except ValueError:
                return default

        def get_float(name: str, default: float) -> float:
            raw = get(name)
            try:
                return float(raw) if raw else default
            except ValueError:
                return default

        defaults = cls(root=root_path)
        return cls(
            root=root_path,
            elevenlabs_api_key=get("ELEVENLABS_API_KEY"),
            openai_api_key=get("OPENAI_API_KEY"),
            bfl_api_key=get("BFL_API_KEY"),
            elevenlabs_base_url=get("ELEVENLABS_BASE_URL") or defaults.elevenlabs_base_url,
            elevenlabs_output_format=get("ELEVENLABS_OUTPUT_FORMAT") or defaults.elevenlabs_output_format,
            openai_base_url=get("OPENAI_BASE_URL") or defaults.openai_base_url,
            openai_image_model=get("OPENAI_IMAGE_MODEL") or defaults.openai_image_model,
            openai_image_size=get("OPENAI_IMAGE_SIZE"),
            openai_image_quality=get("OPENAI_IMAGE_QUALITY") or defaults.openai_image_quality,
            bfl_base_url=get("BFL_BASE_URL") or defaults.bfl_base_url,
            bfl_model=get("BFL_MODEL") or defaults.bfl_model,
            bfl_image_size=get("BFL_IMAGE_SIZE") or defaults.bfl_image_size,
            image_provider_override=get("PIPELINE_IMAGE_PROVIDER"),
            supersample=get_int("PIPELINE_SUPERSAMPLE", defaults.supersample, 1, 8),
            render_workers=get_int("PIPELINE_RENDER_WORKERS", _default_workers(), 1, 16),
            image_workers=get_int("PIPELINE_IMAGE_WORKERS", defaults.image_workers, 1, 4),
            http_attempts=get_int("PIPELINE_HTTP_ATTEMPTS", defaults.http_attempts, 1, 12),
            music_gain_db=get_float("PIPELINE_MUSIC_GAIN_DB", defaults.music_gain_db),
            sfx_gain_db=get_float("PIPELINE_SFX_GAIN_DB", defaults.sfx_gain_db),
            youtube_secrets_dir=get("YOUTUBE_SECRETS_DIR") or defaults.youtube_secrets_dir,
            youtube_client_secret_file=get("YOUTUBE_CLIENT_SECRET_FILE"),
            youtube_oauth_scopes=tuple((get("YOUTUBE_OAUTH_SCOPES") or "").replace(",", " ").split()),
            youtube_oauth_port=get_int("YOUTUBE_OAUTH_PORT", defaults.youtube_oauth_port, 0, 65535),
            youtube_oauth_timeout_sec=get_int("YOUTUBE_OAUTH_TIMEOUT_SEC", defaults.youtube_oauth_timeout_sec,
                                              10, 86400),
            youtube_upload_chunk_mb=get_int("YOUTUBE_UPLOAD_CHUNK_MB", defaults.youtube_upload_chunk_mb, 1, 256),
            youtube_channel_pins=tuple(sorted(
                (name[len(CHANNEL_PIN_PREFIX):].upper(), value.strip()) for name, value in merged.items()
                if name.startswith(CHANNEL_PIN_PREFIX) and value.strip())),
        )


@dataclass(frozen=True)
class OutputPaths:
    """Layout of ``output/<video_id>/`` (task spec, "Структура результата")."""

    base: Path

    @property
    def audio(self) -> Path:
        return self.base / "audio"

    @property
    def images(self) -> Path:
        return self.base / "images"

    @property
    def clips(self) -> Path:
        return self.base / "clips"

    @property
    def stems(self) -> Path:
        return self.base / "stems"

    @property
    def final(self) -> Path:
        return self.base / "final"

    @property
    def shorts(self) -> Path:
        return self.base / "final" / "shorts"

    @property
    def captions(self) -> Path:
        return self.base / "captions"

    @property
    def work(self) -> Path:
        """Scratch area for intermediate files (safe to delete)."""
        return self.base / "work"

    @property
    def log_file(self) -> Path:
        return self.base / "pipeline.log"

    @property
    def metadata(self) -> Path:
        return self.base / "metadata.json"

    def ensure(self) -> None:
        for d in (self.audio, self.images, self.clips, self.stems, self.final, self.shorts,
                  self.captions, self.work):
            d.mkdir(parents=True, exist_ok=True)
