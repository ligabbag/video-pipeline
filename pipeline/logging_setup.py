"""Human-readable logging to the console and to ``output/<video_id>/pipeline.log``.

Every formatted line passes through a redactor that replaces known secret
values with ``***`` — keys must never reach a log, not even partially.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

LOGGER_NAME = "pipeline"


class RedactingFormatter(logging.Formatter):
    """Formatter that removes secret values from the final log line (incl. tracebacks)."""

    def __init__(self, fmt: str, datefmt: str | None, secrets: Iterable[str]) -> None:
        super().__init__(fmt, datefmt)
        # Longest first so overlapping secrets are fully removed.
        self._secrets = sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record), self._secrets)


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "***")
    return text


def setup_logging(log_file: Path | None, secrets: Iterable[str], verbose: bool = False) -> logging.Logger:
    """Configure the ``pipeline`` logger. Safe to call repeatedly (handlers are replaced)."""
    secrets = list(secrets)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(RedactingFormatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S", secrets))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(RedactingFormatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S", secrets))
        logger.addHandler(file_handler)

    # HTTP libraries can log request lines at DEBUG; keep them quiet.
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
