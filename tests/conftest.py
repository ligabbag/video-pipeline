"""Shared fixtures and fakes for the test suite."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import OutputPaths, Settings  # noqa: E402
from pipeline.context import RunContext  # noqa: E402
from pipeline.contract import load_script  # noqa: E402
from pipeline.logging_setup import setup_logging  # noqa: E402
from pipeline.state import StateStore  # noqa: E402

EXAMPLE = PROJECT_ROOT / "video_script.example.json"


@pytest.fixture(scope="session")
def example_path() -> Path:
    return EXAMPLE


@pytest.fixture()
def example_doc() -> dict[str, Any]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def approved(doc: dict[str, Any]) -> dict[str, Any]:
    """A copy of the script that a human has approved."""
    out = copy.deepcopy(doc)
    out["status"] = "approved"
    out["qa"]["facts_verified"] = True
    out["qa"]["reviewer"] = "test"
    return out


def small_doc(doc: dict[str, Any], scene_ids: list[str]) -> dict[str, Any]:
    """Cut the example down to a few scenes of its first section (still valid per the contract)."""
    out = copy.deepcopy(doc)
    section = out["sections"][0]
    section["scenes"] = [s for s in section["scenes"] if s["scene_id"] in scene_ids]
    out["sections"] = [section]
    out["shorts"] = []
    out["distribution"]["telegram_brief"]["image_scene_ids"] = scene_ids[:1]
    return out


def make_context(root: Path, doc: dict[str, Any], *, mock: bool = True, env: dict[str, str] | None = None,
                 **flags: Any) -> RunContext:
    """Build a RunContext the way ``pipeline run`` does, inside ``root``."""
    script_path = root / "script.json"
    script_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    loaded = load_script(script_path)
    settings = Settings.load(root, env=env or {})
    paths = OutputPaths(settings.output_root / loaded.script.video_id)
    paths.ensure()
    log = setup_logging(paths.log_file, settings.secrets())
    store = StateStore(settings.state_dir, loaded.script.video_id)
    return RunContext(settings=settings, script=loaded.script, script_path=script_path, state=store.load(),
                      store=store, paths=paths, log=log, mock=mock, **flags)


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, content: bytes = b"",
                 headers: dict[str, str] | None = None, text: str | None = None) -> None:
        self.status_code = status
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Stands in for ``requests.Session``; records every call."""

    def __init__(self, handler: Callable[[str, str, dict[str, Any]], FakeResponse]) -> None:
        self.handler = handler
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        return self.handler(method, url, kwargs)
