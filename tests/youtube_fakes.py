"""Fakes for the upload tests: an in-memory YouTube, an HTTP-level upload server, an auth backend,
and a helper that builds a "rendered" project without running FFmpeg. No network anywhere."""
from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
import requests

from pipeline.context import RunContext
from pipeline.metadata import build_metadata
from pipeline.state import Artifact, ShortState, TimelineInfo, stable_hash
from pipeline.thumbnail import run_thumbnail_stage
from pipeline.youtube_api import API_BASE, UPLOAD_BASE, ChannelInfo, YouTubeAPI
from pipeline.youtube_auth import Connection
from tests.conftest import FakeResponse, approved, make_context, small_doc

CHANNEL = ChannelInfo(id="UC" + "a" * 22, title="Tech Disasters", long_uploads_status="allowed")
OTHER_CHANNEL = ChannelInfo(id="UC" + "b" * 22, title="Cooking With Bob", long_uploads_status="allowed")
SH1 = "tech-0001-knight-capital-sh1"
SH2 = "tech-0001-knight-capital-sh2"


@pytest.fixture()
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to open a socket fails the test."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted in a test that must run offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# --------------------------------------------------------------------------- #
# in-memory YouTube (implements the interface the upload command depends on)
# --------------------------------------------------------------------------- #
class FakeYouTube(YouTubeAPI):
    def __init__(self, channel: ChannelInfo = CHANNEL) -> None:
        self.channel = channel
        self.calls: list[tuple[str, str]] = []
        self.videos: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, list[Exception]] = {}      # call name -> exceptions raised by the next calls
        self.forced_privacy: str | None = None               # "an unverified API project": everything stays private
        self.drop_audio_language = False
        self.resume_urls: list[str | None] = []
        self.sessions = 0

    def fail_next(self, call: str, *errors: Exception) -> None:
        self.failures.setdefault(call, []).extend(errors)

    def _maybe_fail(self, call: str) -> None:
        queue = self.failures.get(call)
        if queue:
            raise queue.pop(0)

    def count(self, call: str) -> int:
        return sum(1 for name, _ in self.calls if name == call)

    def my_channel(self) -> ChannelInfo:
        self.calls.append(("my_channel", ""))
        self._maybe_fail("my_channel")
        return self.channel

    def upload_video(self, path: Path, body: dict[str, Any], *, notify_subscribers: bool = True,
                     resume_url: str | None = None, on_session: Callable[[str], None] | None = None,
                     on_progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        self.calls.append(("upload", path.name))
        self.resume_urls.append(resume_url)
        if resume_url is None:
            self.sessions += 1
            if on_session:
                on_session(f"https://upload.test/session/{self.sessions}?upload_id=SESSION-SECRET-{self.sessions}")
        if on_progress:
            on_progress(path.stat().st_size // 2, path.stat().st_size)
        self._maybe_fail("upload")
        if on_progress:
            on_progress(path.stat().st_size, path.stat().st_size)
        video_id = f"vid{len(self.videos) + 1:08d}"
        self.videos[video_id] = {"body": json.loads(json.dumps(body)), "sha": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "file": path.name, "notify": notify_subscribers, "thumbnail": None, "captions": []}
        return {"id": video_id, "status": {"uploadStatus": "uploaded"}}

    def set_thumbnail(self, video_id: str, path: Path) -> None:
        self.calls.append(("thumbnail", video_id))
        self._maybe_fail("thumbnail")
        self.videos[video_id]["thumbnail"] = path.read_bytes()

    def insert_caption(self, video_id: str, path: Path, *, language: str, name: str) -> str:
        self.calls.append(("captions", video_id))
        self._maybe_fail("captions")
        self.videos[video_id]["captions"].append({"language": language, "name": name, "text": path.read_text("utf-8")})
        return f"cap-{video_id}"

    def list_videos(self, video_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        self.calls.append(("list", ",".join(video_ids)))
        self._maybe_fail("list")
        found = {}
        for video_id in video_ids:
            video = self.videos.get(video_id)
            if video is None:
                continue
            status = dict(video["body"]["status"], uploadStatus="uploaded")
            if self.forced_privacy:
                status["privacyStatus"] = self.forced_privacy
            snippet = dict(video["body"]["snippet"])
            if self.drop_audio_language:
                snippet.pop("defaultAudioLanguage", None)
            found[video_id] = {"id": video_id, "status": status, "snippet": snippet}
        return found


class Connector:
    """Stands in for ``pipeline.youtube_auth.connect``; counts how often authorisation was needed."""

    def __init__(self, api: FakeYouTube, root: Path) -> None:
        self.api = api
        self.root = root
        self.calls = 0
        self.error: Exception | None = None

    def __call__(self, settings: Any, channel_id: str, log: Any, **kwargs: Any) -> Connection:
        self.calls += 1
        if self.error:
            raise self.error
        return Connection(api=self.api, channel=self.api.my_channel(),
                          token_path=self.root / "secrets" / f"token_{channel_id}.json", first_login=False,
                          secrets=["ACCESS-SECRET-1", "REFRESH-SECRET-1"])


# --------------------------------------------------------------------------- #
# auth backend
# --------------------------------------------------------------------------- #
class FakeAuthBackend:
    """Credentials are plain dicts; ``login`` plays the browser flow."""

    def __init__(self) -> None:
        self.logins = 0
        self.refreshes = 0
        self.refresh_error: Exception | None = None

    def login(self, client_secret_file: Path, scopes: Sequence[str], port: int, log: Any,
              timeout_sec: int = 600) -> dict[str, Any]:
        self.logins += 1
        return {"token": f"ACCESS-SECRET-{self.logins}", "refresh_token": "REFRESH-SECRET-1",
                "client_secret": "CLIENT-SECRET-XYZ", "scopes": list(scopes), "valid": True}

    def from_info(self, info: dict[str, Any], scopes: Sequence[str]) -> dict[str, Any]:
        return dict(info)

    def to_info(self, credentials: dict[str, Any]) -> dict[str, Any]:
        return dict(credentials)

    def is_valid(self, credentials: dict[str, Any]) -> bool:
        return bool(credentials.get("valid"))

    def refresh(self, credentials: dict[str, Any]) -> None:
        self.refreshes += 1
        if self.refresh_error:
            raise self.refresh_error
        credentials.update(valid=True, token="ACCESS-SECRET-REFRESHED")

    def session(self, credentials: dict[str, Any]) -> object:
        return object()

    def secret_values(self, credentials: dict[str, Any]) -> list[str]:
        return [credentials["token"], credentials["refresh_token"], credentials["client_secret"]]

    def translate(self, exc: BaseException) -> BaseException | None:
        return None


def write_client_secret(root: Path, kind: str = "installed") -> Path:
    path = root / "secrets" / "client_secret.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({kind: {"client_id": "id.apps.googleusercontent.com",
                                       "client_secret": "CLIENT-SECRET-XYZ",
                                       "redirect_uris": ["http://localhost"]}}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# HTTP-level upload server for HttpYouTubeAPI (stands behind a FakeSession)
# --------------------------------------------------------------------------- #
class FakeUploadServer:
    """Speaks the documented resumable protocol; faults can be queued for upcoming chunk PUTs.

    Faults: ``("drop", n)`` — keep the first n bytes of the chunk, then the connection dies;
    ``("status", code)`` — answer with that HTTP status without storing anything;
    ``("lost_reply", 0)`` — store the whole chunk but the answer never arrives;
    ``("expire", 0)`` — forget the session (404).
    """

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.initiations: list[dict[str, Any]] = []
        self.chunk_puts: list[tuple[int, int, int]] = []        # (first byte, last byte, total)
        self.status_queries = 0
        self.faults: list[tuple[str, int]] = []
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.thumbnails: dict[str, dict[str, Any]] = {}
        self.captions: list[dict[str, Any]] = []
        self.errors: dict[str, FakeResponse] = {}               # endpoint name -> canned error answer

    def data(self, index: int = -1) -> bytes:
        return bytes(list(self.sessions.values())[index]["data"])

    @staticmethod
    def _incomplete(session: dict[str, Any]) -> FakeResponse:
        headers = {"Range": f"bytes=0-{len(session['data']) - 1}"} if session["data"] else {}
        return FakeResponse(308, headers=headers)

    def _finish(self, session: dict[str, Any]) -> FakeResponse:
        session["resource"] = {"id": f"vid{len(self.sessions):08d}", "status": {"uploadStatus": "uploaded"},
                               "snippet": session["body"]["snippet"]}
        return FakeResponse(201, payload=session["resource"])

    def __call__(self, method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        self.requests.append((method, url, kwargs))
        headers = kwargs.get("headers") or {}
        params = kwargs.get("params") or {}
        if method == "POST" and url == f"{UPLOAD_BASE}/videos":
            if "videos.insert" in self.errors:
                return self.errors["videos.insert"]
            assert params["uploadType"] == "resumable" and kwargs.get("allow_redirects") is False
            session_url = f"https://upload.test/session/{len(self.sessions) + 1}?upload_id=SESSION-SECRET"
            self.sessions[session_url] = {"total": int(headers["X-Upload-Content-Length"]), "data": bytearray(),
                                          "body": json.loads(kwargs["data"]), "resource": None}
            self.initiations.append({"params": params, "headers": headers})
            return FakeResponse(200, headers={"Location": session_url})
        if method == "PUT" and url.startswith("https://upload.test/session/"):
            session = self.sessions.get(url)
            if session is None or session.get("expired"):
                return FakeResponse(404, text="Not Found")
            content_range = headers["Content-Range"]
            if content_range.startswith("bytes */"):
                self.status_queries += 1
                assert int(content_range.split("/")[1]) == session["total"] and headers["Content-Length"] == "0"
                return FakeResponse(201, payload=session["resource"]) if session["resource"] else self._incomplete(session)
            span, total = content_range[len("bytes "):].split("/")
            first, last = (int(x) for x in span.split("-"))
            chunk = kwargs["data"]
            assert int(total) == session["total"] and len(chunk) == last - first + 1 == int(headers["Content-Length"])
            assert first == len(session["data"]), "client resumed from the wrong byte"
            self.chunk_puts.append((first, last, int(total)))
            fault = self.faults.pop(0) if self.faults else None
            if fault and fault[0] == "drop":
                session["data"] += chunk[:fault[1]]
                raise requests.ConnectionError("connection reset by peer")
            if fault and fault[0] == "status":
                return FakeResponse(fault[1], text="backend error")
            if fault and fault[0] == "expire":
                session["expired"] = True
                return FakeResponse(404, text="Not Found")
            session["data"] += chunk
            if fault and fault[0] == "lost_reply":
                if len(session["data"]) == session["total"]:
                    self._finish(session)
                raise requests.Timeout("read timed out")
            return self._finish(session) if len(session["data"]) == session["total"] else self._incomplete(session)
        if method == "GET" and url == f"{API_BASE}/channels":
            if "channels.list" in self.errors:
                return self.errors["channels.list"]
            return FakeResponse(200, payload={"items": [{"id": CHANNEL.id, "snippet": {"title": CHANNEL.title},
                                                        "status": {"longUploadsStatus": "allowed"}}]})
        if method == "GET" and url == f"{API_BASE}/videos":
            ids = params["id"].split(",")
            return FakeResponse(200, payload={"items": [{"id": i, "status": {"privacyStatus": "private"},
                                                        "snippet": {}} for i in ids if i != "missing"]})
        if method == "POST" and url == f"{UPLOAD_BASE}/thumbnails/set":
            if "thumbnails.set" in self.errors:
                return self.errors["thumbnails.set"]
            self.thumbnails[params["videoId"]] = {"data": kwargs["data"], "type": headers["Content-Type"]}
            return FakeResponse(200, payload={"kind": "youtube#thumbnailSetResponse"})
        if method == "POST" and url == f"{UPLOAD_BASE}/captions":
            if "captions.insert" in self.errors:
                return self.errors["captions.insert"]
            self.captions.append({"params": params, "type": headers["Content-Type"], "body": kwargs["data"]})
            return FakeResponse(200, payload={"id": "caption-1"})
        raise AssertionError(f"unexpected request {method} {url}")


def api_error(status: int, reason: str, message: str = "nope", domain: str = "youtube") -> FakeResponse:
    return FakeResponse(status, payload={"error": {"code": status, "message": message,
                                                   "errors": [{"domain": domain, "reason": reason, "message": message}]}})


# --------------------------------------------------------------------------- #
# a project that looks rendered (no FFmpeg involved)
# --------------------------------------------------------------------------- #
def upload_doc(example_doc: dict[str, Any], shorts: int = 2) -> dict[str, Any]:
    doc = approved(small_doc(example_doc, ["s001", "s002", "s003"]))
    doc["localization"]["custom_tracks"] = []
    all_shorts = [
        {"short_id": SH1, "scene_ids": ["s001", "s002"], "reframe": "blur_background",
         "hook_override_text": "This company lost $460 million because of one server.", "end_card_text": "More soon"},
        {"short_id": SH2, "scene_ids": ["s002", "s003"], "reframe": "center_crop"},
    ]
    doc["shorts"] = all_shorts[:shorts]
    return doc


def make_rendered_project(root: Path, doc: dict[str, Any], *, long_bytes: int = 700_000,
                          duration_sec: float = 78.0, mock: bool = False) -> tuple[RunContext, Path, str]:
    """State + files as ``pipeline run`` leaves them. Returns (context, script path, script sha256)."""
    ctx = make_context(root, doc, mock=True)
    run_thumbnail_stage(ctx)                                     # the real stage, mock picture
    ctx.mock = False
    script, state, paths = ctx.script, ctx.state, ctx.paths
    script_path = root / "script.json"
    sha = hashlib.sha256(script_path.read_bytes()).hexdigest()

    (paths.final / "long.mp4").write_bytes(os.urandom(long_bytes))
    state.artifacts["long"] = Artifact(hash="longhash-v1", path=ctx.settings.rel(paths.final / "long.mp4"))
    metadata = build_metadata(script, [(sec.chapter_title, 0.0) for sec in script.sections], duration_sec)
    paths.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    state.artifacts["metadata"] = Artifact(hash=stable_hash(metadata), path=ctx.settings.rel(paths.metadata))
    (paths.captions / "en.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nAugust 1st, 2012.\n", encoding="utf-8")
    state.timeline = TimelineInfo(fps=30, total_frames=int(duration_sec * 30), duration_sec=duration_sec)
    for index, short in enumerate(script.shorts):
        path = paths.shorts / f"{short.short_id}.mp4"
        path.write_bytes(os.urandom(120_000 + index))
        state.shorts[short.short_id] = ShortState(
            short_id=short.short_id, status="done", path=ctx.settings.rel(path), duration_sec=18.5,
            artifacts={"final": Artifact(hash=f"shorthash-{index}", path=ctx.settings.rel(path))})
    for stage in ("tts", "images", "thumbnail", "render", "shorts"):
        state.stage_record(stage).status = "done"
    state.script_sha256, state.mock = sha, mock
    ctx.save()
    return ctx, script_path, sha


def reload(ctx: RunContext) -> RunContext:
    """A fresh context over the same project, as a new process would see it."""
    ctx.state = ctx.store.load()
    return ctx


def log_text(ctx: RunContext) -> str:
    return ctx.paths.log_file.read_text(encoding="utf-8")
