"""``HttpYouTubeAPI`` over real HTTP: ``requests`` talks to a server on 127.0.0.1 (never to the internet).

The other upload tests replace the HTTP session with a fake, which cannot notice transport-level mistakes
(a 308 followed as a redirect, a wrong ``Content-Length``, a half-sent chunk). Here the bytes really travel
through sockets, the server kills one connection in the middle of a chunk and answers 503 once.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import pytest
import requests

import pipeline.youtube_api as youtube_api
from pipeline.logging_setup import setup_logging
from pipeline.youtube_api import CHUNK_UNIT, HttpYouTubeAPI

KEPT_BEFORE_DROP = 70_000          # bytes of the broken chunk the server manages to store


class _State:
    def __init__(self) -> None:
        self.data = bytearray()
        self.total = 0
        self.puts: list[tuple[int, int, str]] = []          # (first byte, last byte, what the server did)
        self.status_queries = 0
        self.faults = ["ok", "drop", "503"]                 # applied to chunk PUTs, in this order
        self.resource: dict[str, Any] | None = None
        self.start: dict[str, Any] = {}
        self.thumbnail: dict[str, Any] = {}
        self.caption: dict[str, Any] = {}
        self.authorizations: set[str | None] = set()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: _State

    def log_message(self, *args: Any) -> None:               # keep pytest output clean
        pass

    def _reply(self, code: int, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if body:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _incomplete(self) -> None:
        stored = len(self.state.data)
        self._reply(308, headers={"Range": f"bytes=0-{stored - 1}"} if stored else {})

    def do_GET(self) -> None:  # noqa: N802 — http.server naming
        self.state.authorizations.add(self.headers.get("Authorization"))
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/youtube/v3/channels" and query.get("mine") == ["true"]:
            self._reply(200, {"items": [{"id": "UC" + "c" * 22, "snippet": {"title": "Loopback"},
                                         "status": {"longUploadsStatus": "allowed"}}]})
        elif url.path == "/youtube/v3/videos":
            self._reply(200, {"items": [{"id": i, "status": {"privacyStatus": "private"}, "snippet": {}}
                                        for i in query["id"][0].split(",")]})
        else:
            self._reply(404, {"error": {"code": 404, "message": "unknown"}})

    def do_POST(self) -> None:  # noqa: N802
        self.state.authorizations.add(self.headers.get("Authorization"))
        url = urlparse(self.path)
        query = parse_qs(url.query)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if url.path == "/upload/youtube/v3/videos":
            self.state.start = {"query": query, "body": json.loads(body),
                                "length": self.headers["X-Upload-Content-Length"],
                                "type": self.headers["X-Upload-Content-Type"]}
            self.state.total = int(self.headers["X-Upload-Content-Length"])
            port = self.server.server_address[1]
            self._reply(200, headers={"Location": f"http://127.0.0.1:{port}/session/1?upload_id=LOOPBACK"})
        elif url.path == "/upload/youtube/v3/thumbnails/set":
            self.state.thumbnail = {"query": query, "type": self.headers["Content-Type"], "data": body}
            self._reply(200, {"kind": "youtube#thumbnailSetResponse"})
        elif url.path == "/upload/youtube/v3/captions":
            self.state.caption = {"query": query, "type": self.headers["Content-Type"], "data": body}
            self._reply(200, {"id": "caption-loopback"})
        else:
            self._reply(404, {"error": {"code": 404, "message": "unknown"}})

    def do_PUT(self) -> None:  # noqa: N802
        state = self.state
        state.authorizations.add(self.headers.get("Authorization"))
        content_range = self.headers["Content-Range"]
        length = int(self.headers.get("Content-Length", 0))
        if content_range.startswith("bytes */"):             # "how much do you have?"
            state.status_queries += 1
            if state.resource:
                self._reply(201, state.resource)
            else:
                self._incomplete()
            return
        span, total = content_range[len("bytes "):].split("/")
        first, last = (int(x) for x in span.split("-"))
        fault = state.faults.pop(0) if state.faults else "ok"
        state.puts.append((first, last, fault))
        if fault == "drop":                                  # part of the chunk arrives, then the line is cut
            state.data += self.rfile.read(KEPT_BEFORE_DROP)
            self.close_connection = True
            self.connection.shutdown(2)
            return
        chunk = self.rfile.read(length)
        if fault == "503":
            self._reply(503, {"error": {"code": 503, "message": "backend error"}})
            return
        if int(total) != state.total or last - first + 1 != length or first != len(state.data):
            self._reply(400, {"error": {"code": 400, "message": f"bad range {content_range}, have {len(state.data)}"}})
            return
        state.data += chunk
        if len(state.data) == state.total:
            state.resource = {"id": "vidLOOPBACK1", "status": {"uploadStatus": "uploaded"}}
            self._reply(201, state.resource)
        else:
            self._incomplete()


@pytest.fixture()
def loopback(monkeypatch: pytest.MonkeyPatch) -> Iterator[_State]:
    state = _State()
    handler = type("Handler", (_Handler,), {"state": state})
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError as exc:                                   # a sandbox without loopback sockets
        pytest.skip(f"cannot open a loopback socket: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(youtube_api, "API_BASE", f"{base}/youtube/v3")
    monkeypatch.setattr(youtube_api, "UPLOAD_BASE", f"{base}/upload/youtube/v3")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)              # 127.0.0.1 must not go through a proxy
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    yield state
    server.shutdown()
    server.server_close()


def test_real_http_resumable_upload_survives_a_cut_connection_and_a_503(tmp_path: Path, loopback: _State) -> None:
    session = requests.Session()
    session.headers["Authorization"] = "Bearer loopback-token"     # what the authorised session adds for real
    sleeps: list[float] = []
    api = HttpYouTubeAPI(session, attempts=5, chunk_size=CHUNK_UNIT, sleep=sleeps.append,
                         log=setup_logging(tmp_path / "api.log", ["loopback-token"]))
    video = tmp_path / "long.mp4"
    video.write_bytes(os.urandom(1_000_000))
    body = {"snippet": {"title": "Тест ✓", "description": "d", "tags": ["a"], "categoryId": "28",
                        "defaultLanguage": "en-US", "defaultAudioLanguage": "en-US"},
            "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": False}}

    assert api.my_channel().title == "Loopback"
    resource = api.upload_video(video, body, notify_subscribers=False)

    assert resource["id"] == "vidLOOPBACK1"
    assert hashlib.sha256(bytes(loopback.data)).digest() == hashlib.sha256(video.read_bytes()).digest()   # byte-exact
    assert loopback.start == {"query": {"uploadType": ["resumable"], "part": ["snippet,status"],
                                        "notifySubscribers": ["false"]},
                              "body": body, "length": "1000000", "type": "video/mp4"}
    # chunk 1 ok; chunk 2 cut after 70 000 bytes -> status query -> resume right after them; then a 503 -> again
    assert [(first, what) for first, _, what in loopback.puts[:4]] == [
        (0, "ok"), (CHUNK_UNIT, "drop"), (CHUNK_UNIT + KEPT_BEFORE_DROP, "503"), (CHUNK_UNIT + KEPT_BEFORE_DROP, "ok")]
    assert loopback.status_queries == 2 and sleeps == [1.0, 2.0]
    assert all((last - first + 1) % CHUNK_UNIT == 0 for first, last, _ in loopback.puts[:-1])   # 256 KB multiples

    thumbnail = tmp_path / "thumbnail.jpg"
    thumbnail.write_bytes(b"\xff\xd8 not really a jpeg")
    api.set_thumbnail("vidLOOPBACK1", thumbnail)
    assert loopback.thumbnail == {"query": {"videoId": ["vidLOOPBACK1"], "uploadType": ["media"]},
                                  "type": "image/jpeg", "data": thumbnail.read_bytes()}

    captions = tmp_path / "en.srt"
    captions.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello — мир\n", encoding="utf-8")
    assert api.insert_caption("vidLOOPBACK1", captions, language="en", name="English") == "caption-loopback"
    content_type = loopback.caption["type"]
    assert content_type.startswith("multipart/related; boundary=")
    parts = loopback.caption["data"].split(b"--" + content_type.split("boundary=")[1].encode())
    assert json.loads(parts[1].split(b"\r\n\r\n", 1)[1]) == {
        "snippet": {"videoId": "vidLOOPBACK1", "language": "en", "name": "English", "isDraft": False}}
    assert parts[2].split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0] == captions.read_bytes()
    assert loopback.caption["query"] == {"part": ["snippet"], "uploadType": ["multipart"]}

    assert list(api.list_videos(["vidLOOPBACK1"])) == ["vidLOOPBACK1"]
    assert loopback.authorizations == {"Bearer loopback-token"}              # on every request, incl. chunk PUTs
    assert "loopback-token" not in (tmp_path / "api.log").read_text(encoding="utf-8")
