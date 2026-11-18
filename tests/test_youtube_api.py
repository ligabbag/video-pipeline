"""HTTP level: the resumable upload protocol, request shapes and error mapping of ``HttpYouTubeAPI``."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pipeline.logging_setup import setup_logging
from pipeline.youtube_api import (CHUNK_UNIT, AuthExpired, HttpYouTubeAPI, QuotaExceeded, UploadLimitExceeded,
                                  YouTubeAPIError, explain_error)
from tests.conftest import FakeResponse, FakeSession
from tests.youtube_fakes import CHANNEL, FakeUploadServer, api_error, no_network  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")

BODY = {"snippet": {"title": "T", "description": "D", "tags": ["a"], "categoryId": "28"},
        "status": {"privacyStatus": "private"}}
CHUNK = 2 * CHUNK_UNIT                     # 512 KB chunks -> several requests for a 1.3 MB file


def make_api(server: FakeUploadServer, tmp_path: Path, attempts: int = 5) -> tuple[HttpYouTubeAPI, list[float]]:
    sleeps: list[float] = []
    log = setup_logging(tmp_path / "api.log", [])
    api = HttpYouTubeAPI(FakeSession(server), attempts=attempts, chunk_size=CHUNK, sleep=sleeps.append, log=log)
    return api, sleeps


def video_file(tmp_path: Path, size: int = 1_300_000) -> Path:
    path = tmp_path / "long.mp4"
    path.write_bytes(os.urandom(size))
    return path


def test_resumable_upload_sends_equal_256k_multiples_and_returns_the_video(tmp_path: Path) -> None:
    server = FakeUploadServer()
    api, _ = make_api(server, tmp_path)
    path = video_file(tmp_path)
    sessions: list[str] = []
    progress: list[tuple[int, int]] = []
    resource = api.upload_video(path, BODY, notify_subscribers=False, on_session=sessions.append,
                                on_progress=lambda sent, total: progress.append((sent, total)))

    assert resource["id"] == "vid00000001" and server.data() == path.read_bytes()
    start = server.initiations[0]
    assert start["params"] == {"uploadType": "resumable", "part": "snippet,status", "notifySubscribers": "false"}
    assert start["headers"]["X-Upload-Content-Length"] == "1300000"
    assert start["headers"]["X-Upload-Content-Type"] == "video/mp4"
    assert start["headers"]["Content-Type"].startswith("application/json")
    assert list(server.sessions.values())[0]["body"] == BODY
    sizes = [last - first + 1 for first, last, _ in server.chunk_puts]
    assert sizes[:-1] == [CHUNK] * (len(sizes) - 1) and CHUNK % (256 * 1024) == 0 and sum(sizes) == 1_300_000
    assert len(sessions) == 1 and len(server.initiations) == 1 and server.status_queries == 0
    assert [sent for sent, _ in progress] == sorted(sent for sent, _ in progress) and progress[-1] == (1_300_000, 1_300_000)


def test_upload_continues_from_the_confirmed_byte_after_a_dropped_connection(tmp_path: Path) -> None:
    """Connection dies in the middle of chunk 2 after the server kept only part of it."""
    server = FakeUploadServer()
    server.faults = [("ok", 0), ("drop", 100_000)]             # chunk 1 fine; chunk 2: 100 000 bytes kept, then reset
    api, sleeps = make_api(server, tmp_path)
    path = video_file(tmp_path)
    resource = api.upload_video(path, BODY)

    assert resource["id"] and server.data() == path.read_bytes()          # byte-exact, nothing doubled or skipped
    assert len(server.initiations) == 1                                    # same session: resumed, not restarted
    assert server.status_queries == 1 and sleeps == [1.0]                  # asked the server, backed off once
    firsts = [first for first, _, _ in server.chunk_puts]
    assert firsts[:3] == [0, CHUNK, CHUNK + 100_000]                       # resumed exactly after the kept bytes
    assert "retry 1/4" in (tmp_path / "api.log").read_text(encoding="utf-8")


def test_upload_survives_5xx_and_a_lost_final_answer(tmp_path: Path) -> None:
    server = FakeUploadServer()
    server.faults = [("status", 503), ("status", 502), ("ok", 0), ("ok", 0), ("lost_reply", 0)]
    api, sleeps = make_api(server, tmp_path)
    path = video_file(tmp_path)
    resource = api.upload_video(path, BODY)
    # the last chunk arrived but its answer was lost: the status query returns the finished video
    assert resource["id"] == "vid00000001" and server.data() == path.read_bytes()
    assert len(server.initiations) == 1 and server.status_queries == 3 and sleeps == [1.0, 2.0, 1.0]


def test_interrupted_upload_resumes_in_a_later_run_from_the_saved_session(tmp_path: Path) -> None:
    server = FakeUploadServer()
    server.faults = [("ok", 0)] + [("drop", 0)] * 20            # after chunk 1 the network is gone for good
    api, _ = make_api(server, tmp_path, attempts=3)
    path = video_file(tmp_path)
    sessions: list[str] = []
    with pytest.raises(YouTubeAPIError) as failure:
        api.upload_video(path, BODY, on_session=sessions.append)
    assert "keeps failing" in str(failure.value) and f"stopped at {CHUNK} of 1300000" in str(failure.value)
    assert "continues from where it stopped" in failure.value.hint
    assert "upload.test" not in str(failure.value) + failure.value.hint    # the session URL never leaks
    assert len(server.data()) == CHUNK

    # "next run": a new client gets the saved session URL, asks for the status and sends only the rest
    server.faults = []
    api2, _ = make_api(server, tmp_path)
    before = len(server.chunk_puts)
    resource = api2.upload_video(path, BODY, resume_url=sessions[0], on_session=sessions.append)
    assert resource["id"] and server.data() == path.read_bytes()
    assert len(server.initiations) == 1 and len(sessions) == 1             # no second videos.insert
    assert server.chunk_puts[before][0] == CHUNK                            # first new chunk starts where it stopped
    assert "continuing from byte 524288" in (tmp_path / "api.log").read_text(encoding="utf-8")


def test_expired_session_starts_over_once(tmp_path: Path) -> None:
    server = FakeUploadServer()
    api, _ = make_api(server, tmp_path)
    path = video_file(tmp_path)
    sessions: list[str] = []
    resource = api.upload_video(path, BODY, resume_url="https://upload.test/session/99?upload_id=OLD",
                                on_session=sessions.append)
    assert resource["id"] and server.data() == path.read_bytes()
    assert len(server.initiations) == 1 and len(sessions) == 1             # 404 on the old one -> one fresh session
    assert "expired" in (tmp_path / "api.log").read_text(encoding="utf-8")


def test_quota_errors_tell_uploads_from_everything_else(tmp_path: Path) -> None:
    server = FakeUploadServer()
    api, _ = make_api(server, tmp_path)
    path = video_file(tmp_path, 300_000)

    server.errors["videos.insert"] = api_error(403, "quotaExceeded", domain="youtube.quota")
    with pytest.raises(QuotaExceeded) as uploads:
        api.upload_video(path, BODY)
    assert uploads.value.bucket == "uploads" and "UPLOAD quota" in str(uploads.value)
    assert "100 uploads a day" in uploads.value.hint and "midnight Pacific" in uploads.value.hint

    server.errors["thumbnails.set"] = api_error(403, "quotaExceeded", domain="youtube.quota")
    thumb = tmp_path / "thumbnail.jpg"
    thumb.write_bytes(b"\xff\xd8jpeg")
    with pytest.raises(QuotaExceeded) as general:
        api.set_thumbnail("vid1", thumb)
    assert general.value.bucket == "general" and "10,000 units" in general.value.hint
    assert str(uploads.value) != str(general.value)

    server.errors["videos.insert"] = api_error(400, "uploadLimitExceeded")
    with pytest.raises(UploadLimitExceeded) as limit:
        api.upload_video(path, BODY)
    assert "per-channel limit" in limit.value.hint


def test_thumbnail_request_and_the_unverified_channel_error(tmp_path: Path) -> None:
    server = FakeUploadServer()
    api, _ = make_api(server, tmp_path)
    thumb = tmp_path / "thumbnail.jpg"
    thumb.write_bytes(b"\xff\xd8\xff\xe0 jpeg bytes")
    api.set_thumbnail("vid1", thumb)
    assert server.thumbnails["vid1"] == {"data": thumb.read_bytes(), "type": "image/jpeg"}

    server.errors["thumbnails.set"] = api_error(403, "forbidden", "The authenticated user doesn't have permissions")
    with pytest.raises(YouTubeAPIError) as failure:
        api.set_thumbnail("vid1", thumb)
    assert "custom thumbnail" in str(failure.value)
    assert "youtube.com/verify" in failure.value.hint and "only the thumbnail" in failure.value.hint


def test_captions_request_is_a_published_english_track(tmp_path: Path) -> None:
    server = FakeUploadServer()
    api, _ = make_api(server, tmp_path)
    srt = tmp_path / "en.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello — $460 million.\n", encoding="utf-8")
    assert api.insert_caption("vid1", srt, language="en", name="English") == "caption-1"
    sent = server.captions[0]
    assert sent["params"] == {"part": "snippet", "uploadType": "multipart"}        # no deprecated `sync`
    boundary = sent["type"].split("boundary=")[1]
    assert sent["type"].startswith("multipart/related")
    parts = sent["body"].split(f"--{boundary}".encode())
    meta = json.loads(parts[1].split(b"\r\n\r\n", 1)[1])
    assert meta == {"snippet": {"videoId": "vid1", "language": "en", "name": "English", "isDraft": False}}
    assert parts[2].split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n") == srt.read_bytes()   # byte-exact file
    assert parts[3].startswith(b"--")

    server.errors["captions.insert"] = api_error(409, "captionExists")
    with pytest.raises(YouTubeAPIError) as failure:
        api.insert_caption("vid1", srt, language="en", name="English")
    assert failure.value.reason == "captionExists" and failure.value.status == 409


def test_channel_and_video_lookups(tmp_path: Path) -> None:
    server = FakeUploadServer()
    api, _ = make_api(server, tmp_path)
    channel = api.my_channel()
    assert (channel.id, channel.title, channel.long_uploads_allowed) == (CHANNEL.id, "Tech Disasters", True)
    method, url, kwargs = server.requests[-1]
    assert kwargs["params"] == {"part": "snippet,status", "mine": "true"}
    assert set(api.list_videos(["a", "missing", "b"])) == {"a", "b"}

    server.errors["channels.list"] = FakeResponse(200, payload={"items": []})
    with pytest.raises(YouTubeAPIError, match="no YouTube channel"):
        api.my_channel()
    server.errors["channels.list"] = api_error(401, "authError", "Invalid Credentials")
    with pytest.raises(AuthExpired):
        api.my_channel()
    server.errors["channels.list"] = api_error(403, "accessNotConfigured")
    with pytest.raises(YouTubeAPIError) as disabled:
        api.my_channel()
    assert "not enabled" in str(disabled.value) and "Enable" in disabled.value.hint


def test_transient_api_errors_are_retried_then_reported(tmp_path: Path) -> None:
    answers = [api_error(503, "backendError"), api_error(403, "rateLimitExceeded"),
               FakeResponse(200, payload={"items": [{"id": "UCx", "snippet": {"title": "X"}}]})]
    sleeps: list[float] = []
    api = HttpYouTubeAPI(FakeSession(lambda m, u, k: answers.pop(0)), attempts=4, sleep=sleeps.append,
                         log=setup_logging(tmp_path / "api.log", []))
    assert api.my_channel().id == "UCx" and len(sleeps) == 2

    always_down = HttpYouTubeAPI(FakeSession(lambda m, u, k: api_error(500, "backendError")), attempts=3,
                                 sleep=sleeps.append, log=setup_logging(tmp_path / "api.log", []))
    with pytest.raises(YouTubeAPIError) as failure:
        always_down.list_videos(["a"])
    assert "gave up after 3 attempts" in str(failure.value) and "run the same command again" in failure.value.hint


def test_explain_error_keeps_unknown_reasons_readable() -> None:
    error = explain_error("videos.insert", 400, "invalidTitle", "The request metadata specifies an invalid title.")
    assert "invalidTitle" in str(error) and "fix the metadata" in error.hint
    plain = explain_error("videos.list", 418, "", "")
    assert str(plain) == "YouTube videos.list: HTTP 418 error" and plain.hint == ""
    with pytest.raises(ValueError):
        HttpYouTubeAPI(object(), chunk_size=1000)                              # not a multiple of 256 KB
