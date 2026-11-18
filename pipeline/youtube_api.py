"""YouTube Data API v3 over plain HTTP: the five calls the upload command needs.

The client talks to an authorised ``requests``-style session (see
:mod:`pipeline.youtube_auth`), so everything here is testable with a fake
session and no network. ``YouTubeAPI`` is the interface the upload command
depends on; tests substitute an in-memory fake for it.

API facts checked against the reference on 2026-11-16 (pages last updated 2026-11-12):
* ``videos.insert`` — ``POST /upload/youtube/v3/videos``; scopes: youtube.upload, youtube,
  youtubepartner or youtube.force-ssl; max 256 GB, ``video/*`` or ``application/octet-stream``.
  Quota: 1 unit in its *own* "video uploads" bucket, 100 uploads a day by default (since 2026-06-01).
* Resumable protocol — ``POST ...?uploadType=resumable&part=...`` with ``X-Upload-Content-Length`` and
  ``X-Upload-Content-Type`` answers 200 + ``Location`` (the session URL). Bytes go by ``PUT`` with
  ``Content-Range: bytes first-last/total``; chunks are multiples of 256 KB, all the same size but the
  last. ``308`` + ``Range: bytes=0-N`` means "send from N+1"; no ``Range`` means nothing stored.
  ``PUT`` with ``Content-Range: bytes */total`` asks for the status. 500/502/503/504 and dropped
  connections are resumable; ``404`` means the session expired — start over.
* ``thumbnails.set`` — ``POST /upload/youtube/v3/thumbnails/set?videoId=``; ~50 units; JPEG/PNG, up to
  50 MB (2 MB until 2026-09-14 — the pipeline still keeps its thumbnail under 2 MB);
  403 ``forbidden`` when the channel may not use custom thumbnails; 429 ``uploadRateLimitExceeded``.
* ``captions.insert`` — ``POST /upload/youtube/v3/captions?part=snippet``; 400 units; scopes
  youtube.force-ssl or youtubepartner; needs ``snippet.videoId``, ``language``, ``name``
  (``isDraft`` optional); 409 ``captionExists``. The ``sync`` parameter is deprecated and not sent.
* ``videos.list`` and ``channels.list`` cost 1 unit; ``channels.list?mine=true`` needs authorisation.
  ``channels.status.longUploadsStatus`` is ``allowed`` / ``eligible`` (phone verification pending) /
  ``disallowed`` (channel not in good standing).
* ``snippet.defaultAudioLanguage`` is a documented video property, but neither ``videos.insert`` nor
  ``videos.update`` lists it among the settable ones (only ``snippet.defaultLanguage``). It is sent anyway
  and the upload command checks with ``videos.list`` whether YouTube kept it.
* All other calls share 10,000 units a day. (Daily quotas reset at midnight Pacific Time — long-standing
  Google API behaviour; the pages above do not restate it.)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import requests

from pipeline.http import RetryableAPIError, call_with_retries
from pipeline.logging_setup import get_logger

API_BASE = "https://www.googleapis.com/youtube/v3"
UPLOAD_BASE = "https://www.googleapis.com/upload/youtube/v3"
CHUNK_UNIT = 256 * 1024                      # resumable chunks must be multiples of this
RESUMABLE_STATUSES = (500, 502, 503, 504)    # per the resumable-upload guide
RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})
QUOTA_REASONS = frozenset({"quotaExceeded", "dailyLimitExceeded"})
LONG_UPLOAD_LIMIT_SEC = 15 * 60              # channels without phone verification cannot go beyond this

VERIFY_HINT = ("verify the channel by phone: https://www.youtube.com/verify (or YouTube Studio -> Settings -> "
               "Channel -> Feature eligibility -> Intermediate features -> Verify phone number)")
STANDING_HINT = ("YouTube reports longUploadsStatus 'disallowed': the channel is not in good standing (Community "
                 "Guidelines strikes or worldwide Content ID blocks). Resolve them in YouTube Studio; phone "
                 "verification alone does not help here")


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class YouTubeError(RuntimeError):
    """A YouTube problem the user can act on: ``hint`` says what to do."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class YouTubeAPIError(YouTubeError):
    def __init__(self, message: str, hint: str = "", *, method: str = "", status: int | None = None,
                 reason: str = "") -> None:
        super().__init__(message, hint)
        self.method = method
        self.status = status
        self.reason = reason


class QuotaExceeded(YouTubeAPIError):
    """Daily quota used up. ``bucket`` is ``"uploads"`` (videos.insert) or ``"general"`` (everything else)."""

    def __init__(self, message: str, hint: str, *, bucket: str, method: str) -> None:
        super().__init__(message, hint, method=method, status=403, reason="quotaExceeded")
        self.bucket = bucket


class UploadLimitExceeded(YouTubeAPIError):
    """The channel (not the API project) hit YouTube's own upload limit."""


class AuthExpired(YouTubeError):
    """The saved authorisation stopped working (expired or revoked token)."""


def explain_error(method: str, status: int, reason: str, message: str) -> YouTubeAPIError:
    """Turn an API error into an exception with a "what to do" hint."""
    text = f"YouTube {method}: HTTP {status} {reason or 'error'}" + (f" — {message}" if message else "")
    common = {"method": method, "status": status, "reason": reason}
    if reason in QUOTA_REASONS:
        if method == "videos.insert":
            return QuotaExceeded(
                "the daily UPLOAD quota of the Google Cloud project is used up",
                "videos.insert has its own quota bucket: 100 uploads a day per project by default. It resets at "
                "midnight Pacific Time. Run the same command again after that — everything already uploaded is "
                "skipped. To raise the limit, apply with the YouTube API Services Audit and Quota Extension Form "
                "(README, section about the audit).", bucket="uploads", method=method)
        return QuotaExceeded(
            f"the daily YouTube Data API quota of the Google Cloud project is used up (at {method})",
            "all calls except uploads share 10,000 units a day (thumbnails.set costs ~50, captions.insert 400, "
            "list calls 1). It resets at midnight Pacific Time. Run the same command again after that — only "
            "the missing parts are sent.", bucket="general", method=method)
    if reason == "uploadLimitExceeded":
        return UploadLimitExceeded(
            "YouTube says this channel has reached its upload limit",
            "this is YouTube's per-channel limit, not the API quota. Wait about 24 hours and run the same command "
            "again; a phone-verified channel with some history gets a higher limit.", **common)
    if reason == "youtubeSignupRequired":
        return YouTubeAPIError("this Google account has no YouTube channel",
                               "open https://www.youtube.com, create the channel, then run the command again.",
                               **common)
    if reason in ("accessNotConfigured", "SERVICE_DISABLED"):
        return YouTubeAPIError(
            "YouTube Data API v3 is not enabled in the Google Cloud project",
            "Google Cloud Console -> APIs & Services -> Library -> 'YouTube Data API v3' -> Enable (README, "
            "step 2), wait a minute and run again.", **common)
    if reason in ("insufficientPermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT"):
        return YouTubeAPIError(
            f"the saved authorisation does not cover {method}",
            "delete secrets/token_<channel_id>.json and run the command again to log in with the right "
            "permissions.", **common)
    if method == "thumbnails.set" and status == 403:
        return YouTubeAPIError(
            "YouTube refused the custom thumbnail: the channel may not set custom thumbnails",
            f"{VERIFY_HINT}. Then run the same command again — only the thumbnail will be uploaded.", **common)
    if method == "thumbnails.set" and status == 429:
        return YouTubeAPIError("the channel has uploaded too many thumbnails recently",
                               "wait an hour and run the same command again — only the thumbnail will be uploaded.",
                               **common)
    if method == "captions.insert" and status == 403:
        return YouTubeAPIError(text, "captions.insert needs the youtube.force-ssl scope: delete "
                                     "secrets/token_<channel_id>.json and log in again.", **common)
    if method == "videos.insert" and reason in ("invalidTitle", "invalidDescription", "invalidTags",
                                               "invalidCategoryId", "invalidPublishAt", "invalidVideoMetadata"):
        return YouTubeAPIError(text, "fix the metadata in the script, re-run `python -m pipeline run`, then upload "
                                     "again.", **common)
    return YouTubeAPIError(text, "", **common)


def parse_error(response: Any) -> tuple[str, str]:
    """(reason, message) from a Google JSON error body; empty strings when it is not one."""
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return "", ""
    if not isinstance(error, dict):
        return "", str(error)[:300]
    details = error.get("errors") or [{}]
    reason = str(details[0].get("reason") or error.get("status") or "")
    return reason, str(error.get("message") or details[0].get("message") or "")[:300]


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelInfo:
    id: str
    title: str
    # channels.status.longUploadsStatus: "allowed" — may upload videos longer than 15 minutes; "eligible" — may,
    # once the owner verifies the phone number; "disallowed" — may not (channel not in good standing).
    long_uploads_status: str | None = None

    @property
    def long_uploads_allowed(self) -> bool | None:
        """``None`` when YouTube did not say."""
        if not self.long_uploads_status or self.long_uploads_status == "longUploadsUnspecified":
            return None
        return self.long_uploads_status == "allowed"

    @property
    def long_uploads_hint(self) -> str:
        """What the owner has to do before long videos (and custom thumbnails) work."""
        return STANDING_HINT if self.long_uploads_status == "disallowed" else VERIFY_HINT


ProgressFn = Callable[[int, int], None]
SessionFn = Callable[[str], None]


class YouTubeAPI(ABC):
    """What the upload command needs from YouTube."""

    @abstractmethod
    def my_channel(self) -> ChannelInfo:
        """The channel the current authorisation belongs to (``channels.list?mine=true``)."""

    @abstractmethod
    def upload_video(self, path: Path, body: dict[str, Any], *, notify_subscribers: bool = True,
                     resume_url: str | None = None, on_session: SessionFn | None = None,
                     on_progress: ProgressFn | None = None) -> dict[str, Any]:
        """Resumable ``videos.insert``. Returns the created video resource (``id``, ``status``...)."""

    @abstractmethod
    def set_thumbnail(self, video_id: str, path: Path) -> None:
        ...

    @abstractmethod
    def insert_caption(self, video_id: str, path: Path, *, language: str, name: str) -> str:
        """Upload a published (non-draft) caption track; returns its ID."""

    @abstractmethod
    def list_videos(self, video_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """``videos.list`` (snippet + status) keyed by video ID; unknown IDs are simply missing."""


# --------------------------------------------------------------------------- #
# HTTP implementation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _UploadReply:
    kind: str                      # "done" | "incomplete" | "gone"
    offset: int = 0                # next byte to send (kind == "incomplete")
    resource: dict[str, Any] | None = None


class HttpYouTubeAPI(YouTubeAPI):
    """The real client. ``session`` is any object with ``requests.Session.request`` that adds authorisation."""

    def __init__(self, session: Any, *, attempts: int = 6, chunk_size: int = 8 * 1024 * 1024,
                 sleep: Callable[[float], None] = time.sleep, log: logging.Logger | None = None,
                 translate_exception: Callable[[BaseException], BaseException | None] | None = None) -> None:
        if chunk_size <= 0 or chunk_size % CHUNK_UNIT:
            raise ValueError("chunk_size must be a positive multiple of 256 KB")
        self._session = session
        self.attempts = max(1, attempts)
        self.chunk_size = chunk_size
        self._sleep = sleep
        self._log = log or get_logger("youtube")
        self._translate = translate_exception or (lambda exc: None)

    # ---- plumbing ---------------------------------------------------------
    def _raw(self, method: str, url: str, **kwargs: Any) -> Any:
        """One HTTP exchange. Network trouble becomes ``RetryableAPIError``; URLs and headers never leak."""
        kwargs.setdefault("timeout", (15, 120))
        kwargs.setdefault("allow_redirects", False)     # 308 is "resume incomplete" here, not a redirect
        try:
            return self._session.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
            raise RetryableAPIError(f"YouTube: network error ({type(exc).__name__})") from None
        except Exception as exc:  # noqa: BLE001 — the auth layer knows its own exception types
            translated = self._translate(exc)
            if translated is None:
                raise
            raise translated from None

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        try:
            return min(float(response.headers.get("Retry-After", "")), 120.0)
        except (TypeError, ValueError):
            return None

    def _check(self, response: Any, api_method: str) -> Any:
        """Map a non-2xx answer of a regular (non-upload-session) call onto an exception."""
        status = response.status_code
        if 200 <= status < 300:
            return response
        reason, message = parse_error(response)
        if status >= 500 or (status == 403 and reason in RATE_LIMIT_REASONS) or \
                (status == 429 and reason != "uploadRateLimitExceeded"):
            raise RetryableAPIError(f"YouTube {api_method}: HTTP {status} {reason}".rstrip(), status=status,
                                    retry_after=self._retry_after(response))
        if status == 401 and reason != "youtubeSignupRequired":
            raise AuthExpired("YouTube no longer accepts the saved authorisation (HTTP 401)")
        raise explain_error(api_method, status, reason, message)

    def _call(self, method: str, url: str, *, api_method: str, **kwargs: Any) -> Any:
        """A regular API call with retries on 5xx / rate limits / network errors."""
        try:
            return call_with_retries(lambda: self._check(self._raw(method, url, **kwargs), api_method),
                                     what=f"YouTube {api_method}", log=self._log, attempts=self.attempts,
                                     sleep=self._sleep)
        except RetryableAPIError as exc:
            raise YouTubeAPIError(f"{exc} (gave up after {self.attempts} attempts)",
                                  "this looks temporary: check the internet connection and run the same command "
                                  "again — finished steps are skipped.", method=api_method,
                                  status=exc.status) from None

    # ---- simple calls -----------------------------------------------------
    def my_channel(self) -> ChannelInfo:
        response = self._call("GET", f"{API_BASE}/channels", api_method="channels.list",
                              params={"part": "snippet,status", "mine": "true"})
        items = response.json().get("items") or []
        if not items:
            raise YouTubeAPIError("this Google account has no YouTube channel",
                                  "open https://www.youtube.com, create the channel, then run the command again.",
                                  method="channels.list")
        item = items[0]
        return ChannelInfo(id=str(item["id"]), title=str(item.get("snippet", {}).get("title", "")),
                           long_uploads_status=item.get("status", {}).get("longUploadsStatus"))

    def list_videos(self, video_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        ids = list(video_ids)
        for start in range(0, len(ids), 50):          # the API takes at most 50 IDs per call
            response = self._call("GET", f"{API_BASE}/videos", api_method="videos.list",
                                  params={"part": "snippet,status", "id": ",".join(ids[start:start + 50])})
            for item in response.json().get("items") or []:
                found[str(item["id"])] = item
        return found

    def set_thumbnail(self, video_id: str, path: Path) -> None:
        content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        self._call("POST", f"{UPLOAD_BASE}/thumbnails/set", api_method="thumbnails.set",
                   params={"videoId": video_id, "uploadType": "media"},
                   headers={"Content-Type": content_type}, data=path.read_bytes())

    def insert_caption(self, video_id: str, path: Path, *, language: str, name: str) -> str:
        metadata = {"snippet": {"videoId": video_id, "language": language, "name": name, "isDraft": False}}
        boundary = f"pipeline-{uuid.uuid4().hex}"
        body = b"".join([
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode(),
            json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
            f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        response = self._call("POST", f"{UPLOAD_BASE}/captions", api_method="captions.insert",
                              params={"part": "snippet", "uploadType": "multipart"},
                              headers={"Content-Type": f"multipart/related; boundary={boundary}"}, data=body)
        return str(response.json().get("id", ""))

    # ---- resumable upload -------------------------------------------------
    def _start_session(self, body: dict[str, Any], total: int, content_type: str, notify_subscribers: bool) -> str:
        response = self._call(
            "POST", f"{UPLOAD_BASE}/videos", api_method="videos.insert",
            params={"uploadType": "resumable", "part": ",".join(sorted(body)),
                    "notifySubscribers": "true" if notify_subscribers else "false"},
            headers={"Content-Type": "application/json; charset=UTF-8", "X-Upload-Content-Length": str(total),
                     "X-Upload-Content-Type": content_type},
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"))
        location = response.headers.get("Location") or response.headers.get("location")
        if not location:
            raise YouTubeAPIError("YouTube videos.insert: no upload session URL in the answer", method="videos.insert")
        return str(location)

    def _upload_put(self, url: str, headers: dict[str, str], data: bytes) -> _UploadReply:
        response = self._raw("PUT", url, headers=headers, data=data, timeout=(15, 600))
        status = response.status_code
        if status in (200, 201):
            return _UploadReply("done", resource=response.json())
        if status == 308:
            value = response.headers.get("Range") or response.headers.get("range") or ""
            try:
                return _UploadReply("incomplete", offset=int(value.rsplit("-", 1)[1]) + 1 if value else 0)
            except (IndexError, ValueError):
                raise RetryableAPIError("YouTube upload: unreadable Range header") from None
        if status == 404:
            return _UploadReply("gone")
        if status in RESUMABLE_STATUSES:
            raise RetryableAPIError(f"YouTube upload: HTTP {status}", status=status,
                                    retry_after=self._retry_after(response))
        reason, message = parse_error(response)
        if status == 401:
            raise AuthExpired("YouTube no longer accepts the saved authorisation (HTTP 401)")
        raise explain_error("videos.insert", status, reason, message)

    def upload_video(self, path: Path, body: dict[str, Any], *, notify_subscribers: bool = True,
                     resume_url: str | None = None, on_session: SessionFn | None = None,
                     on_progress: ProgressFn | None = None) -> dict[str, Any]:
        total = path.stat().st_size
        if total <= 0:
            raise YouTubeError(f"{path.name} is empty — nothing to upload")
        content_type = "video/mp4" if path.suffix.lower() in (".mp4", ".m4v") else "application/octet-stream"

        def start() -> str:
            new_url = self._start_session(body, total, content_type, notify_subscribers)
            if on_session:
                on_session(new_url)
            return new_url

        url = resume_url or start()
        ask_status = resume_url is not None      # a session from an earlier run: ask what the server has
        restarted = resume_url is None
        offset, failures = 0, 0
        with path.open("rb") as fh:
            while True:
                try:
                    if ask_status:
                        reply = self._upload_put(url, {"Content-Length": "0", "Content-Range": f"bytes */{total}"}, b"")
                    else:
                        fh.seek(offset)
                        chunk = fh.read(self.chunk_size)
                        reply = self._upload_put(url, {
                            "Content-Length": str(len(chunk)), "Content-Type": content_type,
                            "Content-Range": f"bytes {offset}-{offset + len(chunk) - 1}/{total}"}, chunk)
                except RetryableAPIError as exc:
                    failures += 1
                    if failures >= self.attempts:
                        raise YouTubeAPIError(
                            f"upload of {path.name} keeps failing ({exc}); stopped at {offset} of {total} bytes",
                            "check the internet connection and run the same command again — the upload continues "
                            "from where it stopped.", method="videos.insert", status=exc.status) from None
                    delay = max(min(2.0 ** (failures - 1), 30.0), exc.retry_after or 0.0)
                    self._log.warning("upload %s: %s — retry %d/%d in %.0f s, then continue from the last byte "
                                      "YouTube confirmed", path.name, exc, failures, self.attempts - 1, delay)
                    self._sleep(delay)
                    ask_status = True            # never assume how much of the chunk arrived
                    continue

                if reply.kind == "done":
                    if on_progress:
                        on_progress(total, total)
                    return reply.resource or {}
                if reply.kind == "gone":         # 404: the session expired
                    if restarted:
                        raise YouTubeAPIError(f"upload of {path.name}: YouTube does not recognise the upload session",
                                              "run the same command again.", method="videos.insert", status=404)
                    self._log.warning("upload %s: the saved upload session has expired — starting over", path.name)
                    url, restarted, ask_status, offset, failures = start(), True, False, 0, 0
                    continue
                if reply.offset > total:
                    raise YouTubeAPIError(f"upload of {path.name}: YouTube reports more bytes than the file has",
                                          "the file changed during the upload — run the command again.",
                                          method="videos.insert")
                if ask_status:
                    self._log.info("upload %s: continuing from byte %d of %d", path.name, reply.offset, total)
                elif reply.offset <= offset:     # a chunk was accepted but nothing was stored
                    failures += 1
                    if failures >= self.attempts:
                        raise YouTubeAPIError(f"upload of {path.name} makes no progress at byte {offset}",
                                              "run the same command again.", method="videos.insert")
                else:
                    failures = 0
                offset, ask_status = reply.offset, False
                if on_progress:
                    on_progress(offset, total)
