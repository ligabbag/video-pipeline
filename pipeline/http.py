"""Shared HTTP plumbing for external APIs: retries with exponential backoff (tenacity).

429 and 5xx responses and network errors are retried; other 4xx are fatal.
Request headers (which carry API keys) are never logged or put into exceptions.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

T = TypeVar("T")


class APIError(RuntimeError):
    """Non-retryable API failure (bad request, auth, moderation...)."""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class RetryableAPIError(RuntimeError):
    """Transient failure: rate limit, server error or network problem."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _retry_after(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    try:
        return min(float(raw), 120.0) if raw else None
    except ValueError:
        return None


def send(session: requests.Session, method: str, url: str, *, service: str, **kwargs: Any) -> requests.Response:
    """One HTTP call; maps failures onto :class:`RetryableAPIError` / :class:`APIError`."""
    kwargs.setdefault("timeout", (10, 180))
    try:
        response = session.request(method, url, **kwargs)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise RetryableAPIError(f"{service}: network error ({type(exc).__name__})") from None
    status = response.status_code
    if status == 429 or status >= 500:
        raise RetryableAPIError(f"{service}: HTTP {status}", status=status, retry_after=_retry_after(response))
    if status >= 400:
        body = (response.text or "")[:400].replace("\n", " ")
        hint = " — check the API key" if status in (401, 403) else ""
        raise APIError(f"{service}: HTTP {status}{hint}: {body}", status=status, body=response.text or "")
    return response


def call_with_retries(fn: Callable[[], T], *, what: str, log: logging.Logger, attempts: int = 6,
                      sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn`` and retry transient failures with exponential backoff + jitter.

    Honours ``Retry-After`` when the server sends it.
    """
    backoff = wait_exponential_jitter(initial=1.0, max=30.0)

    def wait(retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        delay = max(float(backoff(retry_state)), float(getattr(exc, "retry_after", None) or 0.0))
        log.warning("%s: %s — retry %d/%d in %.1f s", what, exc, retry_state.attempt_number, attempts - 1, delay)
        return delay

    retryer = Retrying(stop=stop_after_attempt(attempts), wait=wait,
                       retry=retry_if_exception_type(RetryableAPIError), reraise=True, sleep=sleep)
    return retryer(fn)
