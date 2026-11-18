"""Google OAuth 2.0 for the upload command ("Desktop app" client, local redirect).

* ``secrets/client_secret.json`` — the OAuth client downloaded from Google Cloud Console.
* ``secrets/token_<channel_id>.json`` — one token per *script* ``channel_id``, so one pipeline can
  serve several channels. Besides the credentials the file records which YouTube channel they belong
  to; every later run compares that with what Google reports and refuses to work on a mismatch.

Nothing secret is ever logged: tokens and the client secret are handed to the log redactor, and
errors from Google are reduced to their error code.

OAuth facts checked on 2026-11-16:
* Scope: ``youtube.force-ssl`` alone is accepted by every call used here (videos.insert,
  thumbnails.set, captions.insert, channels.list, videos.list). ``youtube.upload`` adds nothing on top
  of it, and on its own it cannot read the channel — so it is not requested (override with
  ``YOUTUBE_OAUTH_SCOPES``).
* An External app in publishing status "Testing" gets refresh tokens that expire 7 days after
  consent; "In production" removes that limit (Google Auth Platform -> Audience).
* Desktop clients use a loopback redirect (``http://localhost:<port>``); nothing to configure in the console.

The Google libraries (``google-auth``, ``google-auth-oauthlib``) are imported lazily, so ``--dry-run``
and the tests work without them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from pipeline.config import Settings, channel_pin_key
from pipeline.http import RetryableAPIError
from pipeline.state import now_iso
from pipeline.youtube_api import AuthExpired, ChannelInfo, HttpYouTubeAPI, YouTubeAPI

DEFAULT_SCOPES: tuple[str, ...] = ("https://www.googleapis.com/auth/youtube.force-ssl",)
TOKEN_VERSION = 1
_SAFE_CHANNEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_YOUTUBE_CHANNEL = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

TESTING_HINT = ("The usual cause: the OAuth app is in publishing status 'Testing', where Google expires the "
                "refresh token 7 days after login. To stop that, open Google Cloud Console -> Google Auth Platform "
                "-> Audience and press 'Publish app' (status 'In production'); for personal use no Google review is "
                "needed. Other causes: access removed at https://myaccount.google.com/permissions, 6 months "
                "without use, or too many newer logins with the same client. Run the same command again to log in.")


class AuthError(RuntimeError):
    """Authorisation problem with a "what to do" hint. ``refused`` -> the command exits with code 2."""

    refused = False

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class ClientSecretMissing(AuthError):
    refused = True


class ChannelMismatch(AuthError):
    refused = True


class TokenExpired(AuthError):
    """The refresh token was rejected (``invalid_grant``); the token file has been deleted."""


class InvalidGrant(Exception):
    """Raised by a backend when Google answers ``invalid_grant`` to a token refresh."""


# --------------------------------------------------------------------------- #
# token files
# --------------------------------------------------------------------------- #
@dataclass
class TokenFile:
    path: Path
    pipeline_channel_id: str
    youtube_channel_id: str
    youtube_channel_title: str
    scopes: list[str]
    credentials: dict[str, Any] = field(repr=False)
    created_at: str = field(default_factory=now_iso)


def token_path(secrets_dir: Path, channel_id: str) -> Path:
    if not _SAFE_CHANNEL.match(channel_id):
        raise AuthError(f"channel_id '{channel_id}' cannot be used in a file name",
                        "use only letters, digits, '.', '_' and '-' in the script's channel_id.")
    return secrets_dir / f"token_{channel_id}.json"


def load_token(path: Path) -> TokenFile | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TokenFile(path=path, pipeline_channel_id=str(raw["pipeline_channel_id"]),
                         youtube_channel_id=str(raw["youtube_channel_id"]),
                         youtube_channel_title=str(raw.get("youtube_channel_title", "")),
                         scopes=[str(s) for s in raw.get("scopes", [])], credentials=dict(raw["credentials"]),
                         created_at=str(raw.get("created_at", "")))
    except (ValueError, KeyError, TypeError):
        raise AuthError(f"{path.name} is damaged or not a token file written by this pipeline",
                        f"delete {path} and run the command again to log in.") from None


def save_token(token: TokenFile) -> None:
    """Atomic write, readable by the owner only."""
    token.path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(token.path.parent, 0o700)
    except OSError:
        pass
    payload = {"version": TOKEN_VERSION, "pipeline_channel_id": token.pipeline_channel_id,
               "youtube_channel_id": token.youtube_channel_id, "youtube_channel_title": token.youtube_channel_title,
               "scopes": token.scopes, "created_at": token.created_at, "credentials": token.credentials}
    tmp = token.path.with_name(token.path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, token.path)


def read_client_secret(path: Path) -> dict[str, Any]:
    """Validate ``client_secret.json`` (must be a Desktop-app client)."""
    if not path.is_file():
        raise ClientSecretMissing(
            f"OAuth client file not found: {path}",
            "create an OAuth client of type 'Desktop app' in Google Cloud Console (Google Auth Platform -> Clients "
            "-> Create client), download its JSON and save it as secrets/client_secret.json — README, section "
            "'Загрузка на YouTube', steps 1-4. Another location can be set with YOUTUBE_CLIENT_SECRET_FILE in .env.")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise ClientSecretMissing(f"{path} is not valid JSON", "download the client JSON again.") from None
    if not isinstance(config, dict) or "installed" not in config:
        kind = next(iter(config), "?") if isinstance(config, dict) else "?"
        raise ClientSecretMissing(
            f"{path.name} is an OAuth client of type '{kind}', but a 'Desktop app' client is needed",
            "in Google Cloud Console create a new client with Application type = Desktop app and download its JSON.")
    return config


# --------------------------------------------------------------------------- #
# Google libraries behind a small seam (tests use a fake backend)
# --------------------------------------------------------------------------- #
class AuthBackend(Protocol):
    def login(self, client_secret_file: Path, scopes: Sequence[str], port: int, log: logging.Logger,
              timeout_sec: int = 600) -> Any: ...
    def from_info(self, info: dict[str, Any], scopes: Sequence[str]) -> Any: ...
    def to_info(self, credentials: Any) -> dict[str, Any]: ...
    def is_valid(self, credentials: Any) -> bool: ...
    def refresh(self, credentials: Any) -> None: ...
    def session(self, credentials: Any) -> Any: ...
    def secret_values(self, credentials: Any) -> list[str]: ...
    def translate(self, exc: BaseException) -> BaseException | None: ...


def _error_code(exc: BaseException) -> str:
    """The OAuth error code of a google-auth exception — never its full text (may carry request details)."""
    text = " ".join(str(part) for part in getattr(exc, "args", ()) or (str(exc),))
    match = re.search(r"\b(invalid_grant|invalid_client|unauthorized_client|deleted_client|access_denied|"
                      r"invalid_scope|invalid_request|org_internal|admin_policy_enforced)\b", text)
    return match.group(1) if match else type(exc).__name__


class GoogleAuthBackend:
    """The real thing: google-auth + google-auth-oauthlib."""

    @staticmethod
    def _require() -> None:
        try:
            import google.auth  # noqa: F401
            import google_auth_oauthlib  # noqa: F401
        except ImportError:
            raise AuthError("the Google auth libraries are not installed",
                            "run `pip install -r requirements.txt` inside the virtual environment.") from None

    def login(self, client_secret_file: Path, scopes: Sequence[str], port: int, log: logging.Logger,
              timeout_sec: int = 600) -> Any:
        self._require()
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_file), scopes=list(scopes))
        log.info("opening the browser for Google login — choose the Google account and the YouTube channel to "
                 "upload to. If no browser opens, copy the link printed below into a browser on this computer.")
        started = time.monotonic()
        try:
            credentials = flow.run_local_server(
                host="localhost", port=port, open_browser=True, timeout_seconds=timeout_sec,
                access_type="offline", prompt="consent",
                authorization_prompt_message="Open this link to authorise the upload:\n{url}\n",
                success_message="Done. You can close this tab and return to the terminal.")
        except Exception as exc:  # noqa: BLE001 — oauthlib raises many types; none may leak details
            if time.monotonic() - started >= timeout_sec:
                waited = f"{timeout_sec // 60} minutes" if timeout_sec >= 120 else f"{timeout_sec} seconds"
                raise AuthError(f"the Google login was not finished within {waited}",
                                "run the command again and complete the login in the browser. On a server without "
                                "a browser: log in once on your own computer and copy secrets/token_<channel_id>.json "
                                "to the server (README).") from None
            code = _error_code(exc)
            hint = "run the command again and finish the login in the browser."
            if code == "access_denied":
                hint = ("the login was cancelled, or this Google account is not in the app's test users: Google "
                        "Cloud Console -> Google Auth Platform -> Audience -> Test users -> Add users (README, "
                        "step 3).")
            raise AuthError(f"Google login failed ({code})", hint) from None
        if not getattr(credentials, "refresh_token", None):
            raise AuthError("Google returned no refresh token",
                            "remove this app at https://myaccount.google.com/permissions and run the command again.")
        return credentials

    def from_info(self, info: dict[str, Any], scopes: Sequence[str]) -> Any:
        self._require()
        from google.oauth2.credentials import Credentials

        try:
            return Credentials.from_authorized_user_info(info, scopes=list(scopes))
        except ValueError:
            raise AuthError("the token file has no usable credentials",
                            "delete secrets/token_<channel_id>.json and run the command again to log in.") from None

    def to_info(self, credentials: Any) -> dict[str, Any]:
        return dict(json.loads(credentials.to_json()))

    def is_valid(self, credentials: Any) -> bool:
        return bool(credentials.valid)

    def refresh(self, credentials: Any) -> None:
        from google.auth.exceptions import RefreshError, TransportError
        from google.auth.transport.requests import Request

        try:
            credentials.refresh(Request())
        except RefreshError as exc:
            code = _error_code(exc)
            if code == "invalid_grant":
                raise InvalidGrant() from None
            raise AuthError(f"Google refused to refresh the token ({code})",
                            "if the OAuth client was deleted or its secret was reset, download a new "
                            "client_secret.json; then delete secrets/token_<channel_id>.json and run again.") from None
        except TransportError:
            raise RetryableAPIError("Google OAuth: network error while refreshing the token") from None

    def session(self, credentials: Any) -> Any:
        from google.auth.transport.requests import AuthorizedSession

        return AuthorizedSession(credentials)

    def secret_values(self, credentials: Any) -> list[str]:
        values = [getattr(credentials, name, None) for name in ("token", "refresh_token", "client_secret")]
        return [v for v in values if isinstance(v, str) and v]

    def translate(self, exc: BaseException) -> BaseException | None:
        """Map google-auth exceptions raised inside ``session.request`` (token refresh on the fly)."""
        try:
            from google.auth.exceptions import RefreshError, TransportError
        except ImportError:
            return None
        if isinstance(exc, RefreshError):
            return AuthExpired(f"Google refused to refresh the token ({_error_code(exc)})")
        if isinstance(exc, TransportError):
            return RetryableAPIError("YouTube: network error (token refresh)")
        return None


# --------------------------------------------------------------------------- #
# connect
# --------------------------------------------------------------------------- #
@dataclass
class Connection:
    api: YouTubeAPI
    channel: ChannelInfo
    token_path: Path
    first_login: bool
    secrets: list[str] = field(default_factory=list, repr=False)
    _persist: Callable[[], None] = field(default=lambda: None, repr=False)

    def persist(self) -> None:
        """Save the (possibly refreshed) credentials back to the token file."""
        self._persist()


ApiFactory = Callable[..., YouTubeAPI]


def required_scopes(settings: Settings) -> tuple[str, ...]:
    return tuple(settings.youtube_oauth_scopes) or DEFAULT_SCOPES


def connect(settings: Settings, channel_id: str, log: logging.Logger, *, backend: AuthBackend | None = None,
            api_factory: ApiFactory | None = None) -> Connection:
    """Authorise for the script's ``channel_id`` and return a ready API client bound to the right channel."""
    backend = backend or GoogleAuthBackend()
    scopes = required_scopes(settings)
    path = token_path(settings.secrets_dir, channel_id)
    token = load_token(path)
    expected_channel = settings.pinned_channel(channel_id) or (channel_id if _YOUTUBE_CHANNEL.match(channel_id) else None)

    if token is not None and token.pipeline_channel_id != channel_id:
        raise ChannelMismatch(
            f"{path.name} was issued for channel_id '{token.pipeline_channel_id}', not '{channel_id}'",
            f"the file was copied or renamed. Delete {path} and run the command again to log in for "
            f"'{channel_id}'.")
    if token is not None and not set(scopes) <= set(token.scopes):
        log.warning("the saved authorisation lacks a required permission — a new login is needed")
        expected_channel = expected_channel or token.youtube_channel_id
        token = None

    first_login = token is None
    if token is None:
        read_client_secret(settings.client_secret_path)
        credentials = backend.login(settings.client_secret_path, scopes, settings.youtube_oauth_port, log,
                                    settings.youtube_oauth_timeout_sec)
    else:
        credentials = backend.from_info(token.credentials, scopes)
        expected_channel = token.youtube_channel_id
        if not backend.is_valid(credentials):
            try:
                backend.refresh(credentials)
            except InvalidGrant:
                path.unlink(missing_ok=True)
                raise TokenExpired(f"Google rejected the saved token for '{channel_id}' (invalid_grant: expired or "
                                   f"revoked). {path.name} has been deleted.", TESTING_HINT) from None

    make_api = api_factory or HttpYouTubeAPI
    api = make_api(backend.session(credentials), attempts=settings.http_attempts,
                   chunk_size=settings.youtube_upload_chunk_mb * 1024 * 1024, log=log,
                   translate_exception=backend.translate)
    channel = api.my_channel()

    if expected_channel and channel.id != expected_channel:
        if first_login:
            raise ChannelMismatch(
                f"you logged in to YouTube channel '{channel.title}' ({channel.id}), but '{channel_id}' must be "
                f"channel {expected_channel}",
                "nothing was saved. Run the command again and pick the right channel on Google's screen "
                f"(or fix {'YOUTUBE_CHANNEL_ID__' + channel_pin_key(channel_id)} in .env).")
        raise ChannelMismatch(
            f"{path.name} belongs to YouTube channel '{token.youtube_channel_title}' ({token.youtube_channel_id}), "  # type: ignore[union-attr]
            f"but Google now reports channel '{channel.title}' ({channel.id}) for it — refusing to upload",
            f"if the channel really changed, delete {path} and run the command again to log in.")

    saved = TokenFile(path=path, pipeline_channel_id=channel_id, youtube_channel_id=channel.id,
                      youtube_channel_title=channel.title, scopes=list(scopes),
                      credentials=backend.to_info(credentials),
                      created_at=token.created_at if token else now_iso())

    def persist() -> None:
        saved.credentials = backend.to_info(credentials)
        save_token(saved)

    persist()
    if first_login:
        log.info("=" * 72)
        log.info("YouTube channel for '%s': %s  (ID %s)", channel_id, channel.title, channel.id)
        log.info("saved to %s — later runs refuse to work if this token turns out to belong to another channel",
                 settings.rel(path))
        log.info("=" * 72)
    else:
        log.info("YouTube channel: %s (%s)", channel.title, channel.id)
    return Connection(api=api, channel=channel, token_path=path, first_login=first_login,
                      secrets=backend.secret_values(credentials), _persist=persist)
