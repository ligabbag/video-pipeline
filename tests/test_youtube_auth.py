"""OAuth plumbing: per-channel token files, channel binding, invalid_grant, missing client secret."""
from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import Settings
from pipeline.logging_setup import setup_logging
from pipeline.youtube_auth import (DEFAULT_SCOPES, TESTING_HINT, AuthError, ChannelMismatch, ClientSecretMissing,
                                   InvalidGrant, TokenExpired, connect, load_token, required_scopes, token_path)
from tests.youtube_fakes import (CHANNEL, OTHER_CHANNEL, FakeAuthBackend, FakeYouTube, no_network,  # noqa: F401
                                 write_client_secret)

pytestmark = pytest.mark.usefixtures("no_network")
ALIAS = "yt_tech_disasters_en"


class Env:
    """One project root with a fake Google behind it."""

    def __init__(self, root: Path, env: dict[str, str] | None = None) -> None:
        self.root = root
        self.settings = Settings.load(root, env=env or {})
        self.backend = FakeAuthBackend()
        self.api = FakeYouTube()
        self.log = setup_logging(root / "auth.log", [])
        self.factory_kwargs: dict[str, Any] = {}

    def connect(self, alias: str = ALIAS):
        def factory(session: Any, **kwargs: Any) -> FakeYouTube:
            self.factory_kwargs = kwargs
            return self.api

        return connect(self.settings, alias, self.log, backend=self.backend, api_factory=factory)

    @property
    def token(self) -> Path:
        return self.root / "secrets" / f"token_{ALIAS}.json"

    def log_text(self) -> str:
        return (self.root / "auth.log").read_text(encoding="utf-8")


def test_first_login_shows_the_channel_and_saves_a_private_per_channel_token(tmp_path: Path) -> None:
    env = Env(tmp_path)
    write_client_secret(tmp_path)
    conn = env.connect()

    assert conn.first_login and conn.channel == CHANNEL and env.backend.logins == 1
    assert conn.token_path == env.token == token_path(env.settings.secrets_dir, ALIAS)
    saved = json.loads(env.token.read_text(encoding="utf-8"))
    assert saved["pipeline_channel_id"] == ALIAS
    assert (saved["youtube_channel_id"], saved["youtube_channel_title"]) == (CHANNEL.id, "Tech Disasters")
    assert saved["scopes"] == list(DEFAULT_SCOPES) and saved["credentials"]["refresh_token"] == "REFRESH-SECRET-1"
    assert stat.S_IMODE(env.token.stat().st_mode) == 0o600                      # owner-only
    text = env.log_text()
    assert "Tech Disasters" in text and CHANNEL.id in text                      # name and ID are shown
    for secret in ("ACCESS-SECRET-1", "REFRESH-SECRET-1", "CLIENT-SECRET-XYZ"):
        assert secret not in text                                               # ...tokens and secrets are not
    assert set(conn.secrets) == {"ACCESS-SECRET-1", "REFRESH-SECRET-1", "CLIENT-SECRET-XYZ"}
    assert env.factory_kwargs["chunk_size"] == 8 * 1024 * 1024 and env.factory_kwargs["chunk_size"] % (256 * 1024) == 0

    # second run: no browser, the saved token is used and the channel is checked again
    conn2 = env.connect()
    assert not conn2.first_login and env.backend.logins == 1 and env.api.count("my_channel") == 2

    # another script channel_id gets its own token file and its own login
    env.api.channel = OTHER_CHANNEL
    env.connect("yt_cooking_en")
    assert env.backend.logins == 2
    assert json.loads((tmp_path / "secrets" / "token_yt_cooking_en.json").read_text())["youtube_channel_id"] == OTHER_CHANNEL.id
    assert json.loads(env.token.read_text())["youtube_channel_id"] == CHANNEL.id


def test_only_the_scope_that_is_needed_is_requested(tmp_path: Path) -> None:
    assert DEFAULT_SCOPES == ("https://www.googleapis.com/auth/youtube.force-ssl",)
    custom = Settings.load(tmp_path, env={"YOUTUBE_OAUTH_SCOPES": "https://www.googleapis.com/auth/youtube.upload, "
                                                                  "https://www.googleapis.com/auth/youtube.force-ssl"})
    assert required_scopes(custom) == ("https://www.googleapis.com/auth/youtube.upload",
                                       "https://www.googleapis.com/auth/youtube.force-ssl")


def test_token_of_another_channel_is_refused_and_kept(tmp_path: Path) -> None:
    env = Env(tmp_path)
    write_client_secret(tmp_path)
    env.connect()
    before = env.token.read_text(encoding="utf-8")

    env.api.channel = OTHER_CHANNEL                    # Google now says this token belongs to another channel
    with pytest.raises(ChannelMismatch) as failure:
        env.connect()
    message = str(failure.value)
    assert "Tech Disasters" in message and "Cooking With Bob" in message and OTHER_CHANNEL.id in message
    assert "refusing to upload" in message and failure.value.refused is True
    assert env.token.read_text(encoding="utf-8") == before and env.backend.logins == 1   # untouched, no silent re-login

    # a token file copied/renamed from another pipeline channel is refused before any request
    other = tmp_path / "secrets" / "token_yt_other.json"
    other.write_text(before, encoding="utf-8")
    calls = env.api.count("my_channel")
    with pytest.raises(ChannelMismatch, match="was issued for channel_id 'yt_tech_disasters_en'"):
        env.connect("yt_other")
    assert env.api.count("my_channel") == calls


def test_pinned_or_literal_channel_id_must_match_on_first_login(tmp_path: Path) -> None:
    write_client_secret(tmp_path)
    env = Env(tmp_path, env={"YOUTUBE_CHANNEL_ID__YT_TECH_DISASTERS_EN": OTHER_CHANNEL.id})
    with pytest.raises(ChannelMismatch) as failure:    # logged in to CHANNEL, the pin expects OTHER_CHANNEL
        env.connect()
    assert "nothing was saved" in failure.value.hint and not env.token.exists()

    literal = Env(tmp_path)                            # a script whose channel_id is a real YouTube channel ID
    literal.api.channel = OTHER_CHANNEL
    with pytest.raises(ChannelMismatch):
        literal.connect(CHANNEL.id)
    literal.api.channel = CHANNEL
    assert literal.connect(CHANNEL.id).channel == CHANNEL


def test_invalid_grant_deletes_the_token_and_explains_the_7_day_rule(tmp_path: Path) -> None:
    env = Env(tmp_path)
    write_client_secret(tmp_path)
    env.connect()
    saved = json.loads(env.token.read_text(encoding="utf-8"))
    saved["credentials"]["valid"] = False              # access token expired -> a refresh is needed
    env.token.write_text(json.dumps(saved), encoding="utf-8")

    env.backend.refresh_error = InvalidGrant()
    with pytest.raises(TokenExpired) as failure:
        env.connect()
    assert not env.token.exists()                                               # deleted: the next run logs in again
    assert "invalid_grant" in str(failure.value) and "has been deleted" in str(failure.value)
    assert failure.value.hint == TESTING_HINT and failure.value.refused is False
    for phrase in ("Testing", "7 days", "Publish app", "In production", "Run the same command again"):
        assert phrase in failure.value.hint

    env.backend.refresh_error = None                   # next run: a normal first login again
    assert env.connect().first_login and env.backend.logins == 2


def test_expired_access_token_is_refreshed_and_saved(tmp_path: Path) -> None:
    env = Env(tmp_path)
    write_client_secret(tmp_path)
    env.connect()
    saved = json.loads(env.token.read_text(encoding="utf-8"))
    saved["credentials"]["valid"] = False
    env.token.write_text(json.dumps(saved), encoding="utf-8")
    conn = env.connect()
    assert env.backend.refreshes == 1 and env.backend.logins == 1 and not conn.first_login
    assert json.loads(env.token.read_text())["credentials"]["token"] == "ACCESS-SECRET-REFRESHED"
    (tmp_path / "secrets" / "client_secret.json").unlink()                      # not needed once a token exists
    assert env.connect().channel == CHANNEL


def test_missing_or_wrong_client_secret_is_explained(tmp_path: Path) -> None:
    env = Env(tmp_path)
    with pytest.raises(ClientSecretMissing) as missing:
        env.connect()
    assert "secrets/client_secret.json" in str(missing.value).replace("\\", "/") and missing.value.refused
    assert "Desktop app" in missing.value.hint and "YOUTUBE_CLIENT_SECRET_FILE" in missing.value.hint
    assert env.backend.logins == 0

    write_client_secret(tmp_path, kind="web")
    with pytest.raises(ClientSecretMissing, match="type 'web'"):
        env.connect()

    elsewhere = tmp_path / "keys" / "google.json"
    elsewhere.parent.mkdir()
    elsewhere.write_text(write_client_secret(tmp_path).read_text(), encoding="utf-8")
    (tmp_path / "secrets" / "client_secret.json").unlink()
    moved = Env(tmp_path, env={"YOUTUBE_CLIENT_SECRET_FILE": "keys/google.json", "YOUTUBE_SECRETS_DIR": "vault"})
    conn = moved.connect()
    assert conn.token_path == tmp_path / "vault" / f"token_{ALIAS}.json" and conn.token_path.is_file()


def test_damaged_token_and_unsafe_channel_ids(tmp_path: Path) -> None:
    env = Env(tmp_path)
    write_client_secret(tmp_path)
    env.token.write_text("{not json", encoding="utf-8")
    with pytest.raises(AuthError, match="damaged"):
        env.connect()
    assert load_token(tmp_path / "secrets" / "absent.json") is None
    for bad in ("../evil", "a/b", "", "with space"):
        with pytest.raises(AuthError, match="cannot be used in a file name"):
            token_path(tmp_path / "secrets", bad)


def test_token_with_too_few_scopes_forces_a_new_login_for_the_same_channel(tmp_path: Path) -> None:
    env = Env(tmp_path)
    write_client_secret(tmp_path)
    env.connect()
    saved = json.loads(env.token.read_text(encoding="utf-8"))
    saved["scopes"] = ["https://www.googleapis.com/auth/youtube.upload"]
    env.token.write_text(json.dumps(saved), encoding="utf-8")
    env.api.channel = OTHER_CHANNEL                    # ...and the user picks the wrong channel this time
    with pytest.raises(ChannelMismatch):
        env.connect()
    env.api.channel = CHANNEL
    assert env.connect().first_login and env.backend.logins == 3
    assert json.loads(env.token.read_text())["scopes"] == list(DEFAULT_SCOPES)
