"""The ``upload`` command against a fake YouTube: preconditions, metadata, idempotency, Shorts, warnings."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from pipeline.cli import main
from pipeline.http import RetryableAPIError
from pipeline.upload import (UploadOptions, UploadRefused, metadata_problems, parse_publish_at, run_upload,
                             short_description, short_title, tags_length, truncate_words)
from pipeline.youtube_api import (AuthExpired, ChannelInfo, QuotaExceeded, YouTubeAPIError, YouTubeError,
                                  explain_error)
from pipeline.youtube_auth import ChannelMismatch, ClientSecretMissing, TokenExpired
from tests.youtube_fakes import (CHANNEL, SH1, SH2, Connector, FakeYouTube, log_text, make_rendered_project,  # noqa: F401
                                 no_network, reload, upload_doc)

pytestmark = pytest.mark.usefixtures("no_network")
VIDEO_ID = "tech-0001-knight-capital"
NOW = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)


class Project:
    """A rendered project plus a fake YouTube, driven the way the CLI drives it."""

    def __init__(self, root: Path, example_doc: dict[str, Any], **kwargs: Any) -> None:
        self.root = root
        self.ctx, self.script, self.sha = make_rendered_project(root, upload_doc(example_doc), **kwargs)
        self.api = FakeYouTube()
        self.connector = Connector(self.api, root)
        self.printed: list[str] = []

    def upload(self, **options: Any) -> int:
        reload(self.ctx)
        return run_upload(self.ctx, UploadOptions(**options), self.sha, connector=self.connector,
                          out=self.printed.append)

    @property
    def state(self) -> dict[str, Any]:
        return json.loads((self.root / "state" / f"{VIDEO_ID}.json").read_text(encoding="utf-8"))

    def new_log(self) -> str:
        """Log text written since the previous call."""
        text = log_text(self.ctx)
        fresh, self._seen = text[getattr(self, "_seen", 0):], len(text)
        return fresh


@pytest.fixture()
def project(tmp_path: Path, example_doc: dict[str, Any]) -> Project:
    return Project(tmp_path, example_doc)


# --------------------------------------------------------------------------- #
# what is sent
# --------------------------------------------------------------------------- #
def test_full_upload_sends_video_thumbnail_captions_and_shorts_with_the_right_metadata(project: Project) -> None:
    assert project.upload() == 0
    api = project.api
    assert [name for name, _ in api.calls] == ["my_channel", "upload", "thumbnail", "captions", "upload", "upload",
                                               "list"]
    long_id, short1_id, short2_id = "vid00000001", "vid00000002", "vid00000003"
    meta = json.loads((project.ctx.paths.metadata).read_text(encoding="utf-8"))

    long = api.videos[long_id]
    assert long["file"] == "long.mp4" and long["notify"] is True
    assert long["body"]["snippet"] == {
        "title": "How One Forgotten Server Lost $460 Million in 45 Minutes", "description": meta["description"],
        "tags": meta["tags"], "categoryId": "28", "defaultLanguage": "en-US", "defaultAudioLanguage": "en-US"}
    assert long["body"]["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False,
                                      "containsSyntheticMedia": False}          # private by default, no publishAt
    assert long["thumbnail"] == (project.ctx.paths.final / "thumbnail.jpg").read_bytes()
    assert long["captions"] == [{"language": "en", "name": "English",
                                 "text": (project.ctx.paths.captions / "en.srt").read_text("utf-8")}]

    first, second = api.videos[short1_id], api.videos[short2_id]
    assert first["file"] == f"{SH1}.mp4" and second["file"] == f"{SH2}.mp4"
    assert first["body"]["snippet"]["title"] == "This company lost $460 million because of one server. #Shorts"
    assert first["body"]["snippet"]["description"] == (
        "This company lost $460 million because of one server.\n\nFull story: https://youtu.be/vid00000001")
    # no hook_override_text: the long video's title, and the first scene's line as the hook
    assert second["body"]["snippet"]["title"] == "How One Forgotten Server Lost $460 Million in 45 Minutes #Shorts"
    assert second["body"]["snippet"]["description"].endswith("\n\nFull story: https://youtu.be/vid00000001")
    assert second["body"]["snippet"]["description"].count("\n") == 2            # one hook line + the link
    for short in (first, second):                                               # same tags, category and flags
        assert short["body"]["snippet"]["tags"] == meta["tags"] and short["body"]["snippet"]["categoryId"] == "28"
        assert short["body"]["status"] == long["body"]["status"]
        assert short["thumbnail"] is None and short["captions"] == [] and short["notify"] is False

    upload = project.state["upload"]
    assert upload["youtube_channel_id"] == CHANNEL.id and upload["long"]["video_id"] == long_id
    assert upload["long"]["thumbnail_set"] and upload["long"]["captions_uploaded"] and upload["long"]["verified_at"]
    assert upload["long"]["actual_privacy"] == "private" and upload["long"]["session"] is None
    assert {k: v["video_id"] for k, v in upload["shorts"].items()} == {SH1: short1_id, SH2: short2_id}

    log = project.new_log()
    assert "long video: uploaded -> https://youtu.be/vid00000001" in log and "100%" in log     # progress in the log
    # the API cannot link a Short to its long video: a manual reminder with Studio links
    assert "REMINDER" in log and "Related video" in log
    assert "https://studio.youtube.com/video/vid00000002/edit" in log
    assert "https://studio.youtube.com/video/vid00000003/edit" in log
    assert "WARNING" not in log


def test_no_shorts_flag_and_scripts_without_captions(tmp_path: Path, example_doc: dict[str, Any]) -> None:
    doc = upload_doc(example_doc)
    doc["production"]["subtitles"]["caption_tracks"] = False
    ctx, _, sha = make_rendered_project(tmp_path, doc)
    api = FakeYouTube()
    assert run_upload(ctx, UploadOptions(shorts=False), sha, connector=Connector(api, tmp_path)) == 0
    assert [name for name, _ in api.calls] == ["my_channel", "upload", "thumbnail", "list"]
    assert "REMINDER" not in log_text(ctx)
    # later, without --no-shorts: only the Shorts are added
    assert run_upload(reload(ctx), UploadOptions(), sha, connector=Connector(api, tmp_path)) == 0
    assert [name for name, _ in api.calls][4:] == ["my_channel", "upload", "upload", "list"]
    assert api.calls[-1] == ("list", "vid00000002,vid00000003")


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def test_second_run_uploads_nothing_and_third_run_after_a_failed_thumbnail_uploads_only_it(project: Project) -> None:
    assert project.upload() == 0
    calls_after_first = list(project.api.calls)
    project.new_log()

    # second run: nothing is sent — not even a login
    assert project.upload() == 0
    assert project.api.calls == calls_after_first and project.connector.calls == 1
    log = project.new_log()
    assert "nothing to upload" in log and "https://youtu.be/vid00000001" in log and "REMINDER" in log

    # the thumbnail "failed": the state says it is not on YouTube
    state_path = project.root / "state" / f"{VIDEO_ID}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["upload"]["long"]["thumbnail_set"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")

    # third run: only the thumbnail goes out
    assert project.upload() == 0
    assert project.api.calls[len(calls_after_first):] == [("my_channel", ""), ("thumbnail", "vid00000001")]
    assert len(project.api.videos) == 3 and project.state["upload"]["long"]["thumbnail_set"] is True
    assert project.upload() == 0 and project.connector.calls == 2              # and again nothing


def test_thumbnail_failure_does_not_stop_the_rest_and_the_next_run_sends_only_the_thumbnail(project: Project) -> None:
    project.api.fail_next("thumbnail", explain_error("thumbnails.set", 403, "forbidden", "no permission"))
    assert project.upload() == 1                                                # reported as a failed run
    assert [name for name, _ in project.api.calls] == ["my_channel", "upload", "thumbnail", "captions", "upload",
                                                       "upload", "list"]         # everything else still went out
    upload = project.state["upload"]
    assert upload["long"]["video_id"] and not upload["long"]["thumbnail_set"] and upload["long"]["captions_uploaded"]
    log = project.new_log()
    assert "thumbnail failed" in log and "youtube.com/verify" in log            # says what to do
    assert "1 step(s) failed" in log and "only the missing parts" in log

    done = len(project.api.calls)
    assert project.upload() == 0
    assert project.api.calls[done:] == [("my_channel", ""), ("thumbnail", "vid00000001")]
    assert project.api.videos["vid00000001"]["thumbnail"] is not None and len(project.api.videos) == 3


def test_interrupted_long_upload_keeps_its_session_and_the_next_run_resumes_it(project: Project) -> None:
    project.api.fail_next("upload", YouTubeAPIError("upload of long.mp4 keeps failing", "run the same command again"))
    with pytest.raises(YouTubeAPIError):
        project.upload()
    saved = project.state["upload"]["long"]
    assert saved["video_id"] is None and saved["session"]["url"].startswith("https://upload.test/session/1")
    assert saved["session"]["size"] == 700_000 and saved["session"]["source_hash"] == "longhash-v1"
    assert project.api.count("thumbnail") == 0                                   # nothing after the failed upload
    assert "SESSION-SECRET" not in log_text(project.ctx)                         # the session URL is never logged

    assert project.upload() == 0
    assert project.api.resume_urls[:2] == [None, saved["session"]["url"]]        # same session, continued
    assert project.api.sessions == 3                                             # 1 for long + 2 Shorts, no restart
    assert project.state["upload"]["long"]["session"] is None
    assert "unfinished upload was found" in project.new_log()


def test_session_is_not_resumed_when_the_file_or_metadata_changed(project: Project) -> None:
    project.api.fail_next("upload", YouTubeAPIError("boom"))
    with pytest.raises(YouTubeAPIError):
        project.upload()
    assert project.upload(privacy="unlisted") == 0                               # other metadata -> other body
    assert project.api.resume_urls[1] is None
    assert project.api.videos["vid00000001"]["body"]["status"]["privacyStatus"] == "unlisted"


def test_force_upload_makes_new_videos_and_warns_about_duplicates(project: Project) -> None:
    assert project.upload() == 0
    project.new_log()
    assert project.upload(force=True) == 0
    assert len(project.api.videos) == 6                                          # old ones are not deleted
    upload = project.state["upload"]
    assert upload["long"]["video_id"] == "vid00000004" and upload["long"]["previous_video_ids"] == ["vid00000001"]
    assert upload["shorts"][SH1]["previous_video_ids"] == ["vid00000002"]
    assert project.api.videos["vid00000005"]["body"]["snippet"]["description"].endswith("https://youtu.be/vid00000004")
    assert project.api.videos["vid00000004"]["thumbnail"] is not None and project.api.videos["vid00000004"]["captions"]
    log = project.new_log()
    assert log.count("--force-upload:") == 3 and "NOT deleted" in log and "https://youtu.be/vid00000001" in log
    assert project.upload() == 0 and len(project.api.videos) == 6                # and it is idempotent again


def test_changed_files_and_options_after_upload_only_warn(project: Project) -> None:
    assert project.upload() == 0
    state_path = project.root / "state" / f"{VIDEO_ID}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifacts"]["long"]["hash"] = "longhash-v2"                           # the video was re-rendered
    state_path.write_text(json.dumps(state), encoding="utf-8")
    project.new_log()
    assert project.upload(privacy="public") == 0 and len(project.api.videos) == 3
    log = project.new_log()
    assert "re-rendered after it was uploaded" in log and "--force-upload" in log
    assert "does not change uploaded videos" in log and "nothing to upload" in log


# --------------------------------------------------------------------------- #
# publishAt
# --------------------------------------------------------------------------- #
def test_publish_at_needs_a_time_zone_and_a_future_time() -> None:
    assert parse_publish_at("2026-12-01T18:00:00+03:00", NOW) == "2026-12-01T15:00:00Z"      # -> UTC, RFC 3339
    assert parse_publish_at("2026-12-01T15:00:00Z", NOW) == "2026-12-01T15:00:00Z"
    assert parse_publish_at("2026-11-30T23:30:00.250-08:00", NOW) == "2026-12-01T07:30:00Z"
    with pytest.raises(UploadRefused, match="no time zone"):
        parse_publish_at("2026-12-01T18:00:00", NOW)
    with pytest.raises(UploadRefused, match="no time zone"):
        parse_publish_at("2026-12-01", NOW)
    with pytest.raises(UploadRefused, match="in the past"):
        parse_publish_at("2026-11-16T14:59:00+03:00", NOW)                                   # 11:59 UTC < now
    with pytest.raises(UploadRefused, match="in the past"):
        parse_publish_at("2020-01-01T00:00:00Z", NOW)
    with pytest.raises(UploadRefused, match="not an ISO 8601"):
        parse_publish_at("next friday", NOW)


def test_publish_at_forces_private_for_the_long_video_and_the_shorts(project: Project) -> None:
    when = parse_publish_at((datetime.now(timezone.utc) + timedelta(days=3)).isoformat())
    assert project.upload(privacy="public", publish_at=when) == 0
    for video in project.api.videos.values():
        assert video["body"]["status"]["privacyStatus"] == "private"                         # not "public"
        assert video["body"]["status"]["publishAt"] == when and when.endswith("Z")
    saved = project.state["upload"]["long"]
    assert saved["requested_privacy"] == "private" and saved["publish_at"] == when
    log = project.new_log()
    assert "--privacy public is ignored" in log and "!!!!" not in log                        # private == requested


def test_publish_at_errors_exit_with_code_2_before_anything_else(project: Project, capsys: pytest.CaptureFixture[str],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pipeline.upload.connect", project.connector)
    args = ["upload", str(project.script), "--root", str(project.root)]
    assert main([*args, "--publish-at", "2026-12-01T18:00:00"]) == 2
    assert "no time zone" in capsys.readouterr().err
    assert main([*args, "--publish-at", "2001-01-01T00:00:00+00:00"]) == 2
    assert "in the past" in capsys.readouterr().err
    assert project.connector.calls == 0 and not project.api.calls


# --------------------------------------------------------------------------- #
# Shorts titles
# --------------------------------------------------------------------------- #
def test_short_title_rules() -> None:
    long_title = "How One Forgotten Server Lost $460 Million in 45 Minutes"
    assert short_title("This company lost $460 million.", long_title) == "This company lost $460 million. #Shorts"
    assert short_title(None, long_title) == long_title + " #Shorts"              # no hook -> the long title
    assert short_title("   ", long_title) == long_title + " #Shorts"

    exactly_92 = "x" * 92                                                        # 92 + 1 + 7 = 100: still fits
    assert short_title(exactly_92, long_title) == exactly_92 + " #Shorts"
    assert len(short_title(exactly_92, long_title)) == 100
    exactly_93 = "y" * 93                                                        # 101 with the tag: the tag is dropped
    assert short_title(exactly_93, long_title) == exactly_93

    hook = ("One engineer forgot to update a single server out of eight and the whole company was gone "
            "before lunch that very same day")
    title = short_title(hook, long_title)
    assert len(hook) > 100 and len(title) <= 100 and "#Shorts" not in title      # no room for the tag
    assert title.endswith("…") and hook.startswith(title[:-1])                   # cut...
    assert hook[len(title) - 1] == " "                                           # ...exactly at a word boundary
    assert title == "One engineer forgot to update a single server out of eight and the whole company was gone before…"

    assert short_title("Already tagged #shorts", long_title) == "Already tagged #shorts"
    assert short_title("a <b> c", long_title) == "a b c #Shorts"                 # YouTube rejects < and >
    assert truncate_words("word " * 30, 20) == "word word word word…"
    assert truncate_words("Supercalifragilisticexpialidocious", 10) == "Supercali…"
    assert short_description("Hook line.\nsecond", "abc123") == "Hook line. second\n\nFull story: https://youtu.be/abc123"


def test_metadata_limits_are_checked_before_uploading(tmp_path: Path, example_doc: dict[str, Any]) -> None:
    assert tags_length(["foo-baz", "foo baz"]) == 7 + 9 + 1                      # YouTube's way of counting
    assert metadata_problems("ok", "fine", ["a"]) == []
    problems = metadata_problems("t" * 101, "x" * 5001, ["tag number %03d" % i for i in range(40)])
    assert len(problems) == 3 and "101 characters" in problems[0] and "5001 bytes" in problems[1]
    assert metadata_problems("a < b", "", [])

    ctx, _, sha = make_rendered_project(tmp_path, upload_doc(example_doc))
    meta = json.loads(ctx.paths.metadata.read_text(encoding="utf-8"))
    meta["title"] = "T" * 120
    ctx.paths.metadata.write_text(json.dumps(meta), encoding="utf-8")
    connector = Connector(FakeYouTube(), tmp_path)
    with pytest.raises(UploadRefused, match="120 characters"):
        run_upload(ctx, UploadOptions(), sha, connector=connector)
    assert connector.calls == 0


# --------------------------------------------------------------------------- #
# verification after the upload
# --------------------------------------------------------------------------- #
def test_privacy_mismatch_after_upload_gives_a_loud_warning(project: Project) -> None:
    project.api.forced_privacy = "private"                    # what an unaudited API project used to get
    assert project.upload(privacy="public") == 0              # the upload itself worked
    assert project.api.videos["vid00000001"]["body"]["status"]["privacyStatus"] == "public"
    log = project.new_log()
    assert "!" * 78 in log
    assert "WARNING: long video https://youtu.be/vid00000001 is 'private' on YouTube, but 'public' was requested" in log
    assert "unverified API project" in log and "audit" in log and "publish by hand in YouTube Studio" in log
    assert log.count("but 'public' was requested") == 3       # the long video and both Shorts
    saved = project.state["upload"]["long"]
    assert saved["requested_privacy"] == "public" and saved["actual_privacy"] == "private"

    assert project.upload(privacy="public") == 0              # later runs repeat the warning without any request
    assert project.connector.calls == 1 and "although 'public' was requested" in project.new_log()

    assert main(["status", VIDEO_ID, "--root", str(project.root)]) == 0


def test_matching_privacy_gives_no_warning_and_language_loss_is_reported(project: Project) -> None:
    project.api.drop_audio_language = True
    assert project.upload(privacy="unlisted") == 0
    log = project.new_log()
    assert "!!!!" not in log and "was requested" not in log
    assert "did not keep the audio language 'en-US'" in log and "auto-dubbing" in log
    assert project.state["upload"]["long"]["actual_privacy"] == "unlisted"


# --------------------------------------------------------------------------- #
# preconditions (exit code 2) and dry run
# --------------------------------------------------------------------------- #
def cli_upload(project: Project, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
    monkeypatch.setattr("pipeline.upload.connect", project.connector)
    return main(["upload", str(project.script), "--root", str(project.root), *extra])


def rewrite_state(project: Project, change: Any) -> None:
    path = project.root / "state" / f"{VIDEO_ID}.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    change(state)
    path.write_text(json.dumps(state), encoding="utf-8")


def test_refuses_unapproved_or_unverified_scripts(tmp_path: Path, example_doc: dict[str, Any],
                                                  capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path, example_doc)
    doc = json.loads(project.script.read_text(encoding="utf-8"))
    doc["status"] = "needs_review"
    project.script.write_text(json.dumps(doc), encoding="utf-8")
    assert cli_upload(project, monkeypatch) == 2
    err = capsys.readouterr().err
    assert "status is 'needs_review'" in err and "only 'approved'" in err
    assert "the script changed after the last pipeline run" in err               # every problem at once

    doc["status"], doc["qa"]["facts_verified"] = "approved", False               # the validator catches this pair
    project.script.write_text(json.dumps(doc), encoding="utf-8")
    assert cli_upload(project, monkeypatch) == 2
    assert "facts_verified" in capsys.readouterr().err
    assert project.connector.calls == 0 and not project.api.calls

    # the same checks without the validator in front of them
    project.ctx.script = project.ctx.script.model_copy(update={"status": "draft"})
    with pytest.raises(UploadRefused, match="script status is 'draft'"):
        run_upload(project.ctx, UploadOptions(), project.sha, connector=project.connector)
    unverified = project.ctx.script.qa.model_copy(update={"facts_verified": False})
    project.ctx.script = project.ctx.script.model_copy(update={"status": "approved", "qa": unverified})
    with pytest.raises(UploadRefused, match="qa.facts_verified is false"):
        run_upload(project.ctx, UploadOptions(), project.sha, connector=project.connector)


def test_refuses_when_stages_or_files_are_missing(project: Project, capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    def break_things(state: dict[str, Any]) -> None:
        state["stages"]["render"]["status"] = "failed"
        state["thumbnail"]["stage"]["status"] = "pending"

    rewrite_state(project, break_things)
    (project.ctx.paths.final / "thumbnail.jpg").unlink()
    (project.ctx.paths.shorts / f"{SH2}.mp4").unlink()
    assert cli_upload(project, monkeypatch) == 2
    err = capsys.readouterr().err
    assert "stage 'render' is 'failed'" in err and "stage 'thumbnail' is 'pending'" in err
    assert "the thumbnail is missing" in err and f"short {SH2} is missing" in err
    assert "python -m pipeline run" in err                                       # says how to fix it
    assert cli_upload(project, monkeypatch, "--dry-run") == 2                    # a dry run checks the same things
    assert project.connector.calls == 0 and not project.api.calls


def test_refuses_mock_renders_and_projects_that_were_never_rendered(tmp_path: Path, example_doc: dict[str, Any],
                                                                    capsys: pytest.CaptureFixture[str],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path, example_doc, mock=True)
    assert cli_upload(project, monkeypatch) == 2
    assert "--mock" in capsys.readouterr().err
    assert cli_upload(project, monkeypatch, "--dry-run") == 0                    # ...but the plan can be inspected
    assert "a real upload of it will be refused" in capsys.readouterr().out

    (tmp_path / "state" / f"{VIDEO_ID}.json").unlink()
    assert cli_upload(project, monkeypatch) == 2
    assert "nothing has been rendered" in capsys.readouterr().err
    assert main(["upload", str(tmp_path / "absent.json"), "--root", str(tmp_path)]) == 2
    with pytest.raises(SystemExit) as bad_flag:                                  # argparse: unknown privacy value
        main(["upload", str(project.script), "--privacy", "secret"])
    assert bad_flag.value.code == 2
    assert project.connector.calls == 0


def test_dry_run_prints_the_plan_without_login_or_network(project: Project, capsys: pytest.CaptureFixture[str],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    project.connector.error = AssertionError("a dry run must not authorise")
    monkeypatch.setitem(__import__("sys").modules, "google", None)               # and must not need Google libraries
    state_before = (project.root / "state" / f"{VIDEO_ID}.json").read_bytes()
    assert cli_upload(project, monkeypatch, "--dry-run", "--privacy", "unlisted") == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and "no login is performed" in out
    for expected in ("[1] long video — WILL UPLOAD", f"[2] short {SH1} — WILL UPLOAD", f"[3] short {SH2}",
                     "How One Forgotten Server Lost $460 Million in 45 Minutes", "0:00 45 Minutes",
                     "tags        : knight capital, software disaster", "categoryId  : 28",
                     "en-US (defaultLanguage and defaultAudioLanguage)", "selfDeclaredMadeForKids=False",
                     "containsSyntheticMedia=False", "privacy     : unlisted", "final/thumbnail.jpg",
                     "captions/en.srt -> language 'en', name 'English', not a draft",
                     "This company lost $460 million because of one server. #Shorts",
                     "Full story: https://youtu.be/<ID of the long video>", "token_yt_tech_disasters_en.json",
                     "client secret: MISSING", "3 upload(s)", "Related video"):
        assert expected in out, expected
    assert project.connector.calls == 0 and not project.api.calls
    assert (project.root / "state" / f"{VIDEO_ID}.json").read_bytes() == state_before        # nothing recorded
    assert not (project.root / "secrets").exists()

    # after a real upload the dry run shows what is already there
    project.connector.error = None
    assert cli_upload(project, monkeypatch) == 0
    capsys.readouterr()
    assert cli_upload(project, monkeypatch, "--dry-run", "--no-shorts") == 0
    out = capsys.readouterr().out
    assert "already on YouTube: https://youtu.be/vid00000001" in out and "Nothing to do" in out and "[2]" not in out
    assert cli_upload(project, monkeypatch, "--dry-run", "--force-upload") == 0
    assert "as a new video; the old one stays: https://youtu.be/vid00000001" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# errors with advice, exit codes, secrets
# --------------------------------------------------------------------------- #
def test_quota_errors_stop_the_run_and_say_which_quota(project: Project, capsys: pytest.CaptureFixture[str],
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    project.api.fail_next("upload", explain_error("videos.insert", 403, "quotaExceeded", ""))
    assert cli_upload(project, monkeypatch) == 1
    log = project.new_log()
    assert "FAILED: the daily UPLOAD quota" in log and "what to do: videos.insert has its own quota bucket" in log
    assert project.api.count("thumbnail") == 0 and project.state["upload"]["long"]["video_id"] is None

    project.api.fail_next("captions", explain_error("captions.insert", 403, "quotaExceeded", ""))
    assert cli_upload(project, monkeypatch) == 1
    log = project.new_log()
    assert "the daily YouTube Data API quota" in log and "10,000 units a day" in log and "at captions.insert" in log
    assert [name for name, _ in project.api.calls].count("upload") == 2          # stopped before the Shorts
    saved = project.state["upload"]["long"]
    assert saved["video_id"] and saved["thumbnail_set"] and not saved["captions_uploaded"]

    assert cli_upload(project, monkeypatch) == 0                                 # "tomorrow": only the rest goes out
    assert [name for name, _ in project.api.calls][-5:] == ["my_channel", "captions", "upload", "upload", "list"]
    assert len(project.api.videos) == 3


def test_unverified_channel_is_stopped_before_a_long_upload(tmp_path: Path, example_doc: dict[str, Any],
                                                            capsys: pytest.CaptureFixture[str],
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path, example_doc, duration_sec=16 * 60 + 5)
    project.api.channel = ChannelInfo(CHANNEL.id, CHANNEL.title, long_uploads_status="eligible")
    assert cli_upload(project, monkeypatch, "--dry-run") == 0
    assert "longer than 15 minutes" in capsys.readouterr().out                   # warned before any login
    assert cli_upload(project, monkeypatch) == 1
    log = project.new_log()
    assert "16:05" in log and "not allowed to upload videos longer than 15 minutes" in log
    assert "nothing was uploaded" in log and "youtube.com/verify" in log
    assert project.api.count("upload") == 0

    project.api.channel = CHANNEL                                                # verified by phone -> goes through
    assert cli_upload(project, monkeypatch) == 0 and project.api.count("upload") == 3


def test_unverified_channel_with_a_short_video_is_warned_about_the_thumbnail(project: Project) -> None:
    project.api.channel = ChannelInfo(CHANNEL.id, CHANNEL.title, long_uploads_status="eligible")
    assert project.upload() == 0                                                 # 78 s: allowed, so it is uploaded
    log = project.new_log()
    assert "is not phone-verified: YouTube will probably refuse the custom thumbnail" in log
    assert "youtube.com/verify" in log


def test_channel_in_bad_standing_gets_its_own_advice(tmp_path: Path, example_doc: dict[str, Any],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """longUploadsStatus 'disallowed' is not about the phone: do not send the owner to youtube.com/verify."""
    project = Project(tmp_path, example_doc, duration_sec=20 * 60)
    project.api.channel = ChannelInfo(CHANNEL.id, CHANNEL.title, long_uploads_status="disallowed")
    assert cli_upload(project, monkeypatch) == 1
    log = project.new_log()
    assert "not allowed to upload videos longer than 15 minutes" in log and "not in good standing" in log
    assert "what to do: YouTube reports longUploadsStatus 'disallowed'" in log
    assert project.api.count("upload") == 0
    assert ChannelInfo("UC1", "x", "longUploadsUnspecified").long_uploads_allowed is None    # unknown: no guessing


def test_auth_problems_have_exit_codes_and_advice(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    project.connector.error = ChannelMismatch("token_x.json belongs to YouTube channel 'A', but Google now reports "
                                              "channel 'B' — refusing to upload", "delete the token and log in again")
    assert cli_upload(project, monkeypatch) == 2                                 # wrong channel: refused
    log = project.new_log()
    assert "refusing to upload" in log and "what to do: delete the token" in log

    project.connector.error = ClientSecretMissing("OAuth client file not found: secrets/client_secret.json", "README")
    assert cli_upload(project, monkeypatch) == 2
    project.connector.error = TokenExpired("Google rejected the saved token", "7 days")
    assert cli_upload(project, monkeypatch) == 1
    project.connector.error = RetryableAPIError("Google OAuth: network error")
    assert cli_upload(project, monkeypatch) == 1
    log = project.new_log()
    assert "FAILED: Google OAuth: network error" in log and "check the internet connection" in log
    assert not project.api.calls


def test_token_revoked_in_the_middle_deletes_the_token_and_keeps_what_was_uploaded(project: Project,
                                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    token = project.root / "secrets" / "token_yt_tech_disasters_en.json"
    token.parent.mkdir()
    token.write_text("{}", encoding="utf-8")
    project.api.fail_next("captions", AuthExpired("YouTube no longer accepts the saved authorisation (HTTP 401)"))
    assert cli_upload(project, monkeypatch) == 1
    assert not token.exists()
    log = project.new_log()
    assert "has been deleted" in log and "7 days" in log and "Publish app" in log
    saved = project.state["upload"]["long"]
    assert saved["video_id"] == "vid00000001" and saved["thumbnail_set"] and not saved["captions_uploaded"]
    assert cli_upload(project, monkeypatch) == 0 and len(project.api.videos) == 3


def test_tokens_never_reach_the_log_or_the_state(project: Project) -> None:
    assert project.upload() == 0
    project.ctx.log.info("debug dump: token=%s refresh=%s", "ACCESS-SECRET-1", "REFRESH-SECRET-1")
    log = log_text(project.ctx)
    assert "debug dump: token=*** refresh=***" in log
    state_text = (project.root / "state" / f"{VIDEO_ID}.json").read_text(encoding="utf-8")
    for secret in ("ACCESS-SECRET-1", "REFRESH-SECRET-1", "SESSION-SECRET"):
        assert secret not in log and secret not in state_text                    # finished sessions leave no URL


def test_status_shows_links_to_everything_uploaded(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", VIDEO_ID, "--root", str(project.root)]) == 0
    before = capsys.readouterr().out
    assert "thumbnail done" in before and "youtube" not in before
    assert project.upload() == 0
    assert project.upload(force=True, shorts=False) == 0
    assert main(["status", VIDEO_ID, "--root", str(project.root)]) == 0
    out = capsys.readouterr().out
    assert f"youtube  : — channel Tech Disasters ({CHANNEL.id})" in out
    assert "long: https://youtu.be/vid00000004  (private, thumbnail set, captions uploaded)" in out
    assert "earlier upload still on YouTube: https://youtu.be/vid00000001" in out
    # --force-upload --no-shorts re-uploaded only the long video: the Shorts keep their record and link
    assert f"short {SH1}: https://youtube.com/shorts/vid00000002  (private)" in out
    assert len(project.api.videos) == 4 and project.state["upload"]["shorts"][SH1]["previous_video_ids"] == []
    assert "render  done" in out                                                 # the task-1 lines are intact


def test_secrets_folder_is_ignored_by_git_and_docker() -> None:
    root = Path(__file__).resolve().parent.parent
    assert "secrets/" in (root / ".gitignore").read_text(encoding="utf-8").split()
    assert "secrets" in (root / ".dockerignore").read_text(encoding="utf-8").split()
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "google-auth==" in requirements and "google-auth-oauthlib==" in requirements
    assert isinstance(YouTubeError("x"), RuntimeError) and QuotaExceeded.__mro__[1] is YouTubeAPIError


# --------------------------------------------------------------------------- #
# end to end: real mock render -> dry run -> upload through the CLI
# --------------------------------------------------------------------------- #
def test_cli_upload_of_a_really_rendered_project(tmp_path: Path, example_doc: dict[str, Any],
                                                 capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """`run --mock` (FFmpeg and all) -> `upload --dry-run` -> `upload` with the fake YouTube behind `connect`."""
    monkeypatch.setenv("PIPELINE_SUPERSAMPLE", "1")
    script = tmp_path / "script.json"
    script.write_text(json.dumps(upload_doc(example_doc, shorts=1)), encoding="utf-8")
    root = ["--root", str(tmp_path)]
    assert main(["run", str(script), "--mock", *root]) == 0
    out = tmp_path / "output" / VIDEO_ID
    state_path = tmp_path / "state" / f"{VIDEO_ID}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["thumbnail"]["stage"]["status"] == "done" and (out / "final" / "thumbnail.jpg").is_file()
    log = (out / "pipeline.log").read_text(encoding="utf-8")
    assert log.index("--- stage images ---") < log.index("--- stage thumbnail ---") < log.index("--- stage render ---")

    api = FakeYouTube()
    connector = Connector(api, tmp_path)
    monkeypatch.setattr("pipeline.upload.connect", connector)
    capsys.readouterr()
    assert main(["upload", str(script), "--dry-run", *root]) == 0                # a mock render can be inspected...
    plan = capsys.readouterr().out
    assert "final/long.mp4" in plan and "WARNING      : the last pipeline run was --mock" in plan
    assert main(["upload", str(script), *root]) == 2                             # ...but not uploaded
    assert connector.calls == 0

    state["mock"] = False                                                        # pretend it was a real render
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert main(["upload", str(script), *root]) == 0
    assert [name for name, _ in api.calls] == ["my_channel", "upload", "thumbnail", "captions", "upload", "list"]
    long = api.videos["vid00000001"]
    assert long["sha"] == __import__("hashlib").sha256((out / "final" / "long.mp4").read_bytes()).hexdigest()
    assert long["thumbnail"] == (out / "final" / "thumbnail.jpg").read_bytes()
    assert long["captions"][0]["text"] == (out / "captions" / "en.srt").read_text(encoding="utf-8")
    assert "0:00" in long["body"]["snippet"]["description"] and "{{" not in long["body"]["snippet"]["description"]
    assert api.videos["vid00000002"]["file"] == f"{SH1}.mp4"

    assert main(["upload", str(script), *root]) == 0 and connector.calls == 1    # idempotent through the CLI too
    assert main(["status", VIDEO_ID, *root]) == 0
    status = capsys.readouterr().out
    assert "https://youtu.be/vid00000001" in status and "https://youtube.com/shorts/vid00000002" in status

    # re-running the pipeline keeps the upload record (nothing is uploaded twice afterwards)
    assert main(["run", str(script), "--mock", *root]) == 0
    assert json.loads(state_path.read_text(encoding="utf-8"))["upload"]["long"]["video_id"] == "vid00000001"
