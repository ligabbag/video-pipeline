"""``python -m pipeline upload`` — put the finished video, its thumbnail, captions and Shorts on YouTube.

Order of work: long video (resumable) -> thumbnail -> captions -> each Short -> ``videos.list`` check.
Every finished step is written to the state file at once, so a second run uploads only what is
missing; ``--force-upload`` uploads everything again as new videos (old ones are kept).

``--dry-run`` prints the plan without logging in and without a single network request.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from pipeline.context import RunContext
from pipeline.logging_setup import setup_logging
from pipeline.state import UploadedVideo, UploadSession, file_hash, now_iso, stable_hash
from pipeline.thumbnail import MAX_BYTES as THUMBNAIL_MAX_BYTES
from pipeline.youtube_api import (LONG_UPLOAD_LIMIT_SEC, VERIFY_HINT, AuthExpired, QuotaExceeded,
                                  UploadLimitExceeded, YouTubeAPI, YouTubeAPIError, YouTubeError)
from pipeline.youtube_auth import TESTING_HINT, Connection, TokenExpired, connect, token_path

PRIVACY_CHOICES = ("private", "unlisted", "public")
TITLE_LIMIT = 100                 # characters (videos resource: snippet.title)
DESCRIPTION_LIMIT_BYTES = 5000    # bytes (snippet.description)
TAGS_LIMIT = 500                  # characters, counted the way YouTube counts them (snippet.tags[])
SHORTS_TAG = "#Shorts"
SHORT_MAX_SEC = 180.0             # longer vertical videos are not treated as Shorts
CAPTION_NAMES = {"en": "English"}
REQUIRED_STAGES = ("tts", "images", "thumbnail", "render")

Connector = Callable[..., Connection]


class UploadRefused(Exception):
    """A precondition is not met — nothing was sent. The command exits with code 2."""


@dataclass(frozen=True)
class UploadOptions:
    privacy: str = "private"
    publish_at: str | None = None      # UTC, RFC 3339 — see parse_publish_at()
    shorts: bool = True
    dry_run: bool = False
    force: bool = False

    @property
    def effective_privacy(self) -> str:
        """A scheduled video must be private, otherwise YouTube ignores ``publishAt``."""
        return "private" if self.publish_at else self.privacy


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def parse_publish_at(value: str, now: datetime | None = None) -> str:
    """ISO 8601 with a time zone, in the future -> UTC RFC 3339 (``2026-12-01T15:00:00Z``)."""
    text = value.strip()
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise UploadRefused(f"--publish-at '{value}' is not an ISO 8601 date-time "
                            f"(example: 2026-12-01T18:00:00+03:00)") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise UploadRefused(f"--publish-at '{value}' has no time zone — write it explicitly, for example "
                            f"2026-12-01T18:00:00+03:00 or 2026-12-01T15:00:00Z")
    current = now or datetime.now(timezone.utc)
    if moment <= current:
        raise UploadRefused(f"--publish-at '{value}' is in the past (now: "
                            f"{current.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}) — YouTube would "
                            f"publish the video immediately")
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def truncate_words(text: str, limit: int) -> str:
    """Cut at a word boundary so the result is at most ``limit`` characters (ends with an ellipsis if cut)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    kept = text[:limit - 1]
    if text[limit - 1] != " " and " " in kept:        # the cut landed inside a word: drop its beginning too
        kept = kept[:kept.rindex(" ")]
    return kept.rstrip(" ,;:-—–") + "…"


def short_title(hook: str | None, long_title: str, limit: int = TITLE_LIMIT) -> str:
    """Title of a Short: the hook line (else the long video's title), ``#Shorts`` appended only if it fits."""
    base = (hook or "").strip() or long_title
    base = truncate_words(base.replace("<", "").replace(">", ""), limit)
    if SHORTS_TAG.lower() in base.lower():
        return base
    return f"{base} {SHORTS_TAG}" if len(base) + 1 + len(SHORTS_TAG) <= limit else base


def short_description(hook_line: str, long_video_id: str) -> str:
    line = truncate_words(hook_line.replace("<", "").replace(">", ""), 300)
    return f"{line}\n\nFull story: https://youtu.be/{long_video_id}"


def tags_length(tags: Sequence[str]) -> int:
    """YouTube's count: commas between tags count, and a tag with a space counts with two quotation marks."""
    return sum(len(tag) + (2 if " " in tag else 0) for tag in tags) + max(len(tags) - 1, 0)


def metadata_problems(title: str, description: str, tags: Sequence[str]) -> list[str]:
    problems = []
    if not title.strip():
        problems.append("the title is empty")
    if len(title) > TITLE_LIMIT:
        problems.append(f"the title is {len(title)} characters long (YouTube allows {TITLE_LIMIT})")
    if "<" in title + description or ">" in title + description:
        problems.append("the title or description contains '<' or '>' (YouTube rejects them)")
    if len(description.encode("utf-8")) > DESCRIPTION_LIMIT_BYTES:
        problems.append(f"the description is {len(description.encode('utf-8'))} bytes long "
                        f"(YouTube allows {DESCRIPTION_LIMIT_BYTES})")
    if tags_length(tags) > TAGS_LIMIT:
        problems.append(f"the tags add up to {tags_length(tags)} characters (YouTube allows {TAGS_LIMIT})")
    return problems


def video_body(*, title: str, description: str, tags: Sequence[str], category_id: Any, made_for_kids: bool,
               contains_synthetic_media: bool, language: str, privacy: str, publish_at: str | None) -> dict[str, Any]:
    """The ``videos.insert`` request body (parts ``snippet`` and ``status``)."""
    status: dict[str, Any] = {"privacyStatus": privacy, "selfDeclaredMadeForKids": bool(made_for_kids),
                              "containsSyntheticMedia": bool(contains_synthetic_media)}
    if publish_at:
        status["publishAt"] = publish_at
    return {"snippet": {"title": title, "description": description, "tags": list(tags),
                        "categoryId": str(category_id), "defaultLanguage": language,
                        "defaultAudioLanguage": language},      # the audio language drives auto-dubbing
            "status": status}


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    total = int(round(seconds))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}" if total >= 3600 else f"{total // 60}:{total % 60:02d}"


def fmt_size(size: int) -> str:
    return f"{size / 1_000_000:.1f} MB" if size >= 1_000_000 else f"{size / 1000:.0f} KB"


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    key: str                       # "long" or the short_id
    kind: str                      # "long" | "short"
    file: Path
    size: int
    duration_sec: float | None
    source_hash: str
    title: str
    hook_line: str = ""            # Shorts: first line of the description
    notify_subscribers: bool = True

    @property
    def label(self) -> str:
        return "long video" if self.kind == "long" else f"short {self.key}"


@dataclass
class Plan:
    long: Target
    shorts: list[Target]
    metadata: dict[str, Any]
    language: str
    thumbnail: Path
    thumbnail_hash: str
    captions: Path | None
    captions_hash: str | None
    caption_language: str
    caption_name: str
    warnings: list[str] = field(default_factory=list)


def prepare(ctx: RunContext, options: UploadOptions, script_sha256: str) -> Plan:
    """Check every precondition and collect what will be uploaded. Raises :class:`UploadRefused`."""
    script, state, paths, settings = ctx.script, ctx.state, ctx.paths, ctx.settings
    problems: list[str] = []
    warnings: list[str] = []

    if script.status != "approved":
        problems.append(f"script status is '{script.status}' — only 'approved' scripts are uploaded")
    if not script.qa.facts_verified:
        problems.append("qa.facts_verified is false — facts must be verified before publishing")
    want_shorts = options.shorts and bool(script.shorts)
    rerun = "run `python -m pipeline run <script.json>` first (finished work is skipped)"
    for stage in (*REQUIRED_STAGES, *(("shorts",) if want_shorts else ())):
        status = state.stage_record(stage).status
        if status != "done":
            problems.append(f"stage '{stage}' is '{status}', not 'done' — {rerun}")
    if state.script_sha256 and state.script_sha256 != script_sha256:
        problems.append(f"the script changed after the last pipeline run — {rerun}")
    if state.mock:
        if options.dry_run:
            warnings.append("the last pipeline run was --mock (placeholder voice and pictures): a real upload of it "
                            "will be refused")
        else:
            problems.append("the last pipeline run was --mock: the files are placeholders. Run the pipeline "
                            "without --mock before uploading")

    def need(path: Path, what: str) -> bool:
        if path.is_file() and path.stat().st_size > 0:
            return True
        problems.append(f"{what} is missing: {settings.rel(path)} — {rerun}")
        return False

    long_file = paths.final / "long.mp4"
    need(long_file, "the long video")
    thumbnail = paths.final / "thumbnail.jpg"
    if need(thumbnail, "the thumbnail") and thumbnail.stat().st_size >= THUMBNAIL_MAX_BYTES:
        problems.append(f"the thumbnail is {fmt_size(thumbnail.stat().st_size)} — it must be under 2 MB; re-run "
                        f"`python -m pipeline run <script.json> --stages thumbnail --force`")
    metadata: dict[str, Any] = {}
    if need(paths.metadata, "metadata.json"):
        try:
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            for name in ("title", "description", "tags", "category_id", "made_for_kids", "contains_synthetic_media"):
                if name not in metadata:
                    raise KeyError(name)
        except (ValueError, KeyError) as exc:
            problems.append(f"metadata.json is damaged ({type(exc).__name__}: {exc}) — {rerun}")
            metadata = {}
    caption_language = script.language.split("-")[0]
    captions: Path | None = None
    if script.production.subtitles.caption_tracks:
        captions = paths.captions / f"{caption_language}.srt"
        need(captions, "the caption file")

    short_files: list[tuple[Any, Path]] = []
    if want_shorts:
        for short in script.shorts:
            path = paths.shorts / f"{short.short_id}.mp4"
            record = state.shorts.get(short.short_id)
            if record is None or record.status != "done":
                problems.append(f"short {short.short_id} is not rendered — {rerun}")
            elif need(path, f"short {short.short_id}"):
                short_files.append((short, path))

    if metadata:
        problems += metadata_problems(str(metadata["title"]), str(metadata["description"]), list(metadata["tags"]))
    if problems:
        raise UploadRefused("\n".join(problems))

    duration = state.timeline.duration_sec if state.timeline else metadata.get("duration_sec")
    long_artifact = state.artifacts.get("long")
    long = Target(key="long", kind="long", file=long_file, size=long_file.stat().st_size, duration_sec=duration,
                  source_hash=long_artifact.hash if long_artifact else file_hash(long_file),
                  title=str(metadata["title"]))
    if duration and duration > LONG_UPLOAD_LIMIT_SEC:
        warnings.append(f"the long video is {fmt_duration(duration)} — longer than 15 minutes: YouTube accepts it "
                        f"only from a phone-verified channel ({VERIFY_HINT})")

    shorts: list[Target] = []
    for short, path in short_files:
        record = state.shorts[short.short_id]
        final = record.artifacts.get("final")
        hook_line = (short.hook_override_text or "").strip() or script.scene_by_id(short.scene_ids[0]).display_text
        target = Target(key=short.short_id, kind="short", file=path, size=path.stat().st_size,
                        duration_sec=record.duration_sec, source_hash=final.hash if final else file_hash(path),
                        title=short_title(short.hook_override_text, long.title), hook_line=hook_line,
                        notify_subscribers=False)
        if record.duration_sec and record.duration_sec > SHORT_MAX_SEC:
            warnings.append(f"short {short.short_id} is {fmt_duration(record.duration_sec)} long — over 3 minutes, "
                            f"YouTube will show it as a regular video, not a Short")
        shorts.append(target)

    if options.publish_at and options.privacy != "private":
        warnings.append(f"--publish-at needs a private video: --privacy {options.privacy} is ignored, the videos "
                        f"are uploaded as private and go public at {options.publish_at}")
    return Plan(long=long, shorts=shorts, metadata=metadata, language=script.language, thumbnail=thumbnail,
                thumbnail_hash=state.thumbnail.hash or file_hash(thumbnail), captions=captions,
                captions_hash=file_hash(captions) if captions else None, caption_language=caption_language,
                caption_name=CAPTION_NAMES.get(caption_language, caption_language), warnings=warnings)


def body_for(plan: Plan, target: Target, options: UploadOptions, long_video_id: str | None) -> dict[str, Any]:
    meta = plan.metadata
    description = (str(meta["description"]) if target.kind == "long"
                   else short_description(target.hook_line, long_video_id or "<ID of the long video>"))
    return video_body(title=target.title, description=description, tags=list(meta["tags"]),
                      category_id=meta["category_id"], made_for_kids=bool(meta["made_for_kids"]),
                      contains_synthetic_media=bool(meta["contains_synthetic_media"]), language=plan.language,
                      privacy=options.effective_privacy, publish_at=options.publish_at)


def _record(ctx: RunContext, target: Target) -> UploadedVideo:
    upload = ctx.state.upload
    if target.kind == "long":
        if upload.long is None:
            upload.long = UploadedVideo(key="long", kind="long")
        return upload.long
    return upload.shorts.setdefault(target.key, UploadedVideo(key=target.key, kind="short"))


def pending_steps(ctx: RunContext, plan: Plan, options: UploadOptions) -> dict[str, list[str]]:
    """What is still missing on YouTube, per target key (``upload`` / ``thumbnail`` / ``captions`` / ``verify``)."""
    steps: dict[str, list[str]] = {}
    for target in [plan.long, *plan.shorts]:
        record = _record(ctx, target)
        todo: list[str] = []
        fresh = options.force or not record.video_id
        if fresh:
            todo.append("upload")
        if target.kind == "long":
            if fresh or not record.thumbnail_set or record.thumbnail_hash != plan.thumbnail_hash:
                todo.append("thumbnail")
            if plan.captions and (fresh or not record.captions_uploaded):
                todo.append("captions")
        if fresh or not record.verified_at:
            todo.append("verify")
        if todo:
            steps[target.key] = todo
    return steps


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def print_plan(ctx: RunContext, plan: Plan, options: UploadOptions, out: Callable[[str], None]) -> None:
    settings, script = ctx.settings, ctx.script
    steps = pending_steps(ctx, plan, options)
    try:
        token = token_path(settings.secrets_dir, script.channel_id)
        token_note = (f"{settings.rel(token)} — exists" if token.is_file()
                      else f"{settings.rel(token)} — not created yet: the first real run opens a browser for login")
    except Exception as exc:  # noqa: BLE001 — AuthError: unusable channel_id
        token_note = f"PROBLEM: {exc}"
    out("DRY RUN — nothing is sent to YouTube and no login is performed.")
    out(f"channel_id   : {script.channel_id}")
    out(f"token file   : {token_note}")
    if not settings.client_secret_path.is_file():
        out(f"client secret: MISSING ({settings.rel(settings.client_secret_path)}) — a real upload needs it "
            f"(README, section about YouTube, steps 1-4)")
    out(f"privacy      : {options.effective_privacy}")
    if options.publish_at:
        out(f"publish at   : {options.publish_at} (UTC) — the videos stay private until then")
    if options.force:
        out("mode         : --force-upload — everything is uploaded again as NEW videos; old ones are not deleted")
    for warning in plan.warnings:
        out(f"WARNING      : {warning}")

    uploads = 0
    units = 0
    for index, target in enumerate([plan.long, *plan.shorts], start=1):
        record = _record(ctx, target)
        todo = steps.get(target.key, [])
        if "upload" in todo:
            uploads += 1
            headline = "WILL UPLOAD (resumable)"
            if record.video_id:
                headline += f" as a new video; the old one stays: {record.url}"
        else:
            headline = f"already on YouTube: {record.url} — not uploaded again"
        out("")
        out(f"[{index}] {target.label} — {headline}")
        out(f"    file        : {settings.rel(target.file)} ({fmt_size(target.size)}, {fmt_duration(target.duration_sec)})")
        body = body_for(plan, target, options, ctx.state.upload.long.video_id if ctx.state.upload.long else None)
        snippet, status = body["snippet"], body["status"]
        out(f"    title       : {snippet['title']}  ({len(snippet['title'])} characters)")
        out(f"    description : ({len(snippet['description'])} characters)")
        for line in snippet["description"].split("\n"):
            out(f"        | {line}")
        out(f"    tags        : {', '.join(snippet['tags'])}")
        out(f"    categoryId  : {snippet['categoryId']}")
        out(f"    language    : {snippet['defaultLanguage']} (defaultLanguage and defaultAudioLanguage)")
        out(f"    flags       : selfDeclaredMadeForKids={status['selfDeclaredMadeForKids']}, "
            f"containsSyntheticMedia={status['containsSyntheticMedia']}")
        out(f"    privacy     : {status['privacyStatus']}"
            + (f", publishAt={status['publishAt']}" if "publishAt" in status else ""))
        if target.kind == "long":
            thumb_note = "WILL SET" if "thumbnail" in todo else "already set — skipped"
            units += 50 if "thumbnail" in todo else 0
            out(f"    thumbnail   : {settings.rel(plan.thumbnail)} ({fmt_size(plan.thumbnail.stat().st_size)}) — {thumb_note}")
            if plan.captions:
                cap_note = "WILL UPLOAD" if "captions" in todo else "already uploaded — skipped"
                units += 400 if "captions" in todo else 0
                out(f"    captions    : {settings.rel(plan.captions)} -> language '{plan.caption_language}', name "
                    f"'{plan.caption_name}', not a draft — {cap_note}")
            else:
                out("    captions    : none (subtitles.caption_tracks is false)")
        else:
            out("    subscribers : not notified (notifySubscribers=false for Shorts)")
    out("")
    if not steps:
        out("Nothing to do: everything is already on YouTube.")
        return
    out("after upload : videos.list check — a warning if YouTube saved another privacy status than requested")
    out(f"quota        : {uploads} upload(s) from the uploads bucket (100 a day by default) + about {units + 2} of "
        f"the 10,000 daily units for the other calls")
    if plan.shorts:
        out("reminder     : the API cannot set a Short's 'Related video' — after the upload you will get Studio "
            "links to do it by hand")


# --------------------------------------------------------------------------- #
# real run
# --------------------------------------------------------------------------- #
PRIVACY_WARNING = (
    "This is the sign of an unverified API project: until 2026-11-02 YouTube forced every upload from a project "
    "that had not passed the YouTube API Services audit to stay private. The documentation now says that "
    "restriction is lifted, so if you still see this, check the video in Studio and the README section about the "
    "audit. Until it is sorted out, publish by hand in YouTube Studio.")


def _upload_one(ctx: RunContext, api: YouTubeAPI, plan: Plan, target: Target, options: UploadOptions) -> None:
    record = _record(ctx, target)
    long_id = ctx.state.upload.long.video_id if ctx.state.upload.long else None
    body = body_for(plan, target, options, long_id)
    body_hash = stable_hash(body)
    resume_url = None
    session = record.session
    if session and (session.source_hash, session.body_hash, session.size) == (target.source_hash, body_hash, target.size):
        resume_url = session.url
        ctx.log.info("%s: an unfinished upload was found — asking YouTube how much it already has", target.label)
    elif session:
        record.session = None

    def on_session(url: str) -> None:
        record.session = UploadSession(url=url, source_hash=target.source_hash, body_hash=body_hash, size=target.size)
        ctx.save()

    last = [-1]

    def on_progress(sent: int, total: int) -> None:
        percent = 100 * sent // max(total, 1)
        if percent >= last[0] + 5 or sent >= total:
            last[0] = percent
            ctx.log.info("%s: %3d%%  (%s of %s)", target.label, percent, fmt_size(sent), fmt_size(total))

    ctx.log.info("%s: uploading %s (%s, %s) — '%s'", target.label, ctx.settings.rel(target.file),
                 fmt_size(target.size), fmt_duration(target.duration_sec), target.title)
    resource = api.upload_video(target.file, body, notify_subscribers=target.notify_subscribers,
                                resume_url=resume_url, on_session=on_session, on_progress=on_progress)
    video_id = str(resource.get("id") or "")
    if not video_id:
        raise YouTubeAPIError(f"{target.label}: YouTube accepted the file but returned no video ID",
                              "check YouTube Studio; if the video is there, do not upload again.",
                              method="videos.insert")
    record.video_id, record.uploaded_at, record.session = video_id, now_iso(), None
    record.source_hash, record.title = target.source_hash, target.title
    record.requested_privacy, record.publish_at = options.effective_privacy, options.publish_at
    record.actual_privacy = record.upload_status = record.verified_at = None
    record.thumbnail_set = record.captions_uploaded = False
    record.thumbnail_hash = record.captions_hash = record.caption_id = None
    ctx.save()
    ctx.log.info("%s: uploaded -> %s", target.label, record.url)


def _verify(ctx: RunContext, api: YouTubeAPI, plan: Plan, records: list[UploadedVideo], failures: list[str]) -> None:
    """``videos.list`` after the upload: did YouTube keep the privacy status (and the language) we asked for?"""
    found = api.list_videos([r.video_id for r in records if r.video_id])
    for record in records:
        item = found.get(record.video_id or "")
        label = "long video" if record.kind == "long" else f"short {record.key}"
        if item is None:
            ctx.log.warning("%s: videos.list does not return %s — was it deleted in Studio?", label, record.video_id)
            continue
        status, snippet = item.get("status", {}), item.get("snippet", {})
        record.actual_privacy = status.get("privacyStatus")
        record.upload_status = status.get("uploadStatus")
        record.verified_at = now_iso()
        if record.actual_privacy != record.requested_privacy:
            bar = "!" * 78
            for line in (bar, f"WARNING: {label} {record.url} is '{record.actual_privacy}' on YouTube, but "
                              f"'{record.requested_privacy}' was requested.", PRIVACY_WARNING, bar):
                ctx.log.warning(line)
        if record.publish_at and not status.get("publishAt"):
            ctx.log.warning("%s: YouTube did not keep the scheduled time %s — set it in Studio: %s", label,
                            record.publish_at, f"https://studio.youtube.com/video/{record.video_id}/edit")
        if record.upload_status in ("rejected", "failed"):
            reason = status.get("rejectionReason") or status.get("failureReason") or "no reason given"
            hint = f" — {VERIFY_HINT}" if reason == "length" else ""
            failures.append(f"{label}: YouTube marked the upload as {record.upload_status} ({reason}){hint}")
        audio_language = str(snippet.get("defaultAudioLanguage") or "")
        if audio_language.split("-")[0].lower() != plan.language.split("-")[0].lower():
            ctx.log.warning("%s: YouTube did not keep the audio language '%s' (it reports '%s') — set 'Video "
                            "language' in Studio, auto-dubbing depends on it", label, plan.language,
                            audio_language or "none")
    ctx.save()


def summarize(ctx: RunContext, log_line: Callable[[str], None]) -> None:
    """Links to everything on YouTube plus the manual 'Related video' reminder for Shorts."""
    upload = ctx.state.upload
    for record in upload.videos():
        if not record.video_id:
            continue
        name = "long video" if record.kind == "long" else f"short {record.key}"
        privacy = record.actual_privacy or record.requested_privacy or "?"
        extra = f", goes public at {record.publish_at}" if record.publish_at else ""
        log_line(f"{name}: {record.url}  ({privacy}{extra})")
    shorts = [r for r in upload.shorts.values() if r.video_id]
    if shorts and upload.long and upload.long.video_id:
        log_line("REMINDER: the YouTube API has no field for a Short's 'Related video'. Set it by hand — open each "
                 "link, Details -> Related video -> choose the long video "
                 f"'{upload.long.title or ''}' ({upload.long.url}):")
        for record in shorts:
            log_line(f"  https://studio.youtube.com/video/{record.video_id}/edit")


def run_upload(ctx: RunContext, options: UploadOptions, script_sha256: str, *, connector: Connector | None = None,
               out: Callable[[str], None] = print) -> int:
    """The whole command. Returns the exit code; raises :class:`UploadRefused` for unmet preconditions."""
    plan = prepare(ctx, options, script_sha256)
    if options.dry_run:
        print_plan(ctx, plan, options, out)
        return 0

    log = ctx.log
    for warning in plan.warnings:
        log.warning(warning)
    upload = ctx.state.upload

    if options.force:
        # Only what this run uploads is forgotten: with --no-shorts the Shorts already on YouTube stay recorded.
        for record in [_record(ctx, target) for target in [plan.long, *plan.shorts]]:
            if record.video_id:
                log.warning("--force-upload: %s was already uploaded as %s — that video is NOT deleted and becomes "
                            "a duplicate; remove it in YouTube Studio if it is not needed", record.key, record.url)
                record.previous_video_ids.append(record.video_id)
            record.video_id = record.uploaded_at = record.verified_at = record.session = None
            record.thumbnail_set = record.captions_uploaded = False
        ctx.save()
        options = UploadOptions(privacy=options.privacy, publish_at=options.publish_at, shorts=options.shorts)

    steps = pending_steps(ctx, plan, options)
    for target in [plan.long, *plan.shorts]:
        record = _record(ctx, target)
        if record.video_id and record.source_hash and record.source_hash != target.source_hash:
            log.warning("%s was re-rendered after it was uploaded (%s): YouTube still has the old file. Use "
                        "--force-upload to upload the new one as a new video", target.label, record.url)
        if record.video_id and record.requested_privacy and (
                record.requested_privacy != options.effective_privacy or record.publish_at != options.publish_at):
            log.warning("%s is already uploaded (%s) with privacy '%s'%s — this command does not change uploaded "
                        "videos; change it in YouTube Studio", target.label, record.url, record.requested_privacy,
                        f", publishAt {record.publish_at}" if record.publish_at else "")
    if not steps:
        log.info("nothing to upload — everything is already on YouTube:")
        for record in upload.videos():
            if record.video_id and record.actual_privacy and record.actual_privacy != record.requested_privacy:
                log.warning("%s is '%s' on YouTube although '%s' was requested. %s", record.url,
                            record.actual_privacy, record.requested_privacy, PRIVACY_WARNING)
        summarize(ctx, log.info)
        return 0

    conn = (connector or connect)(ctx.settings, ctx.script.channel_id, log)
    ctx.log = log = setup_logging(ctx.paths.log_file, [*ctx.settings.secrets(), *conn.secrets])
    api, channel = conn.api, conn.channel
    upload.youtube_channel_id, upload.youtube_channel_title = channel.id, channel.title
    ctx.save()

    failures: list[str] = []
    try:
        # A channel that is not phone-verified (longUploadsStatus "eligible") or not in good standing
        # ("disallowed") cannot take long videos, and usually no custom thumbnails either: say so before sending.
        if channel.long_uploads_allowed is False:
            duration = plan.long.duration_sec or 0
            if "upload" in steps.get("long", []) and duration > LONG_UPLOAD_LIMIT_SEC:
                raise YouTubeError(
                    f"the long video is {fmt_duration(duration)} but channel '{channel.title}' is not allowed to "
                    f"upload videos longer than 15 minutes — nothing was uploaded",
                    f"{channel.long_uploads_hint}. Then run the same command again.")
            if "thumbnail" in steps.get("long", []):
                why = ("is not phone-verified" if channel.long_uploads_status == "eligible"
                       else "is not in good standing")
                log.warning("channel '%s' %s: YouTube will probably refuse the custom thumbnail (%s)",
                            channel.title, why, channel.long_uploads_hint)

        # 1. long video, then its thumbnail and captions
        long_record = _record(ctx, plan.long)
        long_steps = steps.get("long", [])
        if "upload" in long_steps:
            _upload_one(ctx, api, plan, plan.long, options)

        def step(label: str, action: Callable[[], None]) -> None:
            """Run one follow-up call; an ordinary failure is reported and the rest goes on."""
            try:
                action()
            except (QuotaExceeded, UploadLimitExceeded, AuthExpired):
                raise
            except YouTubeError as exc:
                failures.append(f"{label}: {exc}" + (f" — what to do: {exc.hint}" if exc.hint else ""))
                log.error("%s failed: %s", label, exc)

        def set_thumbnail() -> None:
            api.set_thumbnail(long_record.video_id or "", plan.thumbnail)
            long_record.thumbnail_set, long_record.thumbnail_hash = True, plan.thumbnail_hash
            ctx.save()
            log.info("thumbnail: set (%s)", ctx.settings.rel(plan.thumbnail))

        def insert_captions() -> None:
            assert plan.captions is not None
            try:
                long_record.caption_id = api.insert_caption(long_record.video_id or "", plan.captions,
                                                            language=plan.caption_language, name=plan.caption_name)
            except YouTubeAPIError as exc:
                if exc.reason != "captionExists":
                    raise
                log.warning("captions: YouTube already has a '%s' track named '%s' for this video — left as is",
                            plan.caption_language, plan.caption_name)
            long_record.captions_uploaded, long_record.captions_hash = True, plan.captions_hash
            ctx.save()
            log.info("captions: uploaded (%s, language '%s', name '%s')", ctx.settings.rel(plan.captions),
                     plan.caption_language, plan.caption_name)

        if "thumbnail" in long_steps:
            step("thumbnail", set_thumbnail)
        if "captions" in long_steps:
            step("captions", insert_captions)
        elif plan.captions and long_record.captions_hash and long_record.captions_hash != plan.captions_hash:
            log.warning("captions/%s changed after it was uploaded: YouTube still has the old track (replace it in "
                        "Studio -> Subtitles)", plan.captions.name)

        # 2. Shorts — each one links to the long video in its description
        for target in plan.shorts:
            if "upload" in steps.get(target.key, []):
                step(target.label, lambda t=target: _upload_one(ctx, api, plan, t, options))

        # 3. check what YouTube actually stored
        to_verify = [r for r in upload.videos() if r.video_id and not r.verified_at
                     and (r.kind == "long" or options.shorts)]
        if to_verify:
            step("verification", lambda: _verify(ctx, api, plan, to_verify, failures))
    except AuthExpired as exc:
        conn.token_path.unlink(missing_ok=True)
        raise TokenExpired(f"{exc}. {conn.token_path.name} has been deleted.", TESTING_HINT) from None
    finally:
        ctx.save()
        if conn.token_path.exists():
            try:
                conn.persist()
            except Exception:  # noqa: BLE001 — saving refreshed credentials is best effort
                log.debug("could not save refreshed credentials", exc_info=True)

    log.info("=== on YouTube ===")
    summarize(ctx, log.info)
    if failures:
        log.error("%d step(s) failed — fix the cause and run the same command again: only the missing parts are "
                  "sent", len(failures))
        for failure in failures:
            log.error("  %s", failure)
        return 1
    log.info("=== upload done ===")
    return 0
