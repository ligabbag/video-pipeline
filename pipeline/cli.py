"""Command line: ``python -m pipeline validate|run|status|upload``."""
from __future__ import annotations

import argparse
import dataclasses
import platform
import sys
from pathlib import Path
from typing import Callable, Sequence

from pipeline.config import OutputPaths, Settings
from pipeline.context import RunContext
from pipeline.contract import ContractError, load_script, run_validator_cli
from pipeline.ffmpeg import ensure_ffmpeg
from pipeline.logging_setup import redact, setup_logging
from pipeline.state import RUN_STAGES, PipelineState, StageState, StateStore, ThumbnailState, now_iso

EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


def _stage_functions() -> dict[str, Callable[[RunContext], None]]:
    # Imported lazily so `validate` and `status` work without touching FFmpeg-related modules.
    from pipeline.images import run_images_stage
    from pipeline.render import run_render_stage
    from pipeline.shorts import run_shorts_stage
    from pipeline.thumbnail import run_thumbnail_stage
    from pipeline.tts import run_tts_stage

    return {"tts": run_tts_stage, "images": run_images_stage, "thumbnail": run_thumbnail_stage,
            "render": run_render_stage, "shorts": run_shorts_stage}


def parse_stages(value: str | None) -> list[str]:
    if not value:
        return list(RUN_STAGES)
    wanted = [s.strip() for s in value.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in RUN_STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s): {', '.join(unknown)} (choose from {', '.join(RUN_STAGES)})")
    return [s for s in RUN_STAGES if s in wanted]  # always run in pipeline order


def reset_for_force(state: PipelineState, stages: Sequence[str]) -> None:
    """``--force``: forget what the selected stages produced so everything is rebuilt.

    Cost counters are history and are kept.
    """
    for scene in state.scenes.values():
        if "tts" in stages:
            scene.tts = None
            scene.timing = None
        if "images" in stages:
            scene.image = None
        if "render" in stages:
            scene.clip = None
    if "tts" in stages:
        state.timeline = None
    if "render" in stages:
        state.artifacts.clear()
    if "shorts" in stages:
        state.shorts.clear()
    if "thumbnail" in stages:
        state.thumbnail = ThumbnailState()
    for stage in stages:
        if stage != "thumbnail":
            state.stages[stage] = StageState()


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        return run_validator_cli(Path(args.script))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def cmd_run(args: argparse.Namespace) -> int:
    settings = Settings.load(Path(args.root) if args.root else None)
    if args.image_provider:
        settings = dataclasses.replace(settings, image_provider_override=args.image_provider)
    script_path = Path(args.script)
    try:
        stages = parse_stages(args.stages)
        loaded = load_script(script_path)
    except (ContractError, FileNotFoundError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    script = loaded.script

    # Approval is a human action: only approved scripts are rendered for real.
    if not args.mock and script.status != "approved":
        print(f"refused: script status is '{script.status}', only 'approved' scripts are rendered "
              f"(use --mock for a dry run without API calls)", file=sys.stderr)
        return EXIT_REFUSED
    if script.production.aspect_ratio != "16:9" or script.production.resolution != "1920x1080":
        print("refused: only 16:9 1920x1080 master scripts are supported; vertical videos come from shorts[]",
              file=sys.stderr)
        return EXIT_REFUSED

    paths = OutputPaths(settings.output_root / script.video_id)
    paths.ensure()
    log = setup_logging(paths.log_file, settings.secrets(), verbose=args.verbose)
    log.info("=== run %s | stages: %s | mock=%s force=%s ===", script.video_id, ",".join(stages), args.mock, args.force)
    for warning in loaded.report.warnings:
        log.warning("validator: %s", warning)
    if args.hw and platform.system() != "Darwin":
        log.warning("--hw is only available on macOS (VideoToolbox) — using libx264")

    store = StateStore(settings.state_dir, script.video_id)
    state = store.load()
    state.script_sha256 = loaded.sha256
    state.mock = bool(args.mock)
    valid_ids = set(script.scene_ids())
    for stale in [scene_id for scene_id in state.scenes if scene_id not in valid_ids]:
        del state.scenes[stale]
    if args.force:
        reset_for_force(state, stages)
    store.save(state)

    ctx = RunContext(settings=settings, script=script, script_path=script_path, state=state, store=store,
                     paths=paths, log=log, mock=bool(args.mock), force=bool(args.force),
                     no_cache=bool(args.no_cache), hw=bool(args.hw))
    try:
        ensure_ffmpeg()
        functions = _stage_functions()
        for stage in stages:
            record = state.stage_record(stage)
            record.status, record.started_at, record.finished_at, record.error = "running", now_iso(), None, None
            store.save(state)
            log.info("--- stage %s ---", stage)
            try:
                functions[stage](ctx)
            except Exception as exc:
                record.status = "failed"
                record.error = redact(f"{type(exc).__name__}: {exc}", settings.secrets())[:2000]
                record.finished_at = now_iso()
                store.save(state)
                raise
            record.status, record.finished_at = "done", now_iso()
            store.save(state)
    except KeyboardInterrupt:
        log.error("interrupted — re-run the same command to continue from where it stopped")
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 — top-level boundary: report, keep details in the log file
        log.error("FAILED: %s", exc)
        log.debug("traceback", exc_info=True)
        log.error("fix the cause and re-run the same command: finished work is kept and skipped")
        return EXIT_FAILED
    log.info("=== done: %s ===", settings.rel(paths.base))
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    settings = Settings.load(Path(args.root) if args.root else None)
    store = StateStore(settings.state_dir, args.video_id)
    if not store.exists():
        print(f"no state for '{args.video_id}' in {settings.state_dir}", file=sys.stderr)
        return EXIT_FAILED
    state = store.load()
    print(f"video_id : {state.video_id}")
    print(f"updated  : {state.updated_at}   mock={state.mock}")
    print("stages   :")
    for stage in RUN_STAGES:
        record = state.stage_record(stage)
        line = f"  {stage:<7} {record.status:<8}"
        if record.finished_at:
            line += f" finished {record.finished_at}"
        print(line)
        if record.error:
            print(f"          error: {record.error[:300]}")
    total = len(state.scenes)
    print(f"scenes   : {total} total | audio {sum(1 for s in state.scenes.values() if s.tts)} | "
          f"images {sum(1 for s in state.scenes.values() if s.image)} | "
          f"clips {sum(1 for s in state.scenes.values() if s.clip)}")
    if state.timeline:
        print(f"timeline : {state.timeline.duration_sec:.2f} s, {state.timeline.total_frames} frames "
              f"@ {state.timeline.fps} fps")
    for name in ("long", "stems", "metadata"):
        if name in state.artifacts:
            print(f"{name:<9}: {state.artifacts[name].path}")
    if state.thumbnail.path:
        print(f"thumbnail: {state.thumbnail.path} ({(state.thumbnail.bytes or 0) / 1000:.0f} KB)")
    for short in state.shorts.values():
        duration = f"{short.duration_sec:.1f} s" if short.duration_sec else "-"
        print(f"short    : {short.short_id} {short.status} {duration} {short.path or ''}")
    upload = state.upload
    if upload.videos():
        channel = f" — channel {upload.youtube_channel_title} ({upload.youtube_channel_id})" if upload.youtube_channel_id else ""
        print(f"youtube  :{channel}")
        for video in upload.videos():
            name = "long" if video.kind == "long" else f"short {video.key}"
            if not video.video_id:
                print(f"  {name}: not uploaded" + (" (an interrupted upload will be resumed)" if video.session else ""))
                continue
            notes = [video.actual_privacy or video.requested_privacy or "?"]
            if video.actual_privacy and video.actual_privacy != video.requested_privacy:
                notes.append(f"REQUESTED {video.requested_privacy}!")
            if video.publish_at:
                notes.append(f"goes public at {video.publish_at}")
            if video.kind == "long":
                notes.append("thumbnail set" if video.thumbnail_set else "thumbnail NOT set")
                notes.append("captions uploaded" if video.captions_uploaded else "no captions uploaded")
            print(f"  {name}: {video.url}  ({', '.join(notes)})")
            for old in video.previous_video_ids:
                print(f"    earlier upload still on YouTube: https://youtu.be/{old}")
    costs = state.costs
    print(f"costs    : TTS {costs.tts_characters} chars in {costs.tts_requests} requests "
          f"({costs.tts_cache_hits} cache hits); images {costs.images_generated} generated "
          f"({costs.images_cache_hits} cache hits)")
    return EXIT_OK


def cmd_upload(args: argparse.Namespace) -> int:
    # Imported lazily: `validate`, `run` and `status` must not depend on the upload modules.
    from pipeline.http import RetryableAPIError
    from pipeline.upload import UploadOptions, UploadRefused, parse_publish_at, run_upload
    from pipeline.youtube_api import YouTubeError
    from pipeline.youtube_auth import AuthError

    settings = Settings.load(Path(args.root) if args.root else None)
    script_path = Path(args.script)
    try:
        loaded = load_script(script_path)
        publish_at = parse_publish_at(args.publish_at) if args.publish_at else None
    except (ContractError, FileNotFoundError, UploadRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    script = loaded.script
    store = StateStore(settings.state_dir, script.video_id)
    if not store.exists():
        print(f"refused: nothing has been rendered for '{script.video_id}' yet (no {settings.rel(store.path)}) — "
              f"run `python -m pipeline run {args.script}` first", file=sys.stderr)
        return EXIT_REFUSED
    options = UploadOptions(privacy=args.privacy, publish_at=publish_at, shorts=not args.no_shorts,
                            dry_run=bool(args.dry_run), force=bool(args.force_upload))
    paths = OutputPaths(settings.output_root / script.video_id)
    if options.dry_run:
        log = setup_logging(None, settings.secrets(), verbose=args.verbose)      # a dry run leaves no trace
    else:
        log = setup_logging(paths.log_file, settings.secrets(), verbose=args.verbose)
        log.info("=== upload %s | privacy=%s publish_at=%s shorts=%s force=%s ===", script.video_id,
                 options.effective_privacy, options.publish_at, options.shorts, options.force)
    ctx = RunContext(settings=settings, script=script, script_path=script_path, state=store.load(), store=store,
                     paths=paths, log=log)

    def problem(exc: Exception) -> None:
        log.error("FAILED: %s", exc)
        hint = getattr(exc, "hint", "")
        if hint:
            log.error("what to do: %s", hint)

    try:
        return run_upload(ctx, options, loaded.sha256)
    except UploadRefused as exc:
        print("refused:\n  " + "\n  ".join(str(exc).split("\n")), file=sys.stderr)
        return EXIT_REFUSED
    except AuthError as exc:
        problem(exc)
        return EXIT_REFUSED if exc.refused else EXIT_FAILED
    except YouTubeError as exc:
        problem(exc)
        return EXIT_FAILED
    except RetryableAPIError as exc:       # network trouble before the upload itself (login, token refresh)
        log.error("FAILED: %s", exc)
        log.error("what to do: check the internet connection and run the same command again")
        return EXIT_FAILED
    except KeyboardInterrupt:
        ctx.log.error("interrupted — run the same command again: the upload continues from where it stopped")
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 — top-level boundary: report, keep details in the log file
        ctx.log.error("FAILED: %s: %s", type(exc).__name__, redact(str(exc), settings.secrets())[:500])
        ctx.log.debug("traceback", exc_info=True)
        return EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline",
                                     description="JSON video script -> long video + Shorts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate a script with the supplied validator")
    p_validate.add_argument("script")
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run the pipeline (resumes and skips finished work)")
    p_run.add_argument("script")
    p_run.add_argument("--stages", help=f"comma-separated subset of: {','.join(RUN_STAGES)} (default: all)")
    p_run.add_argument("--mock", action="store_true", help="no API keys: silent TTS, placeholder images; any status")
    p_run.add_argument("--force", action="store_true", help="rebuild everything for the selected stages")
    p_run.add_argument("--no-cache", action="store_true",
                       help="ignore cached API results too (TTS and images are requested and billed again)")
    p_run.add_argument("--hw", action="store_true", help="macOS: encode with h264_videotoolbox")
    p_run.add_argument("--image-provider", choices=["gpt_image", "flux"], help="override visual_style.image_provider")
    p_run.add_argument("--root", help="project root (default: current directory or $PIPELINE_ROOT)")
    p_run.add_argument("-v", "--verbose", action="store_true", help="debug output on the console")
    p_run.set_defaults(func=cmd_run)

    p_upload = sub.add_parser("upload", help="upload the finished video, thumbnail, captions and Shorts to YouTube")
    p_upload.add_argument("script")
    p_upload.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private",
                          help="privacy status of the uploaded videos (default: private)")
    p_upload.add_argument("--publish-at", metavar="ISO8601",
                          help="schedule publication, e.g. 2026-12-01T18:00:00+03:00 (time zone required, must be "
                               "in the future); the videos are uploaded as private")
    p_upload.add_argument("--no-shorts", action="store_true", help="upload only the long video")
    p_upload.add_argument("--dry-run", action="store_true",
                          help="print what would be uploaded and with which metadata; no login, no network")
    p_upload.add_argument("--force-upload", action="store_true",
                          help="upload everything again as new videos (old videos are NOT deleted)")
    p_upload.add_argument("--root", help="project root (default: current directory or $PIPELINE_ROOT)")
    p_upload.add_argument("-v", "--verbose", action="store_true", help="debug output on the console")
    p_upload.set_defaults(func=cmd_upload)

    p_status = sub.add_parser("status", help="show the state of a video and links to the uploaded videos")
    p_status.add_argument("video_id")
    p_status.add_argument("--root", help="project root (default: current directory or $PIPELINE_ROOT)")
    p_status.set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
