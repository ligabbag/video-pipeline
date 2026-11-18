"""End-to-end mock runs through the real CLI: the supplied example, plus a variant script."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest

from pipeline import tts as tts_mod
from pipeline.audio import wav_samples
from pipeline.cli import main
from pipeline.ffmpeg import probe
from tests.conftest import EXAMPLE, small_doc

VIDEO_ID = "tech-0001-knight-capital"
SHORT_ID = "tech-0001-knight-capital-sh1"


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Run ``pipeline run <example> --mock`` once in a clean project root."""
    root = tmp_path_factory.mktemp("project")
    script = root / "video_script.example.json"
    shutil.copyfile(EXAMPLE, script)
    patch = pytest.MonkeyPatch()
    patch.setenv("PIPELINE_SUPERSAMPLE", "2")          # faster; the 4x path is covered in test_audio_render
    for name in ("ELEVENLABS_API_KEY", "OPENAI_API_KEY", "BFL_API_KEY"):
        patch.delenv(name, raising=False)              # mock mode must work without any key
    code = main(["run", str(script), "--mock", "--root", str(root)])
    yield {"root": root, "script": script, "code": code, "out": root / "output" / VIDEO_ID,
           "state": root / "state" / f"{VIDEO_ID}.json"}
    patch.undo()


def load_state(run: dict[str, Any]) -> dict[str, Any]:
    return json.loads(run["state"].read_text(encoding="utf-8"))


def test_mock_run_succeeds_and_produces_the_documented_layout(mock_run: dict[str, Any]) -> None:
    assert mock_run["code"] == 0
    out: Path = mock_run["out"]
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    scene_ids = [s["scene_id"] for sec in doc["sections"] for s in sec["scenes"]]
    for scene_id in scene_ids:
        assert (out / "audio" / f"{scene_id}.mp3").is_file()
        assert (out / "clips" / f"{scene_id}.mp4").is_file()
    reuse = {"s007", "s011", "s014"}
    assert {p.stem for p in (out / "images").glob("s*.png")} == set(scene_ids) - reuse
    for name in ("stems/voice.wav", "stems/music.wav", "stems/sfx.wav", "stems/mix.wav", "final/long.mp4",
                 f"final/shorts/{SHORT_ID}.mp4", "captions/en.srt", "metadata.json", "pipeline.log"):
        assert (out / name).is_file(), name
    assert mock_run["script"].read_bytes() == EXAMPLE.read_bytes()        # the script is never modified


def test_long_video_is_frame_exact_and_in_sync(mock_run: dict[str, Any]) -> None:
    state = load_state(mock_run)
    fps = state["timeline"]["fps"]
    total = state["timeline"]["total_frames"]
    info = probe(mock_run["out"] / "final" / "long.mp4")
    assert (info.width, info.height) == (1920, 1080) and info.fps == pytest.approx(30)
    assert info.video_frames == total
    assert info.video_duration is not None and info.audio_duration is not None
    assert abs(info.video_duration - info.audio_duration) <= 1 / fps      # at most one frame apart
    assert info.video_duration == pytest.approx(total / fps, abs=1e-3)
    for stem in ("voice", "music", "sfx", "mix"):
        assert wav_samples(mock_run["out"] / "stems" / f"{stem}.wav") == total * (48_000 // fps)
    assert sum(s["clip"]["frames"] for s in state["scenes"].values()) == total


def test_state_file_records_everything_the_spec_asks_for(mock_run: dict[str, Any]) -> None:
    state = load_state(mock_run)
    assert {k: v["status"] for k, v in state["stages"].items()} == dict.fromkeys(("tts", "images", "render", "shorts"), "done")
    assert state["mock"] is True and len(state["scenes"]) == 14
    previous_end = 0
    for scene_id in sorted(state["scenes"]):
        scene = state["scenes"][scene_id]
        assert (mock_run["root"] / scene["tts"]["audio_path"]).is_file()
        assert scene["tts"]["duration_sec"] > 1 and scene["tts"]["hash"] and scene["image"]["hash"]
        assert (mock_run["root"] / scene["image"]["path"]).is_file()
        words = scene["tts"]["words"]
        assert words and all(w["end"] >= w["start"] for w in words)
        assert [w["start"] for w in words] == sorted(w["start"] for w in words)
        timing = scene["timing"]
        assert timing["start_frame"] == previous_end                    # scenes are contiguous
        pause = timing["end_sec"] - timing["start_sec"] - scene["tts"]["duration_sec"]
        assert pause >= 0.15 - 1e-6
        previous_end = timing["end_frame"]
    # last scene of a section holds the longer section pause
    s4 = state["scenes"]["s004"]
    assert s4["timing"]["end_sec"] - s4["timing"]["start_sec"] - s4["tts"]["duration_sec"] >= 0.6 - 1e-6
    # subtitle words keep the on-screen notation; reuse scenes point at the source picture
    assert [w["text"] for w in state["scenes"]["s002"]["tts"]["words"]][-2:] == ["$460", "million."]
    assert state["scenes"]["s007"]["image"]["path"] == state["scenes"]["s003"]["image"]["path"]
    assert state["costs"] == {"tts_characters": 0, "tts_requests": 0, "tts_cache_hits": 0,
                              "images_generated": 0, "images_cache_hits": 0}   # mock bills nothing


def test_captions_and_metadata(mock_run: dict[str, Any]) -> None:
    srt = (mock_run["out"] / "captions" / "en.srt").read_text(encoding="utf-8")
    blocks = [b for b in srt.strip().split("\n\n") if b]
    assert len(blocks) > 14
    for block in blocks:
        lines = block.split("\n")
        assert " --> " in lines[1] and 1 <= len(lines[2:]) <= 2
        assert all(len(line) <= 42 for line in lines[2:])
    assert "$460 million." in srt.replace("\n", " ") and "four hundred" not in srt   # subtitle notation
    assert "SMARS" in srt and "smarz" not in srt                                      # original spelling

    meta = json.loads((mock_run["out"] / "metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == "How One Forgotten Server Lost $460 Million in 45 Minutes"
    assert "{{" not in meta["description"]
    assert "0:00 45 Minutes" in meta["description"] and "The Machine" in meta["description"]
    assert "https://www.sec.gov/news/press-release/2013-222" in meta["description"]
    assert meta["category_id"] == 28 and meta["made_for_kids"] is False and meta["contains_synthetic_media"] is False
    assert meta["chapters"][0]["start_sec"] == 0.0 and len(meta["tags"]) == 6
    starts = [c["start_sec"] for c in meta["chapters"]] + [meta["duration_sec"]]
    assert all(b - a >= 10 for a, b in zip(starts, starts[1:]))


def test_short_is_vertical_in_sync_and_starts_with_the_hook(mock_run: dict[str, Any]) -> None:
    state = load_state(mock_run)
    short = state["shorts"][SHORT_ID]
    info = probe(mock_run["out"] / "final" / "shorts" / f"{SHORT_ID}.mp4")
    assert (info.width, info.height) == (1080, 1920)
    assert info.video_frames == short["frames"]
    assert info.video_duration is not None and info.audio_duration is not None
    assert abs(info.video_duration - info.audio_duration) <= 1 / 30
    assert short["status"] == "done" and short["duration_sec"] < 60
    assert short["hook"]["words"][0]["text"] == "This"
    assert (mock_run["root"] / short["hook"]["audio_path"]).is_file()
    assert list(short["scenes"])[0] == "hook" and short["scenes"]["hook"]["start_frame"] == 0
    ass = (mock_run["out"] / "work" / "shorts" / SHORT_ID / "subs.ass").read_text(encoding="utf-8")
    assert "Montserrat ExtraBold" in ass and "&H00D4FF&" in ass and "Full story on the channel" in ass
    log = (mock_run["out"] / "pipeline.log").read_text(encoding="utf-8")
    assert "Shorts limit" not in log                                    # under 60 s: no warning


def test_second_run_skips_everything(mock_run: dict[str, Any]) -> None:
    out: Path = mock_run["out"]
    watched = [out / "final" / "long.mp4", out / "clips" / "s001.mp4", out / "audio" / "s001.mp3",
               out / "stems" / "mix.wav", out / "final" / "shorts" / f"{SHORT_ID}.mp4", out / "images" / "s001.png"]
    before = {p: p.stat().st_mtime_ns for p in watched}
    assert main(["run", str(mock_run["script"]), "--mock", "--root", str(mock_run["root"])]) == 0
    assert {p: p.stat().st_mtime_ns for p in watched} == before
    log = (out / "pipeline.log").read_text(encoding="utf-8")
    assert "clips: 0 to render, 14 up to date" in log and "long.mp4 up to date" in log


def test_status_and_validate_commands(mock_run: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", VIDEO_ID, "--root", str(mock_run["root"])]) == 0
    text = capsys.readouterr().out
    assert "render  done" in text and "scenes   : 14 total | audio 14 | images 14 | clips 14" in text
    assert main(["status", "nope-0000-x", "--root", str(mock_run["root"])]) == 1
    capsys.readouterr()
    assert main(["validate", str(mock_run["script"])]) == 0
    assert "OK: 0 errors" in capsys.readouterr().out


def test_only_approved_scripts_render_without_mock(mock_run: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(mock_run["script"]), "--root", str(mock_run["root"])]) == 2
    assert "only 'approved'" in capsys.readouterr().err
    broken = mock_run["root"] / "broken.json"
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["sections"][0]["scenes"][0]["source_refs"] = ["src_missing"]
    broken.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["run", str(broken), "--mock", "--root", str(mock_run["root"])]) == 2      # validator errors stop a run
    assert "unknown source_ref" in capsys.readouterr().err


def test_resume_after_failure_and_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash mid-stage keeps finished scenes; the next run continues; --force redoes the stage."""
    monkeypatch.setenv("PIPELINE_SUPERSAMPLE", "1")
    script = tmp_path / "script.json"
    shutil.copyfile(EXAMPLE, script)
    real = tts_mod.MockTTS.synthesize
    calls: list[str] = []

    def flaky(self: tts_mod.MockTTS, request: tts_mod.TTSRequest) -> tts_mod.TTSResult:
        calls.append(request.text)
        if len(calls) == 4:
            raise RuntimeError("simulated crash")
        return real(self, request)

    monkeypatch.setattr(tts_mod.MockTTS, "synthesize", flaky)
    args = ["run", str(script), "--mock", "--stages", "tts", "--root", str(tmp_path)]
    assert main(args) == 1
    state_path = tmp_path / "state" / f"{VIDEO_ID}.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stages"]["tts"]["status"] == "failed" and "simulated crash" in state["stages"]["tts"]["error"]
    assert sum(1 for s in state["scenes"].values() if s["tts"]) == 3

    assert main(args) == 0                                   # continues from the failed scene
    assert len(calls) == 4 + 11
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stages"]["tts"]["status"] == "done" and state["stages"]["render"]["status"] == "pending"

    assert main(args) == 0                                   # nothing left to do
    assert len(calls) == 15
    shutil.rmtree(tmp_path / "cache")
    assert main(args) == 0                                   # state alone is enough to skip
    assert len(calls) == 15
    mtime = (tmp_path / "output" / VIDEO_ID / "audio" / "s001.mp3").stat().st_mtime_ns
    assert main(args + ["--force"]) == 0                     # --force rebuilds the stage
    assert len(calls) == 15 + 14
    assert (tmp_path / "output" / VIDEO_ID / "audio" / "s001.mp3").stat().st_mtime_ns > mtime
    assert main(["run", str(script), "--mock", "--stages", "bogus", "--root", str(tmp_path)]) == 2
    assert main(["run", str(script), "--mock", "--stages", "render", "--root", str(tmp_path)]) == 1  # images missing


def test_variant_long_burn_in_other_reframes_and_long_short_warning(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Paths the example does not exercise: burned-in long subtitles, center_crop, pan_scan, >60 s warning."""
    monkeypatch.setenv("PIPELINE_SUPERSAMPLE", "1")
    doc = small_doc(json.loads(EXAMPLE.read_text(encoding="utf-8")), ["s001", "s002", "s003"])
    doc["production"]["subtitles"]["burn_in"]["long"] = True
    doc["production"]["subtitles"]["style"] = "plain"
    doc["production"]["fps"] = 24
    doc["production"]["pacing"]["wpm"] = 110
    doc["localization"]["custom_tracks"] = []
    doc["shorts"] = [
        {"short_id": "sh-crop", "scene_ids": ["s001", "s002"], "reframe": "center_crop"},
        {"short_id": "sh-pan", "scene_ids": ["s002", "s003", "s001", "s002", "s003", "s001", "s002", "s003", "s001",
                                             "s002", "s003"], "reframe": "pan_scan", "end_card_text": "More soon"},
    ]
    script = tmp_path / "variant.json"
    script.write_text(json.dumps(doc), encoding="utf-8")
    assert main(["run", str(script), "--mock", "--root", str(tmp_path)]) == 0
    out = tmp_path / "output" / VIDEO_ID
    state = json.loads((tmp_path / "state" / f"{VIDEO_ID}.json").read_text(encoding="utf-8"))
    long_info = probe(out / "final" / "long.mp4")
    assert long_info.video_frames == state["timeline"]["total_frames"] and state["timeline"]["fps"] == 24
    assert abs((long_info.video_duration or 0) - (long_info.audio_duration or 0)) <= 1 / 24
    assert (out / "work" / "long.ass").is_file()
    for short_id in ("sh-crop", "sh-pan"):
        info = probe(out / "final" / "shorts" / f"{short_id}.mp4")
        assert (info.width, info.height) == (1080, 1920)
        assert info.video_frames == state["shorts"][short_id]["frames"]
        assert abs((info.video_duration or 0) - (info.audio_duration or 0)) <= 1 / 24
    assert state["shorts"]["sh-crop"]["hook"] is None
    assert state["shorts"]["sh-pan"]["duration_sec"] > 60
    log = (out / "pipeline.log").read_text(encoding="utf-8")
    assert "short sh-pan is" in log and "over the 60 s Shorts limit" in log
    assert os.path.getsize(out / "final" / "long.mp4") > 10_000
