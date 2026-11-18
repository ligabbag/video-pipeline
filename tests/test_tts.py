"""ElevenLabs provider and the TTS stage, with the HTTP layer faked."""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from pipeline import tts as tts_mod
from pipeline.ffmpeg import run_ffmpeg
from pipeline.logging_setup import RedactingFormatter
from pipeline.tts import ElevenLabsTTS, MockTTS, TTSError, TTSRequest, run_tts_stage
from tests.conftest import FakeResponse, FakeSession, approved, make_context, small_doc

KEY = "sk_test_SECRET_elevenlabs_0123456789"


@pytest.fixture(scope="module")
def silent_mp3() -> bytes:
    proc = run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "5", "-c:a", "libmp3lame",
                       "-b:a", "64k", "-f", "mp3", "-"], capture_stdout=True)
    return proc.stdout


def ok_response(text: str, audio: bytes, seconds: float = 5.0) -> FakeResponse:
    chars = list(text)
    step = seconds / len(chars)
    alignment = {"characters": chars,
                 "character_start_times_seconds": [i * step for i in range(len(chars))],
                 "character_end_times_seconds": [(i + 1) * step for i in range(len(chars))]}
    return FakeResponse(200, {"audio_base64": base64.b64encode(audio).decode(), "alignment": alignment,
                              "normalized_alignment": alignment})


def test_request_shape_and_result(silent_mp3: bytes) -> None:
    session = FakeSession(lambda m, u, kw: ok_response(kw["json"]["text"], silent_mp3))
    provider = ElevenLabsTTS(KEY, session=session, sleep=lambda s: None)
    result = provider.synthesize(TTSRequest("Hello big world.", "VOICE", "eleven_multilingual_v2",
                                            {"stability": 0.45}, "Before.", "After."))
    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "https://api.elevenlabs.io/v1/text-to-speech/VOICE/with-timestamps"
    assert kwargs["params"] == {"output_format": "mp3_44100_128"}
    assert kwargs["headers"]["xi-api-key"] == KEY
    assert kwargs["json"] == {"text": "Hello big world.", "model_id": "eleven_multilingual_v2",
                              "voice_settings": {"stability": 0.45}, "previous_text": "Before.",
                              "next_text": "After."}
    assert result.audio == silent_mp3
    assert "".join(result.characters) == "Hello big world."
    assert result.billed_characters == len("Hello big world.")


def test_retries_on_429_and_5xx_with_backoff(silent_mp3: bytes) -> None:
    replies = [FakeResponse(429, text="slow down", headers={"Retry-After": "7"}), FakeResponse(503, text="oops")]
    sleeps: list[float] = []

    def handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        return replies.pop(0) if replies else ok_response(kwargs["json"]["text"], silent_mp3)

    session = FakeSession(handler)
    provider = ElevenLabsTTS(KEY, session=session, sleep=sleeps.append)
    result = provider.synthesize(TTSRequest("Retry me please.", "V", "m"))
    assert len(session.calls) == 3 and result.audio
    assert len(sleeps) == 2
    assert sleeps[0] >= 7.0            # Retry-After is honoured
    assert 0 < sleeps[1] <= 31.0       # exponential backoff with jitter, capped


def test_gives_up_after_max_attempts_and_does_not_retry_4xx() -> None:
    session = FakeSession(lambda m, u, kw: FakeResponse(500, text="down"))
    provider = ElevenLabsTTS(KEY, session=session, sleep=lambda s: None, attempts=3)
    with pytest.raises(Exception) as err:
        provider.synthesize(TTSRequest("x" * 12, "V", "m"))
    assert len(session.calls) == 3 and KEY not in str(err.value)

    session = FakeSession(lambda m, u, kw: FakeResponse(401, text='{"detail":"invalid api key"}'))
    provider = ElevenLabsTTS(KEY, session=session, sleep=lambda s: None)
    with pytest.raises(TTSError) as err2:
        provider.synthesize(TTSRequest("x" * 12, "V", "m"))
    assert len(session.calls) == 1 and "401" in str(err2.value) and KEY not in str(err2.value)


def test_falls_back_when_model_rejects_stitching(silent_mp3: bytes) -> None:
    def handler(method: str, url: str, kwargs: dict[str, Any]) -> FakeResponse:
        if "previous_text" in kwargs["json"]:
            return FakeResponse(400, text='{"detail":"previous_text is not supported by this model"}')
        return ok_response(kwargs["json"]["text"], silent_mp3)

    session = FakeSession(handler)
    provider = ElevenLabsTTS(KEY, session=session, sleep=lambda s: None)
    result = provider.synthesize(TTSRequest("Some line here.", "V", "eleven_v3", {}, "Prev.", None))
    assert len(session.calls) == 2 and "previous_text" not in session.calls[1][2]["json"] and result.audio


def test_missing_key_is_a_clear_error() -> None:
    with pytest.raises(TTSError, match="ELEVENLABS_API_KEY"):
        ElevenLabsTTS("")


def test_mock_tts_duration_follows_wpm() -> None:
    result = MockTTS(150).synthesize(TTSRequest("one two three four five", "V", "m"))
    assert result.ends[-1] == pytest.approx(5 / (150 / 60), abs=0.01)   # 5 words at 150 wpm = 2 s
    assert len(result.characters) == len("one two three four five") and result.billed_characters == 0


def test_stage_real_provider_end_to_end(tmp_path: Path, example_doc: dict[str, Any], silent_mp3: bytes,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Pronunciation only in TTS text, neighbours as context, cache, costs, timeline, no key in logs."""
    doc = approved(small_doc(example_doc, ["s001", "s002", "s003"]))
    doc["sections"][0]["scenes"][2]["voiceover_text"] = "No hackers. Just SMARS and one forgotten piece of code."
    session = FakeSession(lambda m, u, kw: ok_response(kw["json"]["text"], silent_mp3))
    ctx = make_context(tmp_path, doc, mock=False, env={"ELEVENLABS_API_KEY": KEY})
    monkeypatch.setattr(tts_mod, "make_tts_provider",
                        lambda c: ElevenLabsTTS(KEY, session=session, sleep=lambda s: None))
    run_tts_stage(ctx)

    bodies = [call[2]["json"] for call in session.calls]
    assert len(bodies) == 3
    assert bodies[2]["text"] == "No hackers. Just smarz and one forgotten piece of code."
    assert "previous_text" not in bodies[0] and bodies[0]["next_text"] == bodies[1]["text"]
    assert bodies[1]["previous_text"] == bodies[0]["text"] and bodies[1]["next_text"] == bodies[2]["text"]
    assert "next_text" not in bodies[2]
    assert bodies[0]["voice_settings"] == {"stability": 0.45, "similarity_boost": 0.8, "style": 0.15}

    s3 = ctx.state.scenes["s003"]
    assert s3.tts is not None and s3.timing is not None
    assert [w.text for w in s3.tts.words][3] == "SMARS"                  # subtitles keep the spelling
    assert [w.text for w in s3.tts.spoken_words][3] == "smarz"
    s1 = ctx.state.scenes["s001"]
    assert s1.tts is not None and [w.text for w in s1.tts.words][1] == "1st,"
    assert (tmp_path / s1.tts.audio_path).is_file() and s1.tts.duration_sec == pytest.approx(5.0, abs=0.15)
    assert ctx.state.costs.tts_requests == 3
    assert ctx.state.costs.tts_characters == sum(len(b["text"]) for b in bodies)
    # scene slots: audio + pause, contiguous, frame-aligned
    t1, t2 = ctx.state.scenes["s001"].timing, ctx.state.scenes["s002"].timing
    assert t1 is not None and t2 is not None and t1.end_frame == t2.start_frame
    assert t1.end_sec - t1.start_sec >= s1.tts.duration_sec + 0.15 - 1e-6

    # second run: nothing is requested again; after losing the state, the cache still prevents billing
    run_tts_stage(ctx)
    assert len(session.calls) == 3
    ctx.state.scenes.clear()
    run_tts_stage(ctx)
    assert len(session.calls) == 3 and ctx.state.costs.tts_cache_hits == 3

    # changing one scene re-requests only that scene (neighbours are not part of the cache key)
    doc["sections"][0]["scenes"][1]["voiceover_text"] = "In forty-five minutes, everything changes for good."
    doc["sections"][0]["scenes"][1]["subtitle_text"] = "In 45 minutes, everything changes for good."
    doc["sections"][0]["scenes"][1]["emphasis_words"] = []
    ctx2 = make_context(tmp_path, doc, mock=False, env={"ELEVENLABS_API_KEY": KEY})
    run_tts_stage(ctx2)
    assert len(session.calls) == 4

    for handler in logging.getLogger("pipeline").handlers:
        handler.flush()
    assert KEY not in ctx.paths.log_file.read_text(encoding="utf-8")
    assert KEY not in (tmp_path / "state" / f"{doc['video_id']}.json").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "script.json").read_text(encoding="utf-8")) == doc    # script untouched


def test_log_formatter_redacts_secrets() -> None:
    formatter = RedactingFormatter("%(message)s", None, [KEY])
    record = logging.LogRecord("pipeline", logging.ERROR, __file__, 1, "header xi-api-key=%s failed", (KEY,), None)
    assert formatter.format(record) == "header xi-api-key=*** failed"
