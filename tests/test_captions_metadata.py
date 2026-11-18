"""SRT splitting, ASS generation, chapters and timeline maths."""
from __future__ import annotations

import pytest

from pipeline.align import Word
from pipeline.captions import (AssStyle, EndCard, SubWord, ass_color, build_cues, chunk_words, mark_emphasis,
                               render_ass, render_srt, srt_timestamp)
from pipeline.metadata import Chapter, chapters_block, compute_chapters
from pipeline.timeline import LineSpec, lay_out, samples_per_frame


def timed(text: str, start: float = 0.0, per_word: float = 0.4) -> list[Word]:
    return [Word(w, start + i * per_word, start + (i + 1) * per_word - 0.05) for i, w in enumerate(text.split())]


# ---- SRT ------------------------------------------------------------------------
LONG = ("Knight Capital was a market maker, standing between ordinary investors and the stock exchanges. "
        "At its heart sat a router called SMARS, which split big orders into smaller ones. "
        "Supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis words still fit somehow.")


def test_srt_cues_respect_line_limits_and_keep_every_word() -> None:
    words = timed(LONG)
    cues = build_cues(words)
    assert " ".join(" ".join(c.lines) for c in cues).split() == LONG.split()
    for cue in cues:
        assert 1 <= len(cue.lines) <= 2
        for line in cue.lines:
            assert len(line) <= 42 or " " not in line   # only a single over-long word may exceed the limit
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))
    assert all(c.end > c.start for c in cues)


def test_srt_breaks_at_sentences_and_clauses() -> None:
    cues = build_cues(timed("No hackers. No market crash. Just one forgotten piece of code."))
    assert [" ".join(c.lines) for c in cues] == ["No hackers. No market crash.", "Just one forgotten piece of code."]
    cues = build_cues(timed("Knight Capital was a market maker, standing between ordinary investors and the stock exchanges."))
    assert " ".join(cues[0].lines) == "Knight Capital was a market maker,"


def test_srt_timing_comes_from_words() -> None:
    words = timed("One two three.", start=10.0)
    cue = build_cues(words)[0]
    assert cue.start == pytest.approx(10.0)
    assert cue.end >= words[-1].end


def test_srt_format() -> None:
    assert srt_timestamp(3725.5) == "01:02:05,500"
    text = render_srt(build_cues(timed("Hello there world.")))
    assert text.startswith("1\n00:00:00,000 --> ")
    assert text.rstrip().endswith("Hello there world.")


# ---- emphasis / ASS -----------------------------------------------------------------
def test_mark_emphasis_matches_words_and_phrases() -> None:
    words = "it will cost over $460 million. The eighth server woke up.".split()
    flags = mark_emphasis(words, ["$460 million", "woke up", "EIGHTH"])
    assert [w for w, f in zip(words, flags) if f] == ["$460", "million.", "eighth", "woke", "up."]


def test_chunks_have_at_most_n_words_and_end_at_sentences() -> None:
    words = [SubWord(w.text, w.start, w.end) for w in timed("One two three four five. Six seven.")]
    chunks = chunk_words(words, 3)
    assert [[w.text for w in c] for c in chunks] == [["One", "two", "three"], ["four", "five."], ["Six", "seven."]]


def test_ass_highlights_current_word_and_emphasis() -> None:
    style = AssStyle("Montserrat ExtraBold", "#FFFFFF", "#FFD400", "bottom_center", True, 3)
    words = [SubWord("big", 0.0, 0.4), SubWord("server", 0.5, 0.9, emphasis=True), SubWord("down.", 1.0, 1.4)]
    ass = render_ass([words], style, (1080, 1920), EndCard("Full story", 1.5, 3.0), total_sec=3.0)
    assert ass_color("#FFD400") == "&H00D4FF&"
    events = [line for line in ass.splitlines() if line.startswith("Dialogue: 0")]
    assert len(events) == 3                                   # one per spoken word
    assert events[0].endswith(r"{\c&H00D4FF&}big {\c&H00D4FF&}server {\c&HFFFFFF&}down.")
    assert events[2].endswith(r"{\c&HFFFFFF&}big {\c&H00D4FF&}server {\c&H00D4FF&}down.")
    assert "PlayResX: 1080" in ass and "Montserrat ExtraBold" in ass
    assert any("EndCard" in line and "Full story" in line for line in ass.splitlines() if line.startswith("Dialogue"))
    plain = render_ass([words], AssStyle("Arial", "#FFFFFF", "#FFD400", "center", False, 3), (1920, 1080))
    assert len([line for line in plain.splitlines() if line.startswith("Dialogue")]) == 1


# ---- chapters -----------------------------------------------------------------------
def test_chapters_start_at_zero_and_keep_long_sections() -> None:
    chapters = compute_chapters([("Intro", 0.0), ("Machine", 24.4), ("Deploy", 46.2)], 80.0)
    assert chapters == [Chapter("Intro", 0.0), Chapter("Machine", 24.4), Chapter("Deploy", 46.2)]
    assert chapters_block(chapters, 80.0) == "0:00 Intro\n0:24 Machine\n0:46 Deploy"


def test_short_chapters_merge_with_neighbours() -> None:
    # B lasts only 6 s: it merges into the previous chapter (A now runs until C)
    assert compute_chapters([("A", 0), ("B", 20), ("C", 26), ("D", 40)], 60) == [
        Chapter("A", 0.0), Chapter("C", 26), Chapter("D", 40)]
    # a short first chapter absorbs the next one and still starts at 0:00
    assert compute_chapters([("A", 0), ("B", 5), ("C", 30)], 60) == [Chapter("A", 0.0), Chapter("C", 30)]
    # a short last chapter merges backwards
    assert compute_chapters([("A", 0), ("B", 30), ("C", 55)], 60) == [Chapter("A", 0.0), Chapter("B", 30)]
    # cascades until everything is long enough
    assert compute_chapters([("A", 0), ("B", 4), ("C", 8), ("D", 12)], 30) == [Chapter("A", 0.0), Chapter("D", 12)]
    assert compute_chapters([("A", 3.0)], 5.0) == [Chapter("A", 0.0)]


def test_chapter_timestamps_use_hours_for_long_videos() -> None:
    block = chapters_block([Chapter("A", 0.0), Chapter("B", 3725.9)], 4000.0)
    assert block == "0:00:00 A\n1:02:05 B"


# ---- timeline -----------------------------------------------------------------------
@pytest.mark.parametrize("fps", [24, 25, 30])
def test_timeline_is_frame_aligned_and_keeps_pauses(fps: int) -> None:
    lines = [LineSpec("a", 5.6441, 0.15), LineSpec("b", 6.8441, 0.6), LineSpec("c", 4.4401, 0.6)]
    slots = lay_out(lines, fps)
    assert slots[0].start_frame == 0
    for prev, cur in zip(slots, slots[1:]):
        assert cur.start_frame == prev.end_frame                       # no gaps, no overlaps
    for line, slot in zip(lines, slots):
        room = slot.end_sec - slot.start_sec
        assert room >= line.audio_sec + line.pause_sec - 1e-6          # the pause is never cut short
        assert room < line.audio_sec + line.pause_sec + 1.0 / fps      # ...and never grows by a whole frame
        assert slot.start_sample == slot.start_frame * samples_per_frame(fps)
    assert 48_000 % fps == 0
