"""Character timings -> words, and voiceover <-> subtitle alignment (incl. numbers)."""
from __future__ import annotations

import pytest

from pipeline.align import (Word, align_words, apply_pronunciation, chars_to_words, normalize_word,
                            split_display_words, words_for_display)


def fake_chars(text: str, per_char: float = 0.1) -> tuple[list[str], list[float], list[float]]:
    chars = list(text)
    return chars, [i * per_char for i in range(len(chars))], [(i + 1) * per_char for i in range(len(chars))]


def spoken(text: str, per_char: float = 0.1) -> list[Word]:
    return chars_to_words(*fake_chars(text, per_char))


# ---- characters -> words ------------------------------------------------------
def test_chars_to_words_basic() -> None:
    words = chars_to_words(*fake_chars("No hackers. No  crash."))
    assert [w.text for w in words] == ["No", "hackers.", "No", "crash."]
    assert words[0].start == pytest.approx(0.0) and words[0].end == pytest.approx(0.2)
    assert words[1].start == pytest.approx(0.3) and words[1].end == pytest.approx(1.1)
    # double space: the gap belongs to no word
    assert words[3].start == pytest.approx(1.6)


def test_chars_to_words_handles_edges_and_validates() -> None:
    assert chars_to_words([], [], []) == []
    words = chars_to_words([" ", "a", "b", " "], [0, 1, 2, 3], [1, 2, 3, 4])
    assert words == [Word("ab", 1.0, 3.0)]
    with pytest.raises(ValueError):
        chars_to_words(["a"], [0.0], [])


def test_chars_to_words_is_monotonic_on_bad_input() -> None:
    words = chars_to_words(["a", " ", "b"], [1.0, 1.1, 0.5], [1.2, 1.2, 0.4])
    assert words[1].start >= words[0].start and words[1].end >= words[1].start


# ---- normalisation --------------------------------------------------------------
@pytest.mark.parametrize("word,tokens", [
    ("$460", ["four", "hundred", "sixty", "dollars"]),
    ("45", ["forty", "five"]),
    ("forty-five", ["forty", "five"]),
    ("1st,", ["first"]),
    ("97%", ["ninety", "seven", "percent"]),
    ("9:30,", ["nine", "thirty"]),
    ("9:05", ["nine", "oh", "five"]),
    ("$12M", ["twelve", "million", "dollars"]),
    ("Street's", ["streets"]),
    ("didn’t.", ["didnt"]),
    ("1,000", ["one", "thousand"]),
    ("—", []),
])
def test_normalize_word(word: str, tokens: list[str]) -> None:
    assert normalize_word(word) == tokens


def test_year_reading_follows_what_was_spoken() -> None:
    assert normalize_word("2012", frozenset({"twenty", "twelve"})) == ["twenty", "twelve"]
    assert normalize_word("2012", frozenset({"two", "thousand", "twelve"})) == ["two", "thousand", "twelve"]


def test_split_display_words_attaches_loose_punctuation() -> None:
    assert split_display_words("one — two !") == ["one —", "two !"]


# ---- alignment ----------------------------------------------------------------
def test_direct_mapping_without_subtitle_text() -> None:
    text = "No hackers. No market crash."
    words = words_for_display(spoken(text), text, None, text)
    assert [w.text for w in words] == text.split()
    assert [(w.start, w.end) for w in words] == [(w.start, w.end) for w in spoken(text)]


def test_numbers_money_align_to_digits() -> None:
    voice = "it will cost it over four hundred sixty million dollars."
    subtitle = "it will cost it over $460 million."
    sp = spoken(voice)
    out = align_words(sp, split_display_words(subtitle))
    assert [w.text for w in out] == subtitle.split()
    by_text = {w.text: w for w in out}
    four = next(w for w in sp if w.text == "four")
    sixty = next(w for w in sp if w.text == "sixty")
    million = next(w for w in sp if w.text == "million")
    dollars = next(w for w in sp if w.text == "dollars.")
    assert by_text["$460"].start == pytest.approx(four.start)
    assert by_text["$460"].end == pytest.approx(sixty.end)
    assert by_text["million."].start == pytest.approx(million.start)
    assert by_text["million."].end == pytest.approx(dollars.end)   # "dollars" is shown as part of "$460 million"
    assert by_text["over"].end <= by_text["$460"].start


def test_ordinals_times_and_counts() -> None:
    voice = "August first, 2012. At nine thirty, it sent ninety-seven emails in forty-five minutes."
    subtitle = "August 1st, 2012. At 9:30, it sent 97 emails in 45 minutes."
    sp = spoken(voice)
    out = align_words(sp, split_display_words(subtitle))
    assert len(out) == len(subtitle.split())
    lookup = {w.text: w for w in out}
    src = {w.text: w for w in sp}
    assert lookup["1st,"].start == pytest.approx(src["first,"].start)
    assert lookup["9:30,"].start == pytest.approx(src["nine"].start)
    assert lookup["9:30,"].end == pytest.approx(src["thirty,"].end)
    assert lookup["97"].start == pytest.approx(src["ninety-seven"].start)
    assert lookup["45"].end == pytest.approx(src["forty-five"].end)
    assert lookup["minutes."].start == pytest.approx(src["minutes."].start)


def test_alignment_is_monotonic_and_complete_even_when_text_differs() -> None:
    sp = spoken("completely different words were spoken here today")
    display = split_display_words("Totally unrelated caption text")
    out = align_words(sp, display)
    assert [w.text for w in out] == display
    starts = [w.start for w in out]
    assert starts == sorted(starts)
    assert all(w.end >= w.start for w in out)
    assert out[0].start >= sp[0].start and out[-1].end <= sp[-1].end + 1e-9
    # proportional to length: the longest word gets the most time
    durations = {w.text: w.end - w.start for w in out}
    assert durations["unrelated"] > durations["text"]


def test_unmatched_middle_is_shared_proportionally() -> None:
    sp = spoken("alpha one two three omega")
    out = align_words(sp, ["alpha", "xx", "yyyy", "omega"])
    assert out[0].start == pytest.approx(sp[0].start) and out[3].start == pytest.approx(sp[4].start)
    gap_start, gap_end = sp[1].start, sp[3].end
    assert out[1].start == pytest.approx(gap_start)
    assert out[2].end == pytest.approx(gap_end)
    assert (out[2].end - out[2].start) == pytest.approx(2 * (out[1].end - out[1].start), rel=0.05)


def test_empty_alignment_falls_back_to_even_spread() -> None:
    out = align_words([], ["a", "bbb"], total_duration=4.0)
    assert out[0].start == 0.0 and out[-1].end == pytest.approx(4.0)


# ---- pronunciation --------------------------------------------------------------
PRON = [("SMARS", "smarz"), ("NYSE", "N Y S E")]


def test_pronunciation_only_changes_tts_text() -> None:
    text = "A router called SMARS, on the NYSE. SMARSX stays."
    tts_text = apply_pronunciation(text, PRON)
    assert tts_text == "A router called smarz, on the N Y S E. SMARSX stays."
    out = words_for_display(spoken(tts_text), text, None, tts_text, PRON)
    assert [w.text for w in out] == text.split()          # subtitles keep the original spelling
    sp = {w.text: w for w in spoken(tts_text)}
    smars = next(w for w in out if w.text == "SMARS,")
    nyse = next(w for w in out if w.text == "NYSE.")
    assert smars.start == pytest.approx(sp["smarz,"].start) and smars.end == pytest.approx(sp["smarz,"].end)
    assert nyse.start == pytest.approx(sp["N"].start) and nyse.end == pytest.approx(sp["E."].end)
