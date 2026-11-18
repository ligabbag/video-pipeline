"""Word timings: characters -> spoken words -> subtitle words.

ElevenLabs returns timings per *character*. This module
1. groups characters into spoken words (:func:`chars_to_words`);
2. maps spoken words onto subtitle words (:func:`align_words`). The subtitle
   says the same words in another notation ("four hundred sixty million dollars"
   vs "$460 million"), so both sides are normalised to plain lowercase tokens
   (numbers spelled out with num2words) and matched as sequences. Where the
   sequences disagree, time is shared out in proportion to token length.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from num2words import num2words

ALIGNER_VERSION = 2  # bump to invalidate stored alignments when the algorithm changes


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


# --------------------------------------------------------------------------- #
# characters -> words
# --------------------------------------------------------------------------- #
def chars_to_words(characters: Sequence[str], starts: Sequence[float], ends: Sequence[float]) -> list[Word]:
    """Group per-character timings into whitespace-delimited words.

    A word starts at its first character's start and ends at its last character's end.
    """
    if not (len(characters) == len(starts) == len(ends)):
        raise ValueError("characters / start times / end times must have equal length")
    words: list[Word] = []
    buf: list[str] = []
    w_start = w_end = 0.0
    for ch, st, en in zip(characters, starts, ends):
        if ch == "" or ch.isspace():
            if buf:
                words.append(Word("".join(buf), w_start, max(w_end, w_start)))
                buf = []
            continue
        if not buf:
            w_start = float(st)
        buf.append(ch)
        w_end = float(en)
    if buf:
        words.append(Word("".join(buf), w_start, max(w_end, w_start)))
    # Guard against non-monotonic input.
    fixed: list[Word] = []
    floor = 0.0
    for word in words:
        start = max(word.start, floor)
        end = max(word.end, start)
        fixed.append(Word(word.text, start, end))
        floor = start
    return fixed


# --------------------------------------------------------------------------- #
# pronunciation dictionary (TTS input only)
# --------------------------------------------------------------------------- #
def apply_pronunciation(text: str, entries: Iterable[tuple[str, str]]) -> str:
    """Replace whole-word occurrences of each ``term`` by ``say_as`` (case-sensitive).

    Used only for the text sent to TTS; subtitles keep the original spelling.
    """
    for term, say_as in sorted(entries, key=lambda e: len(e[0]), reverse=True):
        if not term:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        text = re.sub(pattern, lambda _m, s=say_as: s, text)
    return text


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
_TRANSLATE = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-", " ": " "})
_STRIP = " \t\r\n\"'()[]{}<>.,;:!?…*_~`-"
_CURRENCY = {"$": "dollars", "€": "euros", "£": "pounds", "¥": "yen"}
_MAGNITUDE = {"k": "thousand", "m": "million", "mm": "million", "b": "billion", "bn": "billion", "tn": "trillion"}
_CURRENCY_WORDS = frozenset(_CURRENCY.values())
_MAGNITUDE_WORDS = frozenset({"hundred", "thousand", "million", "billion", "trillion"})
_NUM = r"\d[\d,]*(?:\.\d+)?"
_RE_CURRENCY = re.compile(rf"^(?P<cur>[$€£¥])(?P<num>{_NUM})(?P<mag>[A-Za-z]{{1,2}})?$")
_RE_PERCENT = re.compile(rf"^(?P<num>{_NUM})%$")
_RE_TIME = re.compile(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$")
_RE_ORDINAL = re.compile(r"^(?P<num>\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
_RE_NUMBER = re.compile(rf"^(?P<num>{_NUM})(?P<suf>[A-Za-z]+)?$")


def _words_of(spelled: str) -> list[str]:
    """num2words output -> tokens ('four hundred and sixty-one' -> four hundred sixty one)."""
    return [t for t in re.findall(r"[a-z]+", spelled.lower()) if t != "and"]


def _number_tokens(num: str, hint: frozenset[str] | None) -> list[str]:
    plain = num.replace(",", "")
    try:
        if "." in plain:
            return _words_of(num2words(float(plain)))
        value = int(plain)
        cardinal = _words_of(num2words(value))
        if "," not in num and len(plain) == 4 and 1100 <= value <= 2099:
            # Could be read as a year ("twenty twelve") or a quantity; prefer what was spoken.
            year = _words_of(num2words(value, to="year"))
            if hint is None:
                return year
            score_year = sum(t in hint for t in year) / max(len(year), 1)
            score_card = sum(t in hint for t in cardinal) / max(len(cardinal), 1)
            return cardinal if score_card > score_year else year
        return cardinal
    except (ValueError, OverflowError, NotImplementedError):
        return [plain]


def normalize_word(word: str, hint: frozenset[str] | None = None) -> list[str]:
    """Normalise one written word to spoken-form tokens: lowercase, no punctuation, numbers spelled out.

    ``hint`` is the set of tokens known to be spoken; it disambiguates year vs quantity readings.
    """
    core = word.translate(_TRANSLATE).strip(_STRIP)
    if not core:
        return []
    if m := _RE_CURRENCY.match(core):
        tokens = _number_tokens(m["num"], hint)
        mag = (m["mag"] or "").lower()
        if mag:
            tokens.append(_MAGNITUDE.get(mag, mag))
        tokens.append(_CURRENCY[m["cur"]])
        return tokens
    if m := _RE_PERCENT.match(core):
        return _number_tokens(m["num"], hint) + ["percent"]
    if m := _RE_TIME.match(core):
        tokens = _number_tokens(str(int(m["h"])), None)
        minutes = int(m["m"])
        if 0 < minutes < 10:
            tokens += ["oh"] + _number_tokens(str(minutes), None)
        elif minutes >= 10:
            tokens += _number_tokens(str(minutes), None)
        return tokens
    if m := _RE_ORDINAL.match(core):
        try:
            return _words_of(num2words(int(m["num"]), to="ordinal"))
        except (ValueError, OverflowError, NotImplementedError):
            return [core.lower()]
    if m := _RE_NUMBER.match(core):
        tokens = _number_tokens(m["num"], hint)
        suffix = (m["suf"] or "").lower()
        if suffix:
            tokens.append(_MAGNITUDE.get(suffix, suffix))
        return tokens
    tokens: list[str] = []
    for part in re.findall(r"[a-z]+|\d+", core.lower().replace("'", "")):
        tokens += _number_tokens(part, hint) if part.isdigit() else [part]
    return tokens


def split_display_words(text: str) -> list[str]:
    """Split on-screen text into words; punctuation-only pieces stick to a neighbour."""
    words: list[str] = []
    pending = ""
    for piece in text.split():
        if any(ch.isalnum() for ch in piece):
            words.append(pending + piece)
            pending = ""
        elif words:
            words[-1] = f"{words[-1]} {piece}"
        else:
            pending += piece + " "
    return words


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Token:
    text: str
    start: float
    end: float


def _spread(weights: Sequence[int], start: float, end: float) -> list[tuple[float, float]]:
    """Split [start, end] into consecutive pieces proportional to ``weights``."""
    total = sum(weights) or 1
    span = max(end - start, 0.0)
    out: list[tuple[float, float]] = []
    cursor = start
    for weight in weights:
        piece = span * weight / total
        out.append((cursor, cursor + piece))
        cursor += piece
    return out


def distribute_evenly(words: Sequence[str], start: float, end: float) -> list[Word]:
    """Fallback when nothing can be aligned: time proportional to word length."""
    pieces = _spread([max(len(w), 1) for w in words], start, end)
    return [Word(w, s, e) for w, (s, e) in zip(words, pieces)]


def _spoken_tokens(spoken: Sequence[Word]) -> list[_Token]:
    tokens: list[_Token] = []
    for word in spoken:
        parts = normalize_word(word.text)
        if not parts:
            continue
        for part, (s, e) in zip(parts, _spread([len(p) for p in parts], word.start, word.end)):
            tokens.append(_Token(part, s, e))
    return tokens


def _display_tokens(word: str, pronunciation: dict[str, str], hint: frozenset[str]) -> list[str]:
    core = word.translate(_TRANSLATE).strip(_STRIP)
    if core in pronunciation:  # speak it the way TTS was told to
        tokens: list[str] = []
        for part in pronunciation[core].split():
            tokens += normalize_word(part, hint)
        return tokens
    return normalize_word(word, hint)


def align_words(spoken: Sequence[Word], display: Sequence[str],
                pronunciation: Iterable[tuple[str, str]] = (), total_duration: float | None = None) -> list[Word]:
    """Give every display (subtitle) word a start/end time taken from the spoken words.

    * exact token matches inherit the spoken token's time;
    * unmatched stretches share the time of the spoken tokens they stand in for
      (or the silent gap between their neighbours), in proportion to token length.
    The result is monotonic and has exactly ``len(display)`` words.
    """
    display = list(display)
    if not display:
        return []
    if not spoken:
        return distribute_evenly(display, 0.0, total_duration or 0.0)

    s_tokens = _spoken_tokens(spoken)
    if not s_tokens:
        return distribute_evenly(display, spoken[0].start, spoken[-1].end)
    hint = frozenset(t.text for t in s_tokens)
    pron = dict(pronunciation)

    d_tokens: list[str] = []
    owners: list[int] = []
    carry: str | None = None
    for index, word in enumerate(display):
        tokens = _display_tokens(word, pron, hint)
        if carry is not None:
            tokens = tokens + [carry]
            carry = None
        # "$460 million" is spoken "four hundred sixty million dollars": the currency
        # word belongs after the magnitude that follows the number.
        if (tokens and tokens[-1] in _CURRENCY_WORDS and index + 1 < len(display)
                and _RE_CURRENCY.match(word.translate(_TRANSLATE).strip(_STRIP))
                and len(nxt := normalize_word(display[index + 1])) == 1 and nxt[0] in _MAGNITUDE_WORDS):
            carry = tokens[-1]
            tokens = tokens[:-1]
        for token in tokens:
            d_tokens.append(token)
            owners.append(index)

    times: list[tuple[float, float] | None] = [None] * len(d_tokens)
    matcher = SequenceMatcher(None, [t.text for t in s_tokens], d_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[j1 + k] = (s_tokens[i1 + k].start, s_tokens[i1 + k].end)
        elif j2 > j1:  # "replace" or "insert": display tokens without an exact partner
            if i2 > i1:
                lo, hi = s_tokens[i1].start, s_tokens[i2 - 1].end
            else:
                lo = s_tokens[i1 - 1].end if i1 > 0 else s_tokens[0].start
                hi = s_tokens[i1].start if i1 < len(s_tokens) else lo
            for k, piece in enumerate(_spread([len(t) for t in d_tokens[j1:j2]], lo, max(hi, lo))):
                times[j1 + k] = piece

    spans: list[tuple[float, float] | None] = [None] * len(display)
    for token_index, owner in enumerate(owners):
        piece = times[token_index]
        if piece is None:
            continue
        current = spans[owner]
        spans[owner] = piece if current is None else (min(current[0], piece[0]), max(current[1], piece[1]))

    result: list[Word] = []
    floor = 0.0
    for index, word in enumerate(display):
        span = spans[index]
        if span is None:  # word with no speakable tokens: zero-length at the previous position
            span = (floor, floor)
        start = max(span[0], floor)
        end = max(span[1], start)
        result.append(Word(word, start, end))
        floor = start
    # A word may not run past the start of the next one.
    clipped: list[Word] = []
    for index, word in enumerate(result):
        end = min(word.end, result[index + 1].start) if index + 1 < len(result) else word.end
        clipped.append(Word(word.text, word.start, max(end, word.start)))
    return clipped


def words_for_display(spoken: Sequence[Word], voiceover_text: str, subtitle_text: str | None,
                      tts_text: str, pronunciation: Iterable[tuple[str, str]] = (),
                      total_duration: float | None = None) -> list[Word]:
    """Subtitle word timings for one line.

    Without ``subtitle_text`` (and with no pronunciation rewrite) the spoken and
    shown words are the same list, so times are copied one-to-one; otherwise the
    token aligner is used.
    """
    display = split_display_words(subtitle_text if subtitle_text is not None else voiceover_text)
    if subtitle_text is None and tts_text == voiceover_text and len(display) == len(spoken):
        return [Word(text, w.start, w.end) for text, w in zip(display, spoken)]
    return align_words(spoken, display, pronunciation, total_duration)
