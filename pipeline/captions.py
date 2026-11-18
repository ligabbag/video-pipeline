"""Subtitles: SRT caption track (always exported) and ASS for burned-in subtitles.

SRT: at most 2 lines of 42 characters per cue, timed from the word timings.
ASS: chunks of ``max_words_per_chunk`` words, the word being spoken is shown in
``highlight_color``; ``emphasis_words`` stay highlighted all the time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from pipeline.align import Word

SRT_MAX_CHARS = 42
SRT_MAX_LINES = 2
_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")
_CLAUSE_END = re.compile(r"[,;:][\"')\]]*$")


# --------------------------------------------------------------------------- #
# SRT
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def wrap_greedy(words: Sequence[str], max_chars: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def balance_two_lines(words: Sequence[str], max_chars: int) -> tuple[str, ...]:
    """One line if it fits, else the most even two-line split (preferring a break after punctuation)."""
    text = " ".join(words)
    if len(text) <= max_chars or len(words) < 2:
        return (text,)
    best: tuple[float, tuple[str, str]] | None = None
    for cut in range(1, len(words)):
        first, second = " ".join(words[:cut]), " ".join(words[cut:])
        if len(first) > max_chars or len(second) > max_chars:
            continue
        score = abs(len(first) - len(second))
        if _SENTENCE_END.search(words[cut - 1]) or _CLAUSE_END.search(words[cut - 1]):
            score -= 8  # a natural pause beats perfect symmetry
        if best is None or score < best[0]:
            best = (score, (first, second))
    return best[1] if best else tuple(wrap_greedy(words, max_chars)[:2])


def build_cues(words: Sequence[Word], max_chars: int = SRT_MAX_CHARS, max_lines: int = SRT_MAX_LINES,
               limit: float | None = None) -> list[Cue]:
    """Group timed words (absolute seconds) into cues that fit ``max_lines`` x ``max_chars``.

    A cue also ends at a sentence boundary once it holds at least three words.
    """
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        candidate = [w.text for w in current] + [word.text]
        if current and len(wrap_greedy(candidate, max_chars)) > max_lines:
            # Full. Prefer to end the cue at the last comma/clause break rather than mid-phrase,
            # as long as the remainder still fits together with the new word.
            cut = next((k for k in range(len(current) - 2, 0, -1)
                        if _CLAUSE_END.search(current[k].text) or _SENTENCE_END.search(current[k].text)), None)
            rest = [w.text for w in current[cut + 1:]] + [word.text] if cut is not None else []
            if cut is not None and len(wrap_greedy(rest, max_chars)) <= max_lines:
                groups.append(current[:cut + 1])
                current = current[cut + 1:]
            else:
                groups.append(current)
                current = []
        current.append(word)
        if _SENTENCE_END.search(word.text) and len(current) >= 3:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    cues: list[Cue] = []
    for index, group in enumerate(groups):
        start = group[0].start
        next_start = groups[index + 1][0].start if index + 1 < len(groups) else (limit if limit is not None else None)
        end = max(group[-1].end, start + 0.7)        # readable minimum
        end += 0.25                                    # small linger after the last word
        if next_start is not None:
            end = min(end, next_start - 0.04) if next_start - 0.04 > start else next_start
        end = max(end, start + 0.05)
        cues.append(Cue(start, end, balance_two_lines([w.text for w in group], max_chars)))
    return cues


def srt_timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(cues: Sequence[Cue]) -> str:
    blocks = [f"{i}\n{srt_timestamp(c.start)} --> {srt_timestamp(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# emphasis
# --------------------------------------------------------------------------- #
def _plain(text: str) -> str:
    return re.sub(r"[^\w$%€£]+", "", text.lower().replace("’", "'").replace("'", ""))


def mark_emphasis(words: Sequence[str], phrases: Sequence[str]) -> list[bool]:
    """Flags for words that belong to an emphasis word/phrase (as written in the subtitles)."""
    flags = [False] * len(words)
    plain = [_plain(w) for w in words]
    for phrase in phrases:
        target = [_plain(p) for p in phrase.split() if _plain(p)]
        if not target:
            continue
        for start in range(len(plain) - len(target) + 1):
            if plain[start:start + len(target)] == target:
                for offset in range(len(target)):
                    flags[start + offset] = True
    return flags


# --------------------------------------------------------------------------- #
# ASS
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SubWord:
    text: str
    start: float
    end: float
    emphasis: bool = False


@dataclass(frozen=True)
class AssStyle:
    font: str
    base_color: str          # "#RRGGBB"
    highlight_color: str
    position: str            # bottom_center | center
    dynamic: bool            # highlight the word being spoken
    max_words_per_chunk: int


@dataclass(frozen=True)
class EndCard:
    text: str
    start: float
    end: float


def ass_color(hex_color: str) -> str:
    """#RRGGBB -> ASS &HBBGGRR&."""
    value = hex_color.lstrip("#")
    return f"&H{value[4:6]}{value[2:4]}{value[0:2]}&".upper()


def ass_timestamp(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360_000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def chunk_words(words: Sequence[SubWord], max_words: int, max_gap: float = 0.8) -> list[list[SubWord]]:
    """Split into chunks of at most ``max_words``; sentences and long pauses also end a chunk."""
    chunks: list[list[SubWord]] = []
    current: list[SubWord] = []
    for word in words:
        if current and (len(current) >= max_words or word.start - current[-1].end > max_gap):
            chunks.append(current)
            current = []
        current.append(word)
        if _SENTENCE_END.search(word.text):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def render_ass(lines: Sequence[Sequence[SubWord]], style: AssStyle, frame: tuple[int, int],
               end_card: EndCard | None = None, total_sec: float | None = None) -> str:
    """ASS document. ``lines`` are independent runs of words (one per scene); chunks never span them."""
    width, height = frame
    vertical = height > width
    font_size = round(height * (0.05 if vertical else 0.06))
    margin_v = round(height * (0.2 if vertical else 0.075))
    alignment = 5 if style.position == "center" else 2
    bold = 0 if re.search(r"bold|black|heavy", style.font, re.I) else -1
    outline = max(2, round(font_size * 0.09))
    base, high = ass_color(style.base_color), ass_color(style.highlight_color)
    header = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Default,{style.font},{font_size},{base.replace('&H', '&H00')},{base.replace('&H', '&H00')},"
        f"&H00000000&,&H96000000&,{bold},0,0,0,100,100,0,0,1,{outline},{max(1, outline // 3)},{alignment},"
        f"{round(width * 0.06)},{round(width * 0.06)},{margin_v},1",
        f"Style: EndCard,{style.font},{round(font_size * 0.9)},{high.replace('&H', '&H00')},"
        f"{high.replace('&H', '&H00')},&H00000000&,&H96000000&,{bold},0,0,0,100,100,0,0,1,{outline},"
        f"{max(1, outline // 3)},8,{round(width * 0.06)},{round(width * 0.06)},{round(height * 0.14)},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []
    chunks: list[list[SubWord]] = []
    for line in lines:
        chunks.extend(chunk_words(line, style.max_words_per_chunk))
    for index, chunk in enumerate(chunks):
        next_start = chunks[index + 1][0].start if index + 1 < len(chunks) else total_sec
        chunk_end = chunk[-1].end + 0.2
        if next_start is not None:
            chunk_end = min(chunk_end, next_start)
        chunk_end = max(chunk_end, chunk[-1].start + 0.05)
        # One event per spoken word: same text, a different word lit up.
        steps = range(len(chunk)) if style.dynamic else range(1)
        for active in steps:
            start = chunk[active].start if style.dynamic else chunk[0].start
            end = chunk[active + 1].start if style.dynamic and active + 1 < len(chunk) else chunk_end
            if end <= start:
                continue
            parts: list[str] = []
            for position, word in enumerate(chunk):
                lit = word.emphasis or (style.dynamic and position == active)
                colour = high if lit else base
                parts.append(f"{{\\c{colour}}}{_escape(word.text)}")
            events.append(f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},Default,,0,0,0,,{' '.join(parts)}")
    if end_card and end_card.text.strip() and end_card.end > end_card.start:
        events.append(f"Dialogue: 1,{ass_timestamp(end_card.start)},{ass_timestamp(end_card.end)},EndCard,,0,0,0,,"
                      f"{{\\fad(150,0)}}{_escape(end_card.text)}")
    return "\n".join(header + events) + "\n"
