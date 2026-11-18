"""``metadata.json``: title, description with {{CHAPTERS}} / {{SOURCES}} filled in, tags and flags."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from pipeline.contract import Script

MIN_CHAPTER_SEC = 10.0


@dataclass(frozen=True)
class Chapter:
    title: str
    start_sec: float


def compute_chapters(sections: Sequence[tuple[str, float]], total_sec: float,
                     min_len: float = MIN_CHAPTER_SEC) -> list[Chapter]:
    """YouTube chapters from section start times.

    The first chapter starts at 0:00. Any chapter shorter than ``min_len`` is merged
    into its neighbour: into the previous one, or — for the very first chapter —
    it absorbs the next one. The surviving chapter keeps its own title.
    """
    chapters = [Chapter(title, max(0.0, start)) for title, start in sections]
    if not chapters:
        return []
    chapters[0] = Chapter(chapters[0].title, 0.0)
    while len(chapters) > 1:
        lengths = [(chapters[i + 1].start_sec if i + 1 < len(chapters) else total_sec) - chapters[i].start_sec
                   for i in range(len(chapters))]
        short = next((i for i, length in enumerate(lengths) if length < min_len), None)
        if short is None:
            break
        del chapters[1 if short == 0 else short]
    return chapters


def format_timestamp(seconds: float, with_hours: bool) -> str:
    total = int(seconds)  # floor: a chapter link must not land after its first word
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if with_hours else f"{m}:{s:02d}"


def chapters_block(chapters: Sequence[Chapter], total_sec: float) -> str:
    with_hours = total_sec >= 3600
    return "\n".join(f"{format_timestamp(c.start_sec, with_hours)} {c.title}" for c in chapters)


def sources_block(script: Script) -> str:
    return "\n".join(f"{source.title}: {source.url}" for source in script.editorial.sources)


def build_metadata(script: Script, section_starts: Sequence[tuple[str, float]], total_sec: float) -> dict[str, Any]:
    chapters = compute_chapters(section_starts, total_sec)
    description = (script.seo.description
                   .replace("{{CHAPTERS}}", chapters_block(chapters, total_sec))
                   .replace("{{SOURCES}}", sources_block(script)))
    return {
        "video_id": script.video_id,
        "title": script.seo.title_seo,
        "title_variants": list(script.seo.title_variants),
        "description": description,
        "tags": list(script.seo.youtube_tags),
        "category_id": script.seo.category_id,
        "made_for_kids": script.seo.made_for_kids,
        "contains_synthetic_media": script.seo.contains_synthetic_media,
        "language": script.language,
        "duration_sec": round(total_sec, 3),
        "chapters": [{"title": c.title, "start_sec": round(c.start_sec, 3),
                      "timestamp": format_timestamp(c.start_sec, total_sec >= 3600)} for c in chapters],
    }
