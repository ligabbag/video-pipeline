#!/usr/bin/env python3
"""Validate a video script against the Opus -> pipeline contract.

Usage:
    python validate_script.py path/to/script.json [--schema video_script.schema.json]

Exit code 0 = no errors (warnings allowed), 1 = errors.
Requires: pip install jsonschema
"""
import argparse
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

WORD_RE = re.compile(r"[A-Za-z0-9$%'’-]+")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def semantic_checks(doc: dict) -> tuple[list[str], list[str]]:
    errors, warnings = [], []

    scenes = [sc for sec in doc["sections"] for sc in sec["scenes"]]
    scene_ids = [sc["scene_id"] for sc in scenes]
    section_ids = [sec["section_id"] for sec in doc["sections"]]
    source_ids = {s["source_id"] for s in doc["editorial"]["sources"]}
    entity_ids = {e["entity_id"] for e in doc["production"].get("entities", [])}

    for label, ids in (("scene_id", scene_ids), ("section_id", section_ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            errors.append(f"duplicate {label}: {sorted(dupes)}")

    # Pacing: word-count window derived from wpm and scene length limits.
    pacing = doc["production"]["pacing"]
    wps = pacing["wpm"] / 60
    min_words = int(pacing["scene_min_sec"] * wps)
    max_words = int(round(pacing["scene_max_sec"] * wps))
    total_words = 0

    seen = set()
    for sc in scenes:
        sid = sc["scene_id"]
        n = word_count(sc["voiceover_text"])
        total_words += n
        if n < min_words or n > max_words:
            warnings.append(f"{sid}: {n} words, expected {min_words}-{max_words} "
                            f"for {pacing['scene_min_sec']}-{pacing['scene_max_sec']} s")

        for ref in sc["source_refs"]:
            if ref not in source_ids:
                errors.append(f"{sid}: unknown source_ref '{ref}'")
        if not sc["source_refs"]:
            warnings.append(f"{sid}: no source_refs (fine only for pure narration)")

        vis = sc["visual"]
        for ent in vis.get("entity_refs", []):
            if ent not in entity_ids:
                errors.append(f"{sid}: unknown entity_ref '{ent}'")
        if vis["source"] == "reuse":
            target = vis["reuse_scene_id"]
            if target not in seen:
                errors.append(f"{sid}: reuse_scene_id '{target}' must point to an earlier scene")
            elif next(s for s in scenes if s["scene_id"] == target)["visual"]["source"] == "reuse":
                errors.append(f"{sid}: reuse chain ('{target}' is itself a reuse)")

        sub = sc.get("subtitle_text")
        if sub is not None and abs(word_count(sub) - n) > 6:
            warnings.append(f"{sid}: subtitle_text and voiceover_text differ a lot in length")
        shown = sub if sub is not None else sc["voiceover_text"]
        for w in sc.get("emphasis_words", []):
            if w.lower() not in shown.lower():
                errors.append(f"{sid}: emphasis word '{w}' not found in on-screen text")
        seen.add(sid)

    est_sec = total_words / wps
    target = doc["editorial"]["target_duration_sec"]
    if abs(est_sec - target) > 0.15 * target:
        warnings.append(f"estimated duration {est_sec:.0f} s vs target {target} s "
                        f"({total_words} words at {pacing['wpm']} wpm)")

    for sh in doc.get("shorts", []):
        for sid in sh["scene_ids"]:
            if sid not in scene_ids:
                errors.append(f"short {sh['short_id']}: unknown scene_id '{sid}'")

    tg = doc["distribution"]["telegram_brief"]
    for sid in tg.get("image_scene_ids", []):
        if sid not in scene_ids:
            errors.append(f"telegram_brief: unknown image_scene_id '{sid}'")
    for f in tg["key_facts"]:
        if f["source_ref"] not in source_ids:
            errors.append(f"telegram_brief: unknown source_ref '{f['source_ref']}'")

    # Language-neutral visuals: diagram text must not contain English phrases.
    if doc["localization"]["on_screen_text_policy"] == "language_neutral":
        word = re.compile(r"[A-Za-z]{2,}")
        allowed_keys = {"icon", "template", "highlight_mark"}
        for sc in scenes:
            diag = sc["visual"].get("diagram")
            if not diag:
                continue
            for key, val in diag["params"].items():
                if key in allowed_keys:
                    continue
                if isinstance(val, str) and len(word.findall(val)) >= 2:
                    warnings.append(f"{sc['scene_id']}: diagram param '{key}' looks like an English phrase "
                                    f"('{val}'); it will show under every dubbed track")

    if doc["production"]["subtitles"]["burn_in"]["long"] and doc["localization"]["custom_tracks"]:
        warnings.append("burn_in.long=true with custom dub tracks: English subtitles will show under other languages")

    tags_len = sum(len(t) for t in doc["seo"]["youtube_tags"]) + max(len(doc["seo"]["youtube_tags"]) - 1, 0)
    if tags_len > 500:
        errors.append(f"youtube_tags total length {tags_len} > 500")

    if doc["status"] == "approved" and not doc["qa"]["facts_verified"]:
        errors.append("status=approved but qa.facts_verified=false")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--schema", default=str(Path(__file__).with_name("video_script.schema.json")))
    args = ap.parse_args()

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    doc = json.loads(Path(args.script).read_text(encoding="utf-8"))

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    for e in schema_errors:
        loc = "/".join(str(p) for p in e.path) or "<root>"
        print(f"SCHEMA ERROR  {loc}: {e.message}")
    if schema_errors:
        return 1

    errors, warnings = semantic_checks(doc)
    for w in warnings:
        print(f"WARNING       {w}")
    for e in errors:
        print(f"ERROR         {e}")
    print(f"\n{'FAIL' if errors else 'OK'}: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
