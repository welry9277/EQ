#!/usr/bin/env -S uv run python
"""Merge per-query-type EQ JSONL files into one record per clip (audio_id).

Reads eq_*.jsonl in a directory (compact or pretty-printed blocks), joins on audio_id,
and writes JSONL with original_captions + source_caption + all generated_queries together.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Process full_caption first so merged original_captions is the full set (other types store middle-only).
QUERY_TYPE_FILES = [
    ("full_caption", "eq_full_caption.jsonl"),
    ("key_phrase", "eq_key_phrase.jsonl"),
    ("statement", "eq_statement.jsonl"),
    ("question", "eq_question.jsonl"),
    ("command", "eq_command.jsonl"),
    ("indirect", "eq_indirect.jsonl"),
]


def normalize_captions(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def extract_source_caption(value: Any) -> str | None:
    captions = normalize_captions(value)
    if len(captions) != 1:
        return None
    return captions[0]


def captions_compatible(
    merged_captions: list[str],
    record_captions: list[str],
    qtype: str,
) -> bool:
    if qtype == "full_caption":
        return record_captions == merged_captions
    if len(record_captions) != 1:
        return False
    return record_captions[0] in merged_captions


def iter_json_objects(text: str) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at offset {idx}, got {type(obj)}")
        yield obj
        idx = end


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return list(iter_json_objects(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge EQ JSONL files by audio_id.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the six eq_*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input-dir>/eq_by_clip.jsonl)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON blocks instead of strict one-object-per-line JSONL.",
    )
    args = parser.parse_args()
    pretty = args.pretty
    out_path = args.output or (args.input_dir / "eq_by_clip.jsonl")

    by_audio: dict[str, dict[str, Any]] = {}

    for qtype, filename in QUERY_TYPE_FILES:
        path = args.input_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing file: {path}")
        for rec in load_records(path):
            aid = str(rec.get("audio_id", "")).strip()
            if not aid:
                continue
            gq = str(rec.get("generated_query", "")).strip()
            if not gq:
                raise SystemExit(
                    f"Empty generated_query for audio_id={aid!r} in {path}"
                )
            if aid not in by_audio:
                by_audio[aid] = {
                    "audio_id": aid,
                    "dataset": rec.get("dataset"),
                    "dataset_slug": rec.get("dataset_slug"),
                    "original_captions": rec.get("original_captions"),
                    "source_caption": None,
                    "metadata": rec.get("metadata"),
                    "source_model": rec.get("source_model"),
                    "regen_model": rec.get("regen_model"),
                    "generated_queries": {},
                }
            else:
                merged_captions = normalize_captions(by_audio[aid]["original_captions"])
                record_captions = normalize_captions(rec.get("original_captions"))
                if not captions_compatible(merged_captions, record_captions, qtype):
                    print(
                        f"[WARN] original_captions mismatch for {aid} in {filename}; "
                        "keeping first file's captions.",
                    )

            if qtype != "full_caption":
                source_caption = extract_source_caption(rec.get("original_captions"))
                if source_caption is None:
                    print(
                        f"[WARN] expected exactly one source caption for {aid} in {filename}; "
                        "could not determine source_caption.",
                    )
                else:
                    merged_source = by_audio[aid]["source_caption"]
                    if merged_source is None:
                        by_audio[aid]["source_caption"] = source_caption
                    elif merged_source != source_caption:
                        print(
                            f"[WARN] source_caption mismatch for {aid}: "
                            f"{merged_source!r} != {source_caption!r} ({filename})",
                        )
            by_audio[aid]["generated_queries"][qtype] = gq

    missing: list[tuple[str, list[str]]] = []
    for aid, row in by_audio.items():
        g = row["generated_queries"]
        want = [qt for qt, _ in QUERY_TYPE_FILES]
        absent = [qt for qt in want if qt not in g]
        if absent:
            missing.append((aid, absent))

    if missing:
        for aid, absent in missing:
            print(f"[ERROR] {aid} missing types: {absent}")
        raise SystemExit("Not all query types present for every audio_id.")

    # Stable order: same as first file's encounter order if possible
    first_path = args.input_dir / "eq_key_phrase.jsonl"
    order = [r["audio_id"] for r in load_records(first_path) if r.get("audio_id")]
    seen = set()
    ordered_ids = []
    for aid in order:
        if aid in by_audio and aid not in seen:
            seen.add(aid)
            ordered_ids.append(aid)
    for aid in sorted(by_audio.keys()):
        if aid not in seen:
            ordered_ids.append(aid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        blocks = []
        for aid in ordered_ids:
            blocks.append(
                json.dumps(by_audio[aid], indent=2, ensure_ascii=False),
            )
        out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    else:
        with out_path.open("w", encoding="utf-8") as handle:
            for aid in ordered_ids:
                handle.write(json.dumps(by_audio[aid], ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(ordered_ids)} records -> {out_path}")


if __name__ == "__main__":
    main()
