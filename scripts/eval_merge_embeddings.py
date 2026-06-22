#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_embed_common import (
    DEFAULT_CONFIG_PATH,
    get_enabled_model_configs,
    load_caption_rows,
    load_eval_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge audio, source-text, and generated-text embeddings by audio_id.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default="clotho")
    parser.add_argument("--caption-jsonl", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--text-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any selected caption row is missing an embedding.",
    )
    args = parser.parse_args()
    config = load_eval_config(args.config)
    args.model_configs = get_enabled_model_configs(config, args.models)
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def index_unique(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    label: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")).strip() for field in key_fields)
        if not all(key):
            continue
        if key in indexed:
            raise ValueError(f"Duplicate {label} key: {key}")
        indexed[key] = row
    return indexed


def merge_model(
    model_name: str,
    caption_rows: list[dict[str, Any]],
    audio_root: Path,
    text_root: Path,
    output_root: Path,
    strict: bool,
) -> tuple[int, int]:
    audio_rows = read_jsonl(audio_root / model_name / "emb.jsonl")
    source_rows = read_jsonl(text_root / model_name / "source_emb.jsonl")
    generated_rows = read_jsonl(text_root / model_name / "generated_emb.jsonl")

    audio_by_id = index_unique(audio_rows, ("audio_id",), "audio")
    source_by_id = index_unique(source_rows, ("audio_id",), "source text")
    generated_by_key = index_unique(
        generated_rows,
        ("audio_id", "query_type"),
        "generated text",
    )

    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "merged.jsonl"
    written = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for caption_row in caption_rows:
            audio_id = str(caption_row.get("audio_id", "")).strip()
            if not audio_id:
                continue

            audio = audio_by_id.get((audio_id,))
            source = source_by_id.get((audio_id,))
            generated_queries = caption_row.get("generated_queries", {})
            if not isinstance(generated_queries, dict):
                generated_queries = {}

            missing: list[str] = []
            if audio is None:
                missing.append("audio")
            if source is None:
                missing.append("source")

            generated: dict[str, dict[str, Any]] = {}
            for query_type in generated_queries:
                generated_row = generated_by_key.get((audio_id, str(query_type)))
                if generated_row is None:
                    missing.append(str(query_type))
                    continue
                generated[str(query_type)] = {
                    "text": generated_row.get("generated_text", ""),
                    "emb": generated_row.get("emb", []),
                }

            if missing:
                skipped += 1
                message = (
                    f"{model_name}/{audio_id} is missing embeddings: "
                    f"{', '.join(missing)}"
                )
                if strict:
                    raise ValueError(message)
                print(f"[WARN] {message}; skipping row")
                continue

            payload = {
                "audio_id": audio_id,
                "file_name": audio.get("file_name", ""),
                "model": model_name,
                "audio_emb": audio.get("emb", []),
                "source": {
                    "text": source.get("source_caption", ""),
                    "emb": source.get("emb", []),
                },
                "generated_queries": generated,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    if not written:
        raise RuntimeError(f"No merged rows were written for model {model_name}")
    print(f"[INFO] Wrote {written} merged rows to {output_path}")
    return written, skipped


def main() -> None:
    args = parse_args()
    caption_rows = load_caption_rows(args.caption_jsonl)
    if args.limit is not None:
        caption_rows = caption_rows[: args.limit]
    if not caption_rows:
        raise RuntimeError("Caption JSONL contains no rows.")

    for model_name in args.model_configs:
        merge_model(
            model_name=model_name,
            caption_rows=caption_rows,
            audio_root=Path(args.audio_root),
            text_root=Path(args.text_root),
            output_root=Path(args.output_root),
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
