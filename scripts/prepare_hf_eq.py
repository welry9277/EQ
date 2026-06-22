#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

QUERY_TYPES = (
    "full_caption",
    "key_phrase",
    "statement",
    "question",
    "command",
    "indirect",
)


def safe_file_stem(audio_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", audio_id).strip("._")
    return stem or "audio"


def write_audio(audio: Any, destination: Path) -> None:
    if not isinstance(audio, dict):
        raise ValueError("Expected the Hugging Face audio column to be a dictionary.")

    audio_bytes = audio.get("bytes")
    if audio_bytes:
        destination.write_bytes(audio_bytes)
        return

    source = str(audio.get("path") or "").strip()
    if not source:
        raise ValueError("Hugging Face audio row has neither bytes nor path.")
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
        return

    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio source path not found: {source_path}")
    shutil.copyfile(source_path, destination)


def export_rows(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    dataset_filter: str | None,
    limit: int | None = None,
    hf_repo: str = "msnowchanj/EQ",
    split: str = "test",
) -> int:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    caption_path = output_dir / "eq_by_clip.jsonl"
    written = 0
    seen_ids: set[str] = set()

    with caption_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            dataset_name = str(row.get("dataset", "")).strip().lower()
            if dataset_filter and dataset_name != dataset_filter:
                continue

            audio_id = str(row.get("audio_id", "")).strip()
            if not audio_id:
                raise ValueError("Hugging Face EQ row is missing audio_id.")
            if audio_id in seen_ids:
                raise ValueError(
                    f"Duplicate audio_id in Hugging Face EQ data: {audio_id}"
                )
            seen_ids.add(audio_id)

            file_name = f"{written:06d}_{safe_file_stem(audio_id)}.wav"
            write_audio(row.get("audio"), audio_dir / file_name)
            generated_queries = {
                query_type: str(row.get(query_type, "")).strip()
                for query_type in QUERY_TYPES
                if str(row.get(query_type, "")).strip()
            }
            if len(generated_queries) != len(QUERY_TYPES):
                missing = sorted(set(QUERY_TYPES) - set(generated_queries))
                raise ValueError(
                    f"Hugging Face EQ row {audio_id!r} is missing query fields: "
                    f"{', '.join(missing)}"
                )

            payload = {
                "audio_id": audio_id,
                "dataset": dataset_name,
                "dataset_slug": f"{dataset_name}_hf_test",
                "source_caption": str(row.get("source_caption", "")).strip(),
                "metadata": {
                    "file_name": file_name,
                    "source": "huggingface",
                    "hf_repo": hf_repo,
                    "hf_split": split,
                },
                "generated_queries": generated_queries,
            }
            if not payload["source_caption"]:
                raise ValueError(
                    f"Hugging Face EQ row {audio_id!r} has an empty source_caption."
                )
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            if limit is not None and written >= limit:
                break

    if not written:
        raise ValueError(
            f"No Hugging Face EQ rows matched dataset filter {dataset_filter!r}."
        )

    manifest = {
        "hf_repo": hf_repo,
        "split": split,
        "dataset_filter": dataset_filter,
        "limit": limit,
        "num_examples": written,
        "caption_jsonl": str(caption_path),
        "audio_dir": str(audio_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Hugging Face msnowchanj/EQ and export a split to the local "
            "caption JSONL + audio layout used by eval_pipeline.py."
        ),
    )
    parser.add_argument("--repo", default="msnowchanj/EQ")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help=(
            "Optional repository-relative Parquet files. Use this to download only "
            "shards containing the selected dataset."
        ),
    )
    parser.add_argument(
        "--dataset-filter",
        choices=["audiocaps", "clotho", "mecat", "all"],
        default="audiocaps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/hf_eq/test/audiocaps"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("input/hf_cache"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing exported directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install project dependencies with `uv sync --frozen`."
        ) from exc

    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_files:
        urls = [
            f"https://huggingface.co/datasets/{args.repo}/resolve/main/{path}"
            for path in args.data_files
        ]
        dataset = load_dataset(
            "parquet",
            data_files={args.split: urls},
            split=args.split,
            cache_dir=str(args.cache_dir),
        )
    else:
        dataset = load_dataset(
            args.repo,
            split=args.split,
            cache_dir=str(args.cache_dir),
        )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset_filter = None if args.dataset_filter == "all" else args.dataset_filter
    written = export_rows(
        rows=dataset,
        output_dir=args.output_dir,
        dataset_filter=dataset_filter,
        limit=args.limit,
        hf_repo=args.repo,
        split=args.split,
    )
    print(f"[INFO] Exported {written} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
