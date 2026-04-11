#!/usr/bin/env -S uv run python
"""Export AudioCaps from Hugging Face Hub to a CSV compatible with load_audiocaps / makeeq.

Dataset: https://huggingface.co/datasets/d0rj/audiocaps  (AudioCaps — not Clotho.)

Install optional dependency first:
  uv sync --extra hf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing package `datasets`. Run: uv sync --extra hf\n"
            f"Import error: {exc}"
        ) from exc

    parser = argparse.ArgumentParser(
        description='Export d0rj/audiocaps from Hugging Face to AudioCaps-style CSV.',
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="HF split name (default: test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g. input/audiocaps/test.csv).",
    )
    args = parser.parse_args()

    ds = load_dataset("d0rj/audiocaps", split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["youtube_id", "start_time", "caption"]
    n_written = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ds:
            youtube_id = str(row.get("youtube_id", "")).strip()
            caption = str(row.get("caption", "")).strip()
            st = row.get("start_time")
            if st is None:
                continue
            start_time = str(st).strip()
            if not youtube_id or not start_time or not caption:
                continue
            writer.writerow(
                {
                    "youtube_id": youtube_id,
                    "start_time": start_time,
                    "caption": caption,
                }
            )
            n_written += 1

    print(f"[INFO] Wrote {n_written} CSV rows to {args.output}")


if __name__ == "__main__":
    main()
