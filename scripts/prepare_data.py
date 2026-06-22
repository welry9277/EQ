#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CLOTHO_RECORD_URL = "https://zenodo.org/records/4783391"
CLOTHO_CAPTION_URLS = {
    split: f"{CLOTHO_RECORD_URL}/files/clotho_captions_{split}.csv?download=1"
    for split in ("development", "validation", "evaluation")
}
AUDIOCAPS_TEST_URL = (
    "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/test.csv"
)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "EQ/0.1"})
    print(f"[INFO] Downloading {url}")
    try:
        with (
            urllib.request.urlopen(request) as response,
            destination.open("wb") as output,
        ):
            output.write(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to download {url}: {exc}") from exc
    print(f"[INFO] Wrote {destination}")


def prepare_clotho(input_root: Path, split: str) -> None:
    splits = CLOTHO_CAPTION_URLS if split == "all" else (split,)
    destination_dir = input_root / "clotho"
    for current_split in splits:
        destination = destination_dir / f"clotho_captions_{current_split}.csv"
        download_file(CLOTHO_CAPTION_URLS[current_split], destination)

    audio_dir = destination_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[INFO] Download the matching Clotho audio archive from "
        f"{CLOTHO_RECORD_URL}, extract it, and place the audio files under {audio_dir}."
    )


def prepare_audiocaps(input_root: Path) -> None:
    destination = input_root / "audiocaps" / "test.csv"
    download_file(AUDIOCAPS_TEST_URL, destination)
    with destination.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    group_counts = Counter(
        (row.get("youtube_id", ""), row.get("start_time", "")) for row in rows
    )
    invalid = [key for key, count in group_counts.items() if count != 5]
    if invalid:
        raise SystemExit(
            f"AudioCaps test.csv contains {len(invalid)} groups without five captions."
        )
    print(
        f"[INFO] AudioCaps test.csv contains {len(group_counts)} clips and "
        f"{len(rows)} captions; every clip has five captions."
    )


def prepare_mecat(input_root: Path) -> None:
    mecat_dir = input_root / "mecat"
    json_dir = mecat_dir / "json_files"
    audio_dir = mecat_dir / "flac_files"
    json_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Place MeCAT JSON files under {json_dir}.")
    print(f"[INFO] Place matching .flac files under {audio_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare AudioCaps, Clotho, or MeCAT inputs for EQ generation.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["audiocaps", "clotho", "mecat", "all"],
        help="Dataset to prepare under --input-root.",
    )
    parser.add_argument(
        "--clotho-split",
        choices=["development", "validation", "evaluation", "all"],
        default="evaluation",
        help="Clotho caption split to download. Default: evaluation.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("input"),
        help="Root input directory. Default: input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in {"audiocaps", "all"}:
        prepare_audiocaps(args.input_root)
    if args.dataset in {"clotho", "all"}:
        prepare_clotho(args.input_root, args.clotho_split)
    if args.dataset in {"mecat", "all"}:
        prepare_mecat(args.input_root)


if __name__ == "__main__":
    main()
