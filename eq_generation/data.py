from __future__ import annotations

import csv
from pathlib import Path

import yaml


def load_audiocaps(csv_path: str, split: str = "test") -> list[dict]:
    data: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            youtube_id = row.get("youtube_id", "").strip()
            start_time = row.get("start_time", "").strip()
            caption = row.get("caption", "").strip()
            if not youtube_id or not start_time or not caption:
                continue

            try:
                start_time_float = float(start_time)
                start_time_key = int(start_time_float)
            except ValueError:
                start_time_float = start_time
                start_time_key = start_time

            audio_id = f"{youtube_id}_{start_time_key}"
            record = data.setdefault(
                audio_id,
                {
                    "audio_id": audio_id,
                    "dataset": "audiocaps",
                    "dataset_slug": f"audiocaps_{split}",
                    "original_captions": [],
                    "metadata": {
                        "split": split,
                        "youtube_id": youtube_id,
                        "start_time": start_time_float,
                    },
                },
            )
            record["original_captions"].append(caption)

    return list(data.values())


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
