from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml


def _require_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _caption_columns(fieldnames: list[str]) -> list[str]:
    numbered: list[tuple[int, str]] = []
    for name in fieldnames:
        match = re.fullmatch(r"caption_?(\d+)", name.strip(), flags=re.IGNORECASE)
        if match:
            numbered.append((int(match.group(1)), name))
    return [name for _, name in sorted(numbered)]


def load_audiocaps(
    csv_path: str,
    split: str = "test",
    expected_captions: int | None = 5,
) -> list[dict[str, Any]]:
    """Load and group AudioCaps captions by ``youtube_id`` and ``start_time``."""

    path = _require_file(csv_path, "AudioCaps captions CSV")
    grouped: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"youtube_id", "start_time", "caption"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"AudioCaps CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            youtube_id = _clean(row.get("youtube_id"))
            start_time = _clean(row.get("start_time"))
            caption = _clean(row.get("caption"))
            if not youtube_id or not start_time or not caption:
                continue

            try:
                numeric_start = float(start_time)
                start_key = (
                    str(int(numeric_start))
                    if numeric_start.is_integer()
                    else str(numeric_start)
                )
            except ValueError:
                start_key = start_time

            key = (youtube_id, start_key)
            record = grouped.setdefault(
                key,
                {
                    "audio_id": f"{youtube_id}_{start_key}",
                    "dataset": "audiocaps",
                    "dataset_slug": f"audiocaps_{split}",
                    "original_captions": [],
                    "metadata": {
                        "split": split,
                        "youtube_id": youtube_id,
                        "start_time": start_key,
                        "audiocap_ids": [],
                    },
                },
            )
            record["original_captions"].append(caption)
            audiocap_id = _clean(row.get("audiocap_id"))
            if audiocap_id:
                record["metadata"]["audiocap_ids"].append(audiocap_id)

    records = list(grouped.values())
    if not records:
        raise ValueError(f"No valid AudioCaps caption records found in {path}")

    if expected_captions is not None:
        invalid = [
            (record["audio_id"], len(record["original_captions"]))
            for record in records
            if len(record["original_captions"]) != expected_captions
        ]
        if invalid:
            preview = ", ".join(
                f"{audio_id}={count}" for audio_id, count in invalid[:5]
            )
            raise ValueError(
                f"Expected {expected_captions} caption rows per AudioCaps clip; "
                f"found invalid groups: {preview}"
            )

    return records


def load_clotho(csv_path: str, split: str = "evaluation") -> list[dict[str, Any]]:
    """Load Clotho captions from either wide or long CSV format.

    Official Clotho files use one row per audio file with ``caption_1`` ...
    ``caption_5``. Long-form files with one ``caption`` per row are also
    accepted and grouped by file name.
    """

    path = _require_file(csv_path, "Clotho captions CSV")
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        file_column = next(
            (
                candidate
                for candidate in ("file_name", "filename", "audio_file", "file")
                if candidate in fieldnames
            ),
            None,
        )
        if file_column is None:
            raise ValueError(
                f"Clotho CSV must contain a file-name column; found: {fieldnames}"
            )

        caption_columns = _caption_columns(fieldnames)
        if not caption_columns and "caption" in fieldnames:
            caption_columns = ["caption"]
        if not caption_columns:
            raise ValueError(
                "Clotho CSV must contain caption_1...caption_N columns or a caption column."
            )

        for row in reader:
            file_name = _clean(row.get(file_column))
            if not file_name:
                continue

            audio_id = Path(file_name).stem
            if not audio_id:
                continue

            record = grouped.setdefault(
                audio_id,
                {
                    "audio_id": audio_id,
                    "dataset": "clotho",
                    "dataset_slug": f"clotho_{split}",
                    "original_captions": [],
                    "metadata": {
                        "split": split,
                        "file_name": file_name,
                    },
                },
            )
            existing_file_name = record["metadata"]["file_name"]
            if existing_file_name != file_name:
                raise ValueError(
                    f"Clotho audio_id collision for {audio_id!r}: "
                    f"{existing_file_name!r} and {file_name!r}"
                )

            for column in caption_columns:
                caption = _clean(row.get(column))
                if caption and caption not in record["original_captions"]:
                    record["original_captions"].append(caption)

    records = [record for record in grouped.values() if record["original_captions"]]
    if not records:
        raise ValueError(f"No valid Clotho caption records found in {path}")

    return records


def load_mecat(json_dir: str, split: str = "default") -> list[dict[str, Any]]:
    """Load MeCAT ``short`` captions from one JSON file or a directory tree."""

    data: list[dict[str, Any]] = []
    seen_audio_ids: dict[str, Path] = {}
    base_path = Path(json_dir).expanduser()
    if base_path.is_file():
        json_paths = [base_path]
    elif base_path.is_dir():
        json_paths = sorted(base_path.rglob("*.json"))
    else:
        raise FileNotFoundError(f"MeCAT JSON path not found: {base_path}")

    for json_path in json_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid MeCAT JSON file: {json_path}") from exc

        short_captions = payload.get("short")
        if isinstance(short_captions, str):
            short_captions = [short_captions]
        if not isinstance(short_captions, list):
            continue

        original_captions = list(
            dict.fromkeys(
                _clean(caption) for caption in short_captions if _clean(caption)
            )
        )
        if not original_captions:
            continue

        audio_id = json_path.stem
        if audio_id in seen_audio_ids:
            raise ValueError(
                f"Duplicate MeCAT audio_id {audio_id!r}: "
                f"{seen_audio_ids[audio_id]} and {json_path}"
            )
        seen_audio_ids[audio_id] = json_path
        file_name = (
            _clean(payload.get("file_name"))
            or _clean(payload.get("audio_file"))
            or f"{audio_id}.flac"
        )
        metadata = {
            "split": split,
            "json_file": json_path.name,
            "file_name": Path(file_name).name,
        }
        domain = payload.get("domain")
        if isinstance(domain, str) and domain.strip():
            metadata["domain"] = domain.strip()

        data.append(
            {
                "audio_id": audio_id,
                "dataset": "mecat",
                "dataset_slug": f"mecat_{split}",
                "original_captions": original_captions,
                "metadata": metadata,
            }
        )

    if not data:
        raise ValueError(
            f"No valid MeCAT JSON records with a short field found in {base_path}"
        )

    return data


def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
