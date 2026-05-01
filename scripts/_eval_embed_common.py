#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "input" / "datasets"))


def resolve_project_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def load_eval_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_enabled_model_configs(config: dict[str, Any], model_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    selected_names = {name.lower() for name in model_names} if model_names else None
    model_configs: dict[str, dict[str, Any]] = {}

    for model_config in config.get("models", []):
        if not model_config.get("enabled", True):
            continue

        name = str(model_config.get("name", "")).lower()
        if not name:
            continue
        if selected_names is not None and name not in selected_names:
            continue

        normalized_config = dict(model_config)
        normalized_config["name"] = name
        for key in ("repo_path", "checkpoint_path"):
            if key in normalized_config:
                normalized_config[key] = resolve_project_path(normalized_config[key])
        model_configs[name] = normalized_config

    if selected_names is not None:
        missing = selected_names - set(model_configs)
        if missing:
            raise ValueError(f"Model(s) not found or disabled in config: {', '.join(sorted(missing))}")

    if not model_configs:
        raise ValueError("No enabled models found in config.")

    return model_configs


def get_execution_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config.get("execution", {}).get(key, default)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_caption_rows(caption_jsonl: str | Path) -> list[dict[str, Any]]:
    caption_path = Path(caption_jsonl)
    text = caption_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    rows: list[dict[str, Any]] = []

    try:
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        return rows
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        while cursor < text_len and text[cursor].isspace():
            cursor += 1
        if cursor >= text_len:
            break

        try:
            obj, next_cursor = decoder.raw_decode(text, cursor)
        except json.JSONDecodeError as exc:
            line_number = text.count("\n", 0, cursor) + 1
            raise ValueError(f"Invalid JSON at {caption_path}:{line_number}") from exc

        rows.append(obj)
        cursor = next_cursor

    return rows


def build_file_name_to_audio_id(caption_rows: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in caption_rows:
        file_name = str(row.get("metadata", {}).get("file_name", "")).strip()
        audio_id = str(row.get("audio_id", "")).strip()
        if file_name and audio_id:
            mapping[file_name] = audio_id
    return mapping
