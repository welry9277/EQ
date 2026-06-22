#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "HF_HOME", str(PROJECT_ROOT / "checkpoints" / ".cache" / "huggingface")
)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from scripts._eval_embed_common import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    get_dataset_config,
    get_enabled_model_configs,
    get_execution_value,
    load_caption_rows,
    load_eval_config,
    resolve_device,
)
from src.clap_eval.models import get_model

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings for local evaluation audio with multiple CLAP models."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config containing dataset, model, and execution settings.",
    )
    parser.add_argument(
        "--dataset",
        default="clotho",
        help="Dataset name used for default input/output paths: clotho or mecat.",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory containing evaluation audio files.",
    )
    parser.add_argument(
        "--caption-jsonl",
        default=None,
        help="Caption JSONL used to map file_name to audio_id.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory. Files are written to <output-root>/<model>/emb.jsonl.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run. Defaults to all supported models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use. Example: auto, cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional audio-file limit for smoke testing.",
    )
    args = parser.parse_args()
    config = load_eval_config(args.config)
    dataset_config = get_dataset_config(config, args.dataset)
    args.audio_dir = (
        args.audio_dir
        or dataset_config.get("eval_audio_dir")
        or dataset_config.get("audio_dir")
        or str(PROJECT_ROOT / "input" / args.dataset / "audio")
    )
    args.caption_jsonl = (
        args.caption_jsonl
        or dataset_config.get("eval_caption_jsonl")
        or dataset_config.get("caption_jsonl")
        or str(PROJECT_ROOT / "results" / "eq" / args.dataset / "eq_by_clip.jsonl")
    )
    args.output_root = args.output_root or str(
        PROJECT_ROOT / "results" / "audioEmb" / args.dataset
    )
    args.batch_size = args.batch_size or int(
        get_execution_value(
            config, "audio_batch_size", get_execution_value(config, "batch_size", 16)
        )
    )
    args.device = args.device or str(get_execution_value(config, "device", "auto"))
    args.model_configs = get_enabled_model_configs(config, args.models)
    return args


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_audio_records(audio_dir: Path, caption_rows: list[dict]) -> list[dict]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_paths = [
        path
        for path in sorted(audio_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    by_name = {path.name: path for path in audio_paths}
    by_stem = {path.stem: path for path in audio_paths}
    records: list[dict] = []
    missing: list[str] = []

    for row in caption_rows:
        audio_id = str(row.get("audio_id", "")).strip()
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        requested_name = str(
            row.get("file_name") or metadata.get("file_name") or ""
        ).strip()
        path = by_name.get(Path(requested_name).name) if requested_name else None
        if path is None:
            path = by_stem.get(audio_id)
        if path is None:
            missing.append(requested_name or audio_id)
            continue
        records.append(
            {
                "audio_id": audio_id,
                "file_name": path.name,
                "audio_path": path,
            }
        )

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(
            f"Could not match caption rows to audio under {audio_dir}: {preview}{suffix}"
        )
    if not records:
        raise RuntimeError(f"No audio records matched under {audio_dir}")
    return records


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    caption_rows = load_caption_rows(args.caption_jsonl)
    if args.limit is not None:
        caption_rows = caption_rows[: args.limit]
    audio_records = build_audio_records(Path(args.audio_dir), caption_rows)

    for model_name, model_config in args.model_configs.items():
        model = get_model(model_name, model_config, device)
        output_dir = Path(args.output_root) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "emb.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for batch in tqdm(
                list(batched(audio_records, args.batch_size)),
                desc=f"Audio embeddings: {model_name}",
            ):
                audios = []
                for record in batch:
                    # Standardize to mono 16 kHz here so every model wrapper receives a consistent batch.
                    audio_array, _ = librosa.load(
                        record["audio_path"], sr=16000, mono=True
                    )
                    audios.append(audio_array)

                embeddings = model.get_audio_embedding(audios, 16000)
                for record, emb in zip(batch, embeddings):
                    payload = {
                        "audio_id": record["audio_id"],
                        "file_name": record["file_name"],
                        "emb": emb.flatten().tolist(),
                        "model": model_name,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
