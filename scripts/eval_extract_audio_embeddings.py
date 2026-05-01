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

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "checkpoints" / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "input" / "datasets"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from scripts._eval_embed_common import (
    PROJECT_ROOT,
    build_file_name_to_audio_id,
    get_enabled_model_configs,
    get_execution_value,
    load_eval_config,
    load_caption_rows,
    resolve_device,
)
from src.clap_eval.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings for local evaluation audio with multiple CLAP models."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="YAML config containing dataset, model, and execution settings.",
    )
    parser.add_argument(
        "--dataset",
        default="clotho",
        help="Dataset name used for default input/output paths, e.g. clotho or mecats.",
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
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding extraction.")
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
    dataset_config = config.get("dataset", {})
    args.audio_dir = args.audio_dir or dataset_config.get("eval_audio_dir") or dataset_config.get("audio_dir") or str(PROJECT_ROOT / "input" / args.dataset / "audio")
    args.caption_jsonl = (
        args.caption_jsonl
        or dataset_config.get("eval_caption_jsonl")
        or dataset_config.get("caption_jsonl")
        or str(PROJECT_ROOT / "input" / "captions" / args.dataset / "eq_by_clip.jsonl")
    )
    args.output_root = args.output_root or str(PROJECT_ROOT / "results" / "audioEmb" / args.dataset)
    args.batch_size = args.batch_size or int(get_execution_value(config, "audio_batch_size", get_execution_value(config, "batch_size", 16)))
    args.device = args.device or str(get_execution_value(config, "device", "auto"))
    args.model_configs = get_enabled_model_configs(config, args.models)
    return args


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_audio_records(audio_dir: Path, file_name_to_audio_id: dict[str, str]) -> list[dict]:
    records: list[dict] = []
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file():
            continue
        file_name = path.name
        records.append(
            {
                "audio_id": file_name_to_audio_id.get(file_name, path.stem),
                "file_name": file_name,
                "audio_path": path,
            }
        )
    if not records:
        raise RuntimeError(f"No audio files found under {audio_dir}")
    return records


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    caption_rows = load_caption_rows(args.caption_jsonl)
    file_name_to_audio_id = build_file_name_to_audio_id(caption_rows)

    audio_records = build_audio_records(Path(args.audio_dir), file_name_to_audio_id)
    if args.limit is not None:
        audio_records = audio_records[: args.limit]

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
                    audio_array, _ = librosa.load(record["audio_path"], sr=16000, mono=True)
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
