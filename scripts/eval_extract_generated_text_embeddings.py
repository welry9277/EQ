#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract generated-query text embeddings from eq_by_clip.jsonl with multiple CLAP models."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config containing model and execution settings.",
    )
    parser.add_argument(
        "--dataset",
        default="clotho",
        help="Dataset name used for default input/output paths: clotho or mecat.",
    )
    parser.add_argument(
        "--caption-jsonl",
        default=None,
        help="Input JSONL containing generated_queries fields.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory. Files are written to <output-root>/<model>/generated_emb.jsonl.",
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
        help="Batch size for text embedding extraction.",
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
        help="Optional row limit for smoke testing.",
    )
    args = parser.parse_args()
    config = load_eval_config(args.config)
    dataset_config = get_dataset_config(config, args.dataset)
    args.caption_jsonl = (
        args.caption_jsonl
        or dataset_config.get("eval_caption_jsonl")
        or dataset_config.get("caption_jsonl")
        or str(PROJECT_ROOT / "results" / "eq" / args.dataset / "eq_by_clip.jsonl")
    )
    args.output_root = args.output_root or str(
        PROJECT_ROOT / "results" / "testEmb" / args.dataset
    )
    args.batch_size = args.batch_size or int(
        get_execution_value(
            config, "text_batch_size", get_execution_value(config, "batch_size", 64)
        )
    )
    args.device = args.device or str(get_execution_value(config, "device", "auto"))
    args.model_configs = get_enabled_model_configs(config, args.models)
    return args


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_generated_records(caption_rows: list[dict]) -> list[dict]:
    records: list[dict] = []
    for row in caption_rows:
        generated_queries = row.get("generated_queries", {})
        if not isinstance(generated_queries, dict):
            continue

        audio_id = str(row.get("audio_id", ""))
        file_name = str(row.get("metadata", {}).get("file_name", ""))
        for query_type, query_text in generated_queries.items():
            text = str(query_text).strip()
            if not text:
                continue
            records.append(
                {
                    "audio_id": audio_id,
                    "file_name": file_name,
                    "query_type": str(query_type),
                    "generated_text": text,
                }
            )

    if not records:
        raise RuntimeError("No generated_queries rows found in the caption JSONL.")
    return records


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    caption_rows = load_caption_rows(args.caption_jsonl)
    if args.limit is not None:
        caption_rows = caption_rows[: args.limit]
    generated_records = build_generated_records(caption_rows)

    for model_name, model_config in args.model_configs.items():
        model = get_model(model_name, model_config, device)
        output_dir = Path(args.output_root) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "generated_emb.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for batch in tqdm(
                list(batched(generated_records, args.batch_size)),
                desc=f"Generated text embeddings: {model_name}",
            ):
                texts = [record["generated_text"] for record in batch]
                embeddings = model.get_text_embedding(texts)
                for record, emb in zip(batch, embeddings):
                    payload = {
                        "audio_id": record["audio_id"],
                        "file_name": record["file_name"],
                        "query_type": record["query_type"],
                        "generated_text": record["generated_text"],
                        "emb": emb.flatten().tolist(),
                        "model": model_name,
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
