#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_embed_common import get_execution_value, load_eval_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full evaluation embedding pipeline: source text, generated text, audio, and merge."
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
        "--caption-jsonl",
        default=None,
        help="Caption JSONL used by all stages.",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory containing audio files.",
    )
    parser.add_argument(
        "--audio-output-root",
        default=None,
        help="Root output directory for audio embeddings.",
    )
    parser.add_argument(
        "--text-output-root",
        default=None,
        help="Root output directory for source/generated text embeddings.",
    )
    parser.add_argument(
        "--merged-output-root",
        default=None,
        help="Root output directory for merged embeddings.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run. Defaults to all supported models.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=None,
        help="Batch size for source/generated text embedding extraction.",
    )
    parser.add_argument(
        "--audio-batch-size",
        type=int,
        default=None,
        help="Batch size for audio embedding extraction.",
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
        help="Optional row/file limit for smoke testing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Pass --strict to the merge step.",
    )
    args = parser.parse_args()
    config = load_eval_config(args.config)
    dataset_config = config.get("dataset", {})
    dataset = args.dataset
    args.caption_jsonl = (
        args.caption_jsonl
        or dataset_config.get("eval_caption_jsonl")
        or dataset_config.get("caption_jsonl")
        or str(PROJECT_ROOT / "input" / "captions" / dataset / "eq_by_clip.jsonl")
    )
    args.audio_dir = args.audio_dir or dataset_config.get("eval_audio_dir") or dataset_config.get("audio_dir") or str(PROJECT_ROOT / "input" / dataset / "audio")
    args.audio_output_root = args.audio_output_root or str(PROJECT_ROOT / "results" / "audioEmb" / dataset)
    args.text_output_root = args.text_output_root or str(PROJECT_ROOT / "results" / "testEmb" / dataset)
    args.merged_output_root = args.merged_output_root or str(PROJECT_ROOT / "results" / "mergedEmb" / dataset)
    args.text_batch_size = args.text_batch_size or int(get_execution_value(config, "text_batch_size", get_execution_value(config, "batch_size", 64)))
    args.audio_batch_size = args.audio_batch_size or int(get_execution_value(config, "audio_batch_size", get_execution_value(config, "batch_size", 16)))
    args.device = args.device or str(get_execution_value(config, "device", "auto"))
    return args


def run_step(step_name: str, command: list[str]) -> None:
    print(f"[pipeline] Starting {step_name}")
    print(f"[pipeline] Command: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(f"[pipeline] Finished {step_name}")


def main() -> None:
    args = parse_args()

    base_args = [
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--caption-jsonl",
        args.caption_jsonl,
        "--device",
        args.device,
    ]
    if args.models:
        base_args.extend(["--models", *args.models])
    if args.limit is not None:
        base_args.extend(["--limit", str(args.limit)])

    source_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_extract_source_text_embeddings.py"),
        *base_args,
        "--output-root",
        args.text_output_root,
        "--batch-size",
        str(args.text_batch_size),
    ]
    generated_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_extract_generated_text_embeddings.py"),
        *base_args,
        "--output-root",
        args.text_output_root,
        "--batch-size",
        str(args.text_batch_size),
    ]
    audio_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_extract_audio_embeddings.py"),
        *base_args,
        "--audio-dir",
        args.audio_dir,
        "--output-root",
        args.audio_output_root,
        "--batch-size",
        str(args.audio_batch_size),
    ]
    merge_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_merge_embeddings.py"),
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--caption-jsonl",
        args.caption_jsonl,
        "--audio-root",
        args.audio_output_root,
        "--text-root",
        args.text_output_root,
        "--output-root",
        args.merged_output_root,
    ]
    if args.models:
        merge_command.extend(["--models", *args.models])
    if args.strict:
        merge_command.append("--strict")

    run_step("source embeddings", source_command)
    run_step("generated embeddings", generated_command)
    run_step("audio embeddings", audio_command)
    run_step("merge embeddings", merge_command)


if __name__ == "__main__":
    main()
