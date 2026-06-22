#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_embed_common import (
    DEFAULT_CONFIG_PATH,
    get_dataset_config,
    get_execution_value,
    load_eval_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete text-to-audio evaluation: extract embeddings, merge "
            "them, compute Recall@K, and create a summary plot."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--dataset",
        choices=["audiocaps", "clotho", "mecat"],
        default="clotho",
    )
    parser.add_argument("--caption-jsonl", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--audio-output-root", default=None)
    parser.add_argument("--text-output-root", default=None)
    parser.add_argument("--merged-output-root", default=None)
    parser.add_argument("--retrieval-output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--pools", nargs="+", default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=None)
    parser.add_argument("--text-batch-size", type=int, default=None)
    parser.add_argument("--audio-batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate the first N caption rows for a smoke test.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Skip rows with missing embeddings instead of failing during merge.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate the Recall@K bar chart.",
    )
    parser.add_argument(
        "--plot-pool",
        default=None,
        help="Query pool shown in the bar chart. Defaults to full_caption or the first selected pool.",
    )
    parser.add_argument(
        "--refresh-hf",
        action="store_true",
        help="Re-download and replace configured Hugging Face evaluation inputs.",
    )
    args = parser.parse_args()

    config = load_eval_config(args.config)
    dataset_config = get_dataset_config(config, args.dataset)
    args.caption_jsonl = (
        args.caption_jsonl
        or dataset_config.get("caption_jsonl")
        or dataset_config.get("eval_caption_jsonl")
        or str(PROJECT_ROOT / "results" / "eq" / args.dataset / "eq_by_clip.jsonl")
    )
    args.audio_dir = (
        args.audio_dir
        or dataset_config.get("audio_dir")
        or dataset_config.get("eval_audio_dir")
        or str(PROJECT_ROOT / "input" / args.dataset / "audio")
    )
    args.audio_output_root = args.audio_output_root or str(
        PROJECT_ROOT / "results" / "audioEmb" / args.dataset
    )
    args.text_output_root = args.text_output_root or str(
        PROJECT_ROOT / "results" / "testEmb" / args.dataset
    )
    args.merged_output_root = args.merged_output_root or str(
        PROJECT_ROOT / "results" / "mergedEmb" / args.dataset
    )
    args.retrieval_output_dir = args.retrieval_output_dir or str(
        PROJECT_ROOT / "results" / "retrieval" / args.dataset
    )
    args.text_batch_size = args.text_batch_size or int(
        get_execution_value(
            config,
            "text_batch_size",
            get_execution_value(config, "batch_size", 64),
        )
    )
    args.audio_batch_size = args.audio_batch_size or int(
        get_execution_value(
            config,
            "audio_batch_size",
            get_execution_value(config, "batch_size", 16),
        )
    )
    args.device = args.device or str(get_execution_value(config, "device", "auto"))
    args.ks = args.ks or list(
        config.get("evaluation", {}).get("top_k_list", [1, 5, 10])
    )
    args.plot_pool = args.plot_pool or (
        args.pools[0]
        if args.pools and "full_caption" not in args.pools
        else "full_caption"
    )
    args.dataset_config = dataset_config
    return args


def run_step(step_name: str, command: list[str]) -> None:
    print(f"\n[pipeline] {step_name}")
    print(f"[pipeline] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def add_selected_models(command: list[str], models: list[str] | None) -> None:
    if models:
        command.extend(["--models", *models])


def main() -> None:
    args = parse_args()
    hf_repo = args.dataset_config.get("hf_repo")
    caption_path = Path(args.caption_jsonl)
    audio_path = Path(args.audio_dir)
    hf_data_file_counts = args.dataset_config.get("hf_data_file_dataset_counts", [])
    available_hf_count = sum(int(count) for count in hf_data_file_counts)
    requested_hf_count = (
        min(args.limit, available_hf_count)
        if args.limit is not None and available_hf_count
        else args.limit
    )
    manifest_path = (
        Path(args.dataset_config.get("hf_output_dir") or caption_path.parent)
        / "manifest.json"
    )
    manifest_requires_refresh = False
    if hf_repo and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        exported_limit = manifest.get("limit")
        exported_count = int(manifest.get("num_examples", 0))
        manifest_requires_refresh = (
            args.limit is None and exported_limit is not None
        ) or (requested_hf_count is not None and exported_count < requested_hf_count)
    if hf_repo and (
        args.refresh_hf
        or manifest_requires_refresh
        or not caption_path.is_file()
        or not audio_path.is_dir()
    ):
        output_dir = Path(
            args.dataset_config.get("hf_output_dir") or caption_path.parent
        )
        prepare_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_hf_eq.py"),
            "--repo",
            str(hf_repo),
            "--split",
            str(args.dataset_config.get("hf_split", "test")),
            "--dataset-filter",
            str(args.dataset_config.get("hf_dataset_filter", args.dataset)),
            "--output-dir",
            str(output_dir),
            "--cache-dir",
            str(
                args.dataset_config.get(
                    "hf_cache_dir", PROJECT_ROOT / "input" / "hf_cache"
                )
            ),
            "--force",
        ]
        hf_data_files = args.dataset_config.get("hf_data_files", [])
        if requested_hf_count is not None and len(hf_data_files) == len(
            hf_data_file_counts
        ):
            selected_files: list[str] = []
            selected_count = 0
            for data_file, count in zip(hf_data_files, hf_data_file_counts):
                selected_files.append(str(data_file))
                selected_count += int(count)
                if selected_count >= requested_hf_count:
                    break
            hf_data_files = selected_files
        if hf_data_files:
            prepare_command.extend(["--data-files", *map(str, hf_data_files)])
        if requested_hf_count is not None:
            prepare_command.extend(["--limit", str(requested_hf_count)])
        run_step("0/6 prepare Hugging Face evaluation data", prepare_command)

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
    add_selected_models(merge_command, args.models)
    if args.limit is not None:
        merge_command.extend(["--limit", str(args.limit)])
    if not args.allow_partial:
        merge_command.append("--strict")

    retrieval_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_text_to_audio_retrieval.py"),
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--merged-root",
        args.merged_output_root,
        "--output-dir",
        args.retrieval_output_dir,
        "--ks",
        *[str(k) for k in args.ks],
    ]
    add_selected_models(retrieval_command, args.models)
    if args.pools:
        retrieval_command.extend(["--pools", *args.pools])

    run_step("1/5 source-text embeddings", source_command)
    run_step("2/5 generated-query embeddings", generated_command)
    run_step("3/5 audio embeddings", audio_command)
    run_step("4/5 merge embeddings", merge_command)
    run_step("5/5 Recall@K evaluation", retrieval_command)

    if not args.no_plot:
        plot_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "eval_plot_text_to_audio_recall.py"),
            "--config",
            args.config,
            "--dataset",
            args.dataset,
            "--input",
            str(Path(args.retrieval_output_dir) / "text_to_audio_recall.json"),
            "--output",
            str(Path(args.retrieval_output_dir) / "text_to_audio_recall_bar.png"),
            "--pool",
            args.plot_pool,
            "--ks",
            *[str(k) for k in args.ks],
        ]
        add_selected_models(plot_command, args.models)
        run_step("summary plot", plot_command)

    print("\n[pipeline] Evaluation complete")
    print(
        "[pipeline] Metrics: "
        f"{Path(args.retrieval_output_dir) / 'text_to_audio_recall.csv'}"
    )


if __name__ == "__main__":
    main()
