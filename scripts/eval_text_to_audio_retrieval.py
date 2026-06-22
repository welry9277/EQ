#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_embed_common import (
    DEFAULT_CONFIG_PATH,
    get_enabled_model_configs,
    load_eval_config,
)

ORIGINAL_POOL = "original"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute text-to-audio Recall@K from merged CLAP embeddings.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset", default="clotho")
    parser.add_argument("--merged-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--pools",
        nargs="+",
        default=None,
        help="Query pools to evaluate. Default: original and every generated query type.",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_eval_config(args.config)
    args.model_configs = get_enabled_model_configs(config, args.models)
    args.ks = args.ks or list(
        config.get("evaluation", {}).get("top_k_list", [1, 5, 10])
    )
    if not args.ks or any(k <= 0 for k in args.ks):
        raise ValueError("--ks must contain positive integers.")
    args.ks = sorted(set(args.ks))
    args.merged_root = args.merged_root or str(
        PROJECT_ROOT / "results" / "mergedEmb" / args.dataset
    )
    args.output_dir = args.output_dir or str(
        PROJECT_ROOT / "results" / "retrieval" / args.dataset
    )
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Merged embedding file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise ValueError(f"No merged rows found in {path}")
    return rows


def normalized_matrix(vectors: list[Any], label: str) -> np.ndarray:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} embeddings do not form a numeric matrix.") from exc
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(
            f"{label} embeddings must have shape [N, D], got {matrix.shape}."
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{label} embeddings contain a zero vector.")
    return matrix / norms


def available_pools(rows: list[dict[str, Any]]) -> list[str]:
    generated: set[str] = set()
    for row in rows:
        queries = row.get("generated_queries", {})
        if isinstance(queries, dict):
            generated.update(str(key) for key in queries)
    return [ORIGINAL_POOL, *sorted(generated)]


def query_vectors_for_pool(
    rows: list[dict[str, Any]],
    pool: str,
) -> tuple[list[str], list[Any], list[str]]:
    audio_ids: list[str] = []
    vectors: list[Any] = []
    texts: list[str] = []
    for row in rows:
        audio_id = str(row.get("audio_id", "")).strip()
        if pool == ORIGINAL_POOL:
            query = row.get("source", {})
        else:
            generated = row.get("generated_queries", {})
            query = generated.get(pool, {}) if isinstance(generated, dict) else {}
        if not isinstance(query, dict) or not query.get("emb"):
            continue
        audio_ids.append(audio_id)
        vectors.append(query["emb"])
        texts.append(str(query.get("text", "")))
    return audio_ids, vectors, texts


def top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, scores.shape[1])
    if top_k == scores.shape[1]:
        return np.argsort(-scores, axis=1)[:, :top_k]
    candidates = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def evaluate_pool(
    rows: list[dict[str, Any]],
    pool: str,
    ks: list[int],
    query_batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audio_ids = [str(row.get("audio_id", "")).strip() for row in rows]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Merged embeddings contain duplicate audio_id values.")
    audio_index = {audio_id: index for index, audio_id in enumerate(audio_ids)}
    audio_matrix = normalized_matrix(
        [row.get("audio_emb", []) for row in rows],
        "Audio",
    )

    query_ids, query_vectors, query_texts = query_vectors_for_pool(rows, pool)
    if not query_ids:
        raise ValueError(f"No query embeddings found for pool {pool!r}.")
    unknown_ids = sorted(set(query_ids) - set(audio_index))
    if unknown_ids:
        raise ValueError(
            f"Queries reference audio IDs missing from the audio pool: {unknown_ids[:5]}"
        )
    query_matrix = normalized_matrix(query_vectors, f"{pool} query")
    if query_matrix.shape[1] != audio_matrix.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch for {pool}: "
            f"text={query_matrix.shape[1]}, audio={audio_matrix.shape[1]}"
        )

    max_k = max(ks)
    ranks: list[int] = []
    retrieval_rows: list[dict[str, Any]] = []
    for start in range(0, len(query_ids), query_batch_size):
        end = start + query_batch_size
        batch_scores = query_matrix[start:end] @ audio_matrix.T
        batch_top = top_indices(batch_scores, max_k)
        for local_index, score_row in enumerate(batch_scores):
            query_index = start + local_index
            correct_index = audio_index[query_ids[query_index]]
            correct_score = score_row[correct_index]
            rank = int(np.count_nonzero(score_row > correct_score) + 1)
            ranks.append(rank)
            ranked_ids = [audio_ids[index] for index in batch_top[local_index]]
            retrieval_rows.append(
                {
                    "query_audio_id": query_ids[query_index],
                    "query_pool": pool,
                    "query_text": query_texts[query_index],
                    "rank": rank,
                    "top_audio_ids": ranked_ids,
                }
            )

    rank_array = np.asarray(ranks)
    metrics = {f"R@{k}": float(np.mean(rank_array <= k)) for k in ks}
    return (
        {
            "n_queries": len(query_ids),
            "n_audio": len(audio_ids),
            "text_to_audio": metrics,
        },
        retrieval_rows,
    )


def write_retrieval_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(
    path: Path,
    summary: dict[str, Any],
    ks: list[int],
) -> None:
    fieldnames = ["model", "pool", "n_queries", "n_audio", *[f"R@{k}" for k in ks]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, pools in summary["models"].items():
            for pool, values in pools.items():
                row = {
                    "model": model_name,
                    "pool": pool,
                    "n_queries": values["n_queries"],
                    "n_audio": values["n_audio"],
                    **values["text_to_audio"],
                }
                writer.writerow(row)


def print_summary(summary: dict[str, Any], ks: list[int]) -> None:
    headers = ["model", "pool", "queries", *[f"R@{k}" for k in ks]]
    rows: list[list[str]] = []
    for model_name, pools in summary["models"].items():
        for pool, values in pools.items():
            rows.append(
                [
                    model_name,
                    pool,
                    str(values["n_queries"]),
                    *[f"{values['text_to_audio'][f'R@{k}']:.4f}" for k in ks],
                ]
            )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "ks": args.ks,
        "models": {},
    }

    for model_name in args.model_configs:
        merged_path = Path(args.merged_root) / model_name / "merged.jsonl"
        rows = read_jsonl(merged_path)
        pools = args.pools or available_pools(rows)
        summary["models"][model_name] = {}
        for pool in pools:
            metrics, retrieval_rows = evaluate_pool(
                rows=rows,
                pool=pool,
                ks=args.ks,
                query_batch_size=args.query_batch_size,
            )
            summary["models"][model_name][pool] = metrics
            write_retrieval_rows(
                output_dir / f"{model_name}_{pool}_retrieval.jsonl",
                retrieval_rows,
            )

    json_path = output_dir / "text_to_audio_recall.json"
    csv_path = output_dir / "text_to_audio_recall.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_summary_csv(csv_path, summary, args.ks)
    print_summary(summary, args.ks)
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {csv_path}")


if __name__ == "__main__":
    main()
