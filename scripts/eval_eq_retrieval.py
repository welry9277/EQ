#!/usr/bin/env python
"""
EQ Dataset Retrieval Evaluation Script
- Loads EQ dataset from HuggingFace
- Extracts audio & text embeddings for each CLAP model
- Computes cosine similarity → saves rank@200 results as JSONL

Usage (from project root):
    python scripts/eval_eq_retrieval.py --config config.yaml
    python scripts/eval_eq_retrieval.py --config config_laion.yaml --models laion
"""
from __future__ import annotations

import argparse
import json
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Query types in EQ
QUERY_TYPES = ["full_caption", "key_phrase", "statement", "question", "command", "indirect"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_cfg: dict, device: str):
    """Instantiate the correct BaseClapModel subclass via get_model."""
    from src.clap_eval.models import get_model
    return get_model(name=model_cfg["name"], config=model_cfg, device=device)


def audio_array_from_example(example: dict) -> tuple[np.ndarray, int]:
    """Extract (array, sr) from a HuggingFace audio column dict."""
    audio = example["audio"]
    arr = np.array(audio["array"], dtype=np.float32)
    sr  = int(audio["sampling_rate"])
    return arr, sr


def batch_iter(lst: list, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_audio_embeddings(
    model,
    examples: list[dict],
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Returns (embeddings [N, D], audio_ids [N])."""
    audio_ids = [ex["audio_id"] for ex in examples]
    all_embs = []
    for batch in tqdm(list(batch_iter(examples, batch_size)), desc=f"[{model.name}] audio"):
        arrays, sr = [], None
        for ex in batch:
            arr, sr = audio_array_from_example(ex)
            arrays.append(arr)
        emb = model.get_audio_embedding(arrays, sr)
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0), audio_ids


def extract_text_embeddings(
    model,
    examples: list[dict],
    query_type: str,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Returns (embeddings [N, D], audio_ids [N]) for one query type."""
    audio_ids = [ex["audio_id"] for ex in examples]
    texts     = [ex[query_type] for ex in examples]
    all_embs = []
    for batch_texts in tqdm(
        list(batch_iter(texts, batch_size)),
        desc=f"[{model.name}] text/{query_type}",
    ):
        emb = model.get_text_embedding(batch_texts)
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0), audio_ids


# ── Ranking ───────────────────────────────────────────────────────────────────

def compute_ranks(
    text_embs: np.ndarray,   # [N, D]
    audio_embs: np.ndarray,  # [N, D]
    audio_ids: list[str],
    top_k: int = 200,
) -> list[list[str]]:
    """
    For each query i, rank all audio clips by cosine similarity (desc).
    Assumes embeddings are already L2-normalised.
    """
    sims = text_embs @ audio_embs.T          # [N_text, N_audio]
    top_indices = np.argsort(-sims, axis=1)[:, :top_k]
    return [[audio_ids[j] for j in row] for row in top_indices]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device     = args.device     or config.get("execution", {}).get("device", "cuda")
    batch_size = args.batch_size or config.get("execution", {}).get("batch_size", 16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EQ dataset …")
    dataset  = load_dataset("msnowchanj/EQ", cache_dir="input/EQ")
    examples = list(dataset[args.split])
    print(f"  {args.split} split: {len(examples)} examples")

    model_cfgs = [m for m in config["models"] if m.get("enabled", True)]
    if args.models:
        model_cfgs = [m for m in model_cfgs if m["name"] in args.models]

    for model_cfg in model_cfgs:
        model_name = model_cfg["name"]
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        model = build_model(model_cfg, device)

        # Audio embeddings — computed once, reused for all query types
        audio_embs, audio_ids_ordered = extract_audio_embeddings(
            model, examples, batch_size
        )

        for query_type in QUERY_TYPES:
            print(f"\n  Query type: {query_type}")
            text_embs, query_audio_ids = extract_text_embeddings(
                model, examples, query_type, batch_size
            )

            ranked_lists = compute_ranks(text_embs, audio_embs, audio_ids_ordered, args.top_k)

            out_path = output_dir / model_name / f"{query_type}_rank{args.top_k}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                for audio_id, ranked in zip(query_audio_ids, ranked_lists):
                    record = {
                        "query_id":         audio_id,  # ground truth = self
                        "query_type":       query_type,
                        "model":            model_name,
                        "ranked_audio_ids": ranked,
                    }
                    f.write(json.dumps(record) + "\n")

            print(f"    Saved → {out_path}")

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EQ retrieval rank@K evaluation")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--output-dir", default="results/eq_ranks")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--top-k",      type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device",     default=None)
    parser.add_argument("--models",     nargs="+", default=None,
                        help="Subset of models, e.g. --models laion mga")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())