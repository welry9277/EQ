#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.makeuiq import load_audiocaps
from uiq_generation.semantic_mapping import DEFAULT_MODEL_NAME, SemanticMapper, load_json_list, save_assignments


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each VGGSound category, find the single most similar AudioCaps caption "
            "using BGE text embeddings."
        )
    )
    parser.add_argument(
        "--captions-csv",
        required=True,
        help="Path to the AudioCaps CSV file.",
    )
    parser.add_argument(
        "--categories-json",
        required=True,
        help="Path to a JSON file containing a list of VGGSound category strings.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to save the structured category-to-caption results.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split label passed into load_audiocaps(). Default: test",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model name. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top captions to retain per category for analysis.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cpu" or "cuda".',
    )

    args = parser.parse_args()

    records = load_audiocaps(args.captions_csv, split=args.split)
    categories = load_json_list(args.categories_json)

    captions: list[str] = []
    caption_metadata: list[dict] = []
    for record in records:
        for caption in record["original_captions"]:
            captions.append(caption)
            caption_metadata.append(
                {
                    "audio_id": record["audio_id"],
                    "dataset": record["dataset"],
                    "dataset_slug": record["dataset_slug"],
                    "metadata": record["metadata"],
                }
            )

    mapper = SemanticMapper(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    category_names, prototype_embeddings = mapper.build_category_prototypes(categories)
    caption_embeddings = mapper.encode_texts(captions)

    similarities = prototype_embeddings @ caption_embeddings.T
    effective_top_k = min(args.top_k, len(captions))
    topk_indices = np.argsort(-similarities, axis=1)[:, :effective_top_k]

    results = []
    for category_idx, category_name in enumerate(category_names):
        best_caption_idx = int(topk_indices[category_idx, 0])
        top_matches = []
        for caption_idx in topk_indices[category_idx]:
            caption_idx = int(caption_idx)
            top_matches.append(
                {
                    "caption": captions[caption_idx],
                    "audio_id": caption_metadata[caption_idx]["audio_id"],
                    "similarity": float(similarities[category_idx, caption_idx]),
                }
            )

        results.append(
            {
                "category": category_name,
                "matched_caption": captions[best_caption_idx],
                "audio_id": caption_metadata[best_caption_idx]["audio_id"],
                "dataset": caption_metadata[best_caption_idx]["dataset"],
                "dataset_slug": caption_metadata[best_caption_idx]["dataset_slug"],
                "metadata": caption_metadata[best_caption_idx]["metadata"],
                "similarity": float(similarities[category_idx, best_caption_idx]),
                "top_k_captions": top_matches,
            }
        )

    save_assignments(results, args.output_json)
    print(f"[INFO] Loaded {len(records)} audio records from AudioCaps")
    print(f"[INFO] Expanded to {len(captions)} captions")
    print(f"[INFO] Built {len(category_names)} category prototypes")
    print(f"[INFO] Saved {len(results)} category-to-caption matches to {Path(args.output_json)}")


if __name__ == "__main__":
    main()
