"""
Semantic mapping from free-form audio captions to fixed label prototypes.

This module implements a prototype-based assignment pipeline:
1. Build one embedding prototype per VGGSound category from prompt templates
2. Encode AudioCaps captions into the same embedding space
3. Compute cosine similarity with normalized embeddings
4. Assign each caption to the top-1 category
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_TEMPLATES = (
    "{category}",
    "sound of {category}",
    "a recording of {category}",
    "audio of {category}",
)


def _mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@dataclass
class SemanticMapper:
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    batch_size: int = 32
    max_length: int = 128

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 1024))
            return np.empty((0, hidden_size), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)

    def build_category_prototypes(
        self,
        categories: Sequence[str],
        templates: Sequence[str] = DEFAULT_TEMPLATES,
    ) -> tuple[list[str], np.ndarray]:
        if not categories:
            raise ValueError("categories must not be empty")

        prototype_vectors: list[np.ndarray] = []
        for category in categories:
            template_texts = [template.format(category=category) for template in templates]
            template_embeddings = self.encode_texts(template_texts)
            prototype = template_embeddings.mean(axis=0)
            prototype /= np.linalg.norm(prototype) + 1e-12
            prototype_vectors.append(prototype.astype(np.float32))

        return list(categories), np.vstack(prototype_vectors)

    def assign_captions(
        self,
        captions: Sequence[str],
        categories: Sequence[str],
        prototype_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        if not captions:
            return []
        if len(categories) != len(prototype_embeddings):
            raise ValueError("categories and prototype_embeddings must have the same length")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        caption_embeddings = self.encode_texts(captions)
        similarities = caption_embeddings @ prototype_embeddings.T

        effective_top_k = min(top_k, len(categories))
        topk_indices = np.argsort(-similarities, axis=1)[:, :effective_top_k]

        results: list[dict] = []
        for row_idx, caption in enumerate(captions):
            assigned_idx = int(topk_indices[row_idx, 0])
            top_matches = [
                {
                    "category": categories[int(category_idx)],
                    "similarity": float(similarities[row_idx, int(category_idx)]),
                }
                for category_idx in topk_indices[row_idx]
            ]
            results.append(
                {
                    "caption": caption,
                    "assigned_category": categories[assigned_idx],
                    "similarity": float(similarities[row_idx, assigned_idx]),
                    "top_k": top_matches,
                }
            )

        return results


def build_category_prototypes(
    categories: Sequence[str],
    mapper: Optional[SemanticMapper] = None,
    templates: Sequence[str] = DEFAULT_TEMPLATES,
) -> tuple[list[str], np.ndarray]:
    mapper = mapper or SemanticMapper()
    return mapper.build_category_prototypes(categories=categories, templates=templates)


def assign_captions(
    captions: Sequence[str],
    categories: Sequence[str],
    prototype_embeddings: np.ndarray,
    mapper: Optional[SemanticMapper] = None,
    top_k: int = 5,
) -> list[dict]:
    mapper = mapper or SemanticMapper()
    return mapper.assign_captions(
        captions=captions,
        categories=categories,
        prototype_embeddings=prototype_embeddings,
        top_k=top_k,
    )


def load_json_list(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"{path} must contain a JSON list of strings")
    return data


def save_assignments(assignments: Iterable[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(list(assignments), handle, indent=2, ensure_ascii=False)
