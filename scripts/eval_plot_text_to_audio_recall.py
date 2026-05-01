#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from scripts._eval_embed_common import get_enabled_model_configs, load_eval_config

DEFAULT_MODEL_ORDER = ("msclap", "laion", "mga", "m2d")
DEFAULT_KS = (1, 5, 10)
DEFAULT_COLORS = ("#ff3b30", "#34c759", "#007aff")
DISPLAY_NAMES = {
    "msclap": "MSCLAP",
    "laion": "LAION CLAP",
    "mga": "MGA CLAP",
    "m2d": "M2D CLAP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model-wise text-to-audio Recall@K bar chart."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="YAML config containing model and evaluation settings.",
    )
    parser.add_argument(
        "--dataset",
        default="clotho",
        help="Dataset name used for default input/output paths, e.g. clotho or mecats.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to text_to_audio_recall.json",
    )
    parser.add_argument(
        "--pool",
        default="full_caption",
        help="Which query pool to visualize, e.g. original, full_caption, statement, command, key_phrase, indirect, question",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model order on the x-axis.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=None,
        help="Recall@K values to plot in order.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path.",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Dataset label used in the chart title.",
    )
    args = parser.parse_args()
    config = load_eval_config(args.config)
    args.input = args.input or str(PROJECT_ROOT / "results" / "retrieval" / args.dataset / "text_to_audio_recall.json")
    args.output = args.output or str(PROJECT_ROOT / "results" / "retrieval" / args.dataset / "text_to_audio_recall_bar.png")
    args.dataset_label = args.dataset_label or args.dataset.upper()
    args.models = args.models or list(get_enabled_model_configs(config))
    args.ks = args.ks or list(config.get("evaluation", {}).get("top_k_list", DEFAULT_KS))
    return args


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_scores(
    summary: dict[str, Any],
    pool: str,
    models: list[str],
    ks: list[int],
) -> tuple[list[str], list[float], list[str]]:
    model_data = summary.get("models", {})
    labels: list[str] = []
    scores: list[float] = []
    series_names: list[str] = []

    for model_name in models:
        if model_name not in model_data:
            raise KeyError(f"Model '{model_name}' not found in summary.")
        pool_data = model_data[model_name].get(pool)
        if pool_data is None:
            raise KeyError(f"Pool '{pool}' not found for model '{model_name}'.")
        recall_map = pool_data.get("text_to_audio", {})

        display_name = DISPLAY_NAMES.get(model_name, model_name.upper())
        for k in ks:
            key = f"R@{k}"
            if key not in recall_map:
                raise KeyError(f"Metric '{key}' not found for model '{model_name}' pool '{pool}'.")
            labels.append(f"{display_name}\nR@{k}")
            scores.append(float(recall_map[key]))
            series_names.append(key)

    return labels, scores, series_names


def plot_bar_chart(
    labels: list[str],
    scores: list[float],
    series_names: list[str],
    models: list[str],
    ks: list[int],
    output_path: Path,
    pool: str,
    dataset_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    color_map = {f"R@{k}": DEFAULT_COLORS[idx % len(DEFAULT_COLORS)] for idx, k in enumerate(ks)}
    colors = [color_map[name] for name in series_names]
    intra_group_gap = 1.0
    inter_group_gap = 1.25
    bar_width = 1.0

    x: list[float] = []
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    cursor = 0.0

    for model_name in models:
        group_positions = []
        for _ in ks:
            x.append(cursor)
            group_positions.append(cursor)
            cursor += intra_group_gap
        tick_positions.append(float(np.mean(group_positions)))
        tick_labels.append(DISPLAY_NAMES.get(model_name, model_name.upper()))
        cursor += inter_group_gap

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    bars = ax.bar(x, scores, color=colors, width=bar_width, edgecolor="none", linewidth=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[f"R@{k}"], label=f"Recall@{k}")
        for k in ks
    ]

    ax.set_xticks(tick_positions, tick_labels, fontsize=11)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0.0, min(1.0, max(scores) + 0.12))
    ax.set_title(f"{dataset_label} Text-to-Audio Retrieval", fontsize=13, pad=12)
    ax.legend(
        handles=legend_handles,
        title="Metric",
        frameon=False,
        ncol=len(ks),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=10,
        title_fontsize=10,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    for xpos, score in zip(x, scores):
        ax.text(
            xpos,
            score + 0.012,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = load_summary(Path(args.input))
    labels, scores, series_names = collect_scores(
        summary=summary,
        pool=args.pool,
        models=args.models,
        ks=args.ks,
    )
    plot_bar_chart(
        labels=labels,
        scores=scores,
        series_names=series_names,
        models=args.models,
        ks=args.ks,
        output_path=Path(args.output),
        pool=args.pool,
        dataset_label=args.dataset_label,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
