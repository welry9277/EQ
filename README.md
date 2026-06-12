# EQ Generation Pipeline

LLM-based toolkit for generating **EQ** (Extended Query) data for text-to-audio retrieval from AudioCaps-style caption CSVs or MeCAT JSON captions. Outputs are validated and saved as JSONL.

## Project Overview

`EQ` generates six expressive query variants per audio clip from AudioCaps, Clotho, or MeCAT captions, then evaluates how well each variant retrieves its corresponding audio with four CLAP-family models: LAION, MGA, MS-CLAP, and M2D.

Query types: `key_phrase`, `statement`, `question`, `command`, `indirect`, `full_caption`.

## Dataset and Follow-up Work

[![EQ Dataset](https://img.shields.io/badge/HuggingFace-EQ%20Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/msnowchanj/EQ)
[![CORA Experiments](https://img.shields.io/badge/GitHub-CORA%20Experiments-181717?logo=github)](https://github.com/EMNLP-2026/emnlp2026-CORA-diagnosis)
[![Paper](https://img.shields.io/badge/Paper-Google%20Docs-34A853?logo=googledrive&logoColor=white)](https://docs.google.com/document/d/16uvxyb-CTYksIGxQiB4Y6OsZYLhsZhOe/edit?usp=sharing&ouid=103099717435348800308&rtpof=true&sd=true)

The generated EQ dataset is available on Hugging Face. This project was later extended into **CORA**, a follow-up study submitted to EMNLP 2026.

## Installation

Requirements:

- Python `>=3.9`
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key for generation

Clone and install:

```bash
git clone https://github.com/welry9277/EQ.git
cd EQ
uv sync --frozen
```

To use the Hugging Face helper script below, also install the optional `datasets` extra:

```bash
uv sync --frozen --extra hf
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Default model settings live in [`config.yaml`](config.yaml) under the `model` key:

```yaml
model:
  source_model: gpt-4o-mini
  regen_model: gpt-4o-mini
  backend: gpt
  temperature: 0.7
  batch_size: 10
  max_tokens: 100
```

## Data Preparation

Use [`scripts/prepare_data.py`](scripts/prepare_data.py) to create input directories and download supported metadata:

- `clotho`: downloads official `*_evaluation` metadata from [Clotho](https://zenodo.org/records/3490684)
- `audiocaps`: downloads official `train.csv`, `val.csv`, and `test.csv` into `input/audiocaps/` from [cdjkim/audiocaps](https://github.com/cdjkim/audiocaps/tree/master/dataset)
- `mecat`: creates `input/mecat/json_files/` for manual placement

### Preparing MeCAT files

MeCAT is not downloaded automatically. Download the ZIP files from GitHub, place them under `input/mecat/`, and extract them there.

Example:

```bash
mkdir -p input/mecat

# After placing the downloaded ZIP files under input/mecat/
mv ~/Downloads/json_files.zip input/mecat/json_files.zip
mv ~/Downloads/flac_files.zip input/mecat/flac_files.zip

# With unzip
unzip input/mecat/json_files.zip -d input/mecat/
unzip input/mecat/flac_files.zip -d input/mecat/

# Without unzip, use Python
python3 -c "import zipfile; zipfile.ZipFile('input/mecat/json_files.zip').extractall('input/mecat/')"
python3 -c "import zipfile; zipfile.ZipFile('input/mecat/flac_files.zip').extractall('input/mecat/')"
```

If extraction places `.json` and `.flac` files directly under `input/mecat/`, move them into separate directories:

```bash
mkdir -p input/mecat/json_files input/mecat/flac_files
find input/mecat -maxdepth 1 -type f -name '*.json' -exec mv -t input/mecat/json_files {} +
find input/mecat -maxdepth 1 -type f -name '*.flac' -exec mv -t input/mecat/flac_files {} +
```

After cleanup, the directory should look like this:

```text
input/mecat/
├── flac_files.zip
├── flac_files/
│   ├── <audio_id>.flac
│   ├── <audio_id>.flac
│   └── ...
├── json_files.zip
└── json_files/
    ├── <audio_id>.json
    ├── <audio_id>.json
    └── ...
```

`makeeq.py` reads the `short` field from each JSON file in `input/mecat/json_files/`.

### AudioCaps via Hugging Face (`datasets`)

For **official** train/val/test caption CSVs from the paper repo, use `prepare_data.py --dataset audiocaps` (no `datasets` install required).

The snippet below uses the **[Hugging Face Hub](https://huggingface.co/datasets)** and the [`datasets`](https://huggingface.co/docs/datasets) library. The repository ID **`d0rj/audiocaps`** is a mirror of **AudioCaps**, not Clotho. Clotho is a different dataset; use `prepare_data.py --dataset clotho` or another Hub ID for Clotho.

After `uv sync --extra hf`, export a split to the same CSV columns `makeeq` expects (`youtube_id`, `start_time`, `caption`):

```bash
./scripts/download_audiocaps_hf.py --split test --output input/audiocaps/test.csv
```

Splits on the Hub are named `train`, `validation`, and `test`.

### One row per clip

Official CSVs can contain several caption lines per `(youtube_id, start_time)`. To collapse them into a single row, keeping the first `audiocap_id` and joining captions with ` | ` by default:

```bash
# Defaults: input/audiocaps/test.csv -> input/audiocaps/test_merged.csv
./scripts/merge_audiocaps_captions.py

./scripts/merge_audiocaps_captions.py \
  --input input/audiocaps/val.csv \
  --output input/audiocaps/val_merged.csv
```

Use `--separator` to change the join string. Note that `makeeq` already groups by clip when reading multi-row CSVs; merging is only needed for tools that require one row per clip.

## EQ Query Types

The pipeline generates six variants per clip:

- `key_phrase`
- `statement`
- `question`
- `command`
- `indirect`
- `full_caption`

Generation behavior:

- **`full_caption`**: prompted with the **full caption list** for the clip; `original_captions` in JSONL stores that full list.
- **`key_phrase`, `statement`, `question`, `command`, `indirect`**: prompted with the **middle caption only** (index `len // 2`); `original_captions` is a one-element list containing that caption. Metadata adds `eq_reference`, `middle_caption_index`, and `full_caption_count`.
- **MeCAT**: the loader uses the JSON `short` field as the caption list. That means `full_caption` uses all `short` captions, and every other query type uses only the middle `short` caption.

## Generate EQ

Use [`scripts/makeeq.py`](scripts/makeeq.py):

```bash
./scripts/makeeq.py \
  --dataset audiocaps \
  --captions-csv input/audiocaps/test.csv \
  --output-dir results/eq/audiocaps_test \
  --split test
```

MeCAT example:

```bash
./scripts/makeeq.py \
  --dataset mecat \
  --captions-path input/mecat/json_files \
  --output-dir results/eq/mecat_default \
  --split default
```

Optional arguments:

- `--dataset`: original dataset source to convert
- `--split`: split label stored in metadata, default `test`
- `--num-queries`: cap the number of clips after grouping
- `--config`: custom config file

Generated files:

- `eq_key_phrase.jsonl`
- `eq_statement.jsonl`
- `eq_question.jsonl`
- `eq_command.jsonl`
- `eq_indirect.jsonl`
- `eq_full_caption.jsonl`

Validation logs are written to `eq_validation.log`.

### Merge EQ types per clip (one file)

After generation, combine the six `eq_*.jsonl` files into **one record per `audio_id`** with `original_captions` and all six strings under `generated_queries`:

```bash
./scripts/merge_eq_by_clip.py --input-dir results/eq/test_sample5
# writes results/eq/test_sample5/eq_by_clip.jsonl (pretty-printed blocks; use --compact for strict one-line JSONL)
```

## Output Schema

Each JSONL line follows this shape:

```json
{
  "audio_id": "...",
  "dataset": "...",
  "dataset_slug": "...",
  "query_type": "...",
  "generated_query": "...",
  "original_captions": ["..."],
  "metadata": {},
  "source_model": "...",
  "regen_model": "..."
}
```

## Text-to-Audio Retrieval Evaluation

This evaluation uses a caption JSONL and audio files to build audio and text embeddings for each model, then computes text-to-audio retrieval performance. The scripts default to `clotho`, and the same workflow can be reused for another dataset such as `mecats` by passing `--dataset mecats`.

This workflow reads the project `config*.yaml` files. Model paths, model IDs, enabled models, device, and default batch size come from `--config` (default: `config.yaml`). Dataset paths can come from the config when present, or from `--dataset` and explicit path arguments.

### 1. Prepare inputs

Default input paths for a dataset named `{dataset}`:

- Caption JSONL: `input/captions/{dataset}/eq_by_clip.jsonl`
- Audio directory: `input/{dataset}/audio`

The caption JSONL should contain `audio_id`, `file_name`, the original caption, and generated query text. If `file_name` is empty, rows are matched by `audio_id`.

### 2. Generate and merge embeddings

Generate source text embeddings, generated text embeddings, and audio embeddings, then merge them into one `merged.jsonl` file per model.

```bash
uv run python scripts/eval_pipeline.py
```

To run with one of the per-model configs:

```bash
uv run python scripts/eval_pipeline.py --config config_mga.yaml
```

For MECATS, pass the dataset name:

```bash
uv run python scripts/eval_pipeline.py --config config.yaml --dataset mecats
```

For a quick check on a subset of models, use `--models` and `--limit`:

```bash
uv run python scripts/eval_pipeline.py --config config.yaml --dataset mecats --models msclap laion --limit 100
```

Main output paths:

- Audio embedding: `results/audioEmb/{dataset}/{model}/emb.jsonl`
- Text embedding: `results/testEmb/{dataset}/{model}/source_emb.jsonl`, `results/testEmb/{dataset}/{model}/generated_emb.jsonl`
- Merged embedding: `results/mergedEmb/{dataset}/{model}/merged.jsonl`

### 3. Compute retrieval metrics

Compute text-to-audio Recall@K from the merged embeddings.

```bash
uv run python scripts/eval_text_to_audio_retrieval.py
```

By default, the script evaluates all query pools: `original`, `full_caption`, `statement`, `command`, `key_phrase`, `indirect`, and `question`. It saves Recall@1/5/10 for each pool.

Main output paths:

- Metric summary: `results/retrieval/{dataset}/text_to_audio_recall.json`
- Per-query retrieval results: `results/retrieval/{dataset}/{model}_{pool}_retrieval.jsonl`

To evaluate only selected models or query pools:

```bash
uv run python scripts/eval_text_to_audio_retrieval.py \
  --config config.yaml \
  --dataset mecats \
  --models msclap laion mga m2d \
  --pools full_caption statement command \
  --ks 1 5 10
```

### 4. Generate a recall bar plot

Create a model-wise Recall@K bar plot from the metric JSON.

```bash
uv run python scripts/eval_plot_text_to_audio_recall.py
```

Default output:

```text
results/retrieval/{dataset}/text_to_audio_recall_bar.png
```

To plot another query pool or change the dataset label in the chart title:

```bash
uv run python scripts/eval_plot_text_to_audio_recall.py \
  --config config.yaml \
  --dataset mecats \
  --pool statement \
  --dataset-label MECATS
```

### 5. Interpret results

- `R@1`: the fraction of queries where the correct audio is ranked first
- `R@5`: the fraction of queries where the correct audio appears in the top 5
- `R@10`: the fraction of queries where the correct audio appears in the top 10

The `original` pool uses the original caption as the query. The other pools use different generated query types. Comparing Recall across generated query pools helps identify which text formats are easier or harder for each model in audio retrieval.

## Repository Layout

```text
EQ/
├── config.yaml                   # EQ generation + CLAP eval settings
├── config_laion.yaml             # LAION-only eval config
├── config_msclap.yaml            # MS-CLAP-only eval config
├── config_mga.yaml               # MGA-CLAP-only eval config
├── config_m2d.yaml               # M2D-only eval config
├── eq_generation/                # EQ generation package
│   ├── __init__.py
│   ├── data.py
│   ├── query_types.py
│   └── generators/
│       ├── base.py
│       ├── factory.py
│       ├── gpt_generator.py
│       └── prompts.py
├── src/clap_eval/models/         # CLAP model wrappers
│   ├── base.py
│   ├── laion.py
│   ├── mga.py
│   ├── m2d.py
│   └── msclap.py
├── scripts/
│   ├── makeeq.py                 # Main EQ generation script
│   ├── prepare_data.py
│   ├── download_audiocaps_hf.py
│   ├── merge_audiocaps_captions.py
│   ├── merge_eq_by_clip.py
│   ├── eval_pipeline.py          # Orchestrate all eval steps
│   ├── eval_extract_audio_embeddings.py
│   ├── eval_extract_source_text_embeddings.py
│   ├── eval_extract_generated_text_embeddings.py
│   ├── eval_merge_embeddings.py
│   ├── eval_text_to_audio_retrieval.py
│   └── eval_plot_text_to_audio_recall.py
└── input/                        # Not tracked by git; create locally
    ├── audiocaps/
    ├── clotho/
    └── mecat/
```
