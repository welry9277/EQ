# EQ Generation and Retrieval Evaluation

Generate Extended Query (EQ) variants from **AudioCaps**, **Clotho**, or **MeCAT** captions and evaluate text-to-audio retrieval with CLAP-family models.

The complete evaluation pipeline produces Recall@1, Recall@5, and Recall@10 from one command.

[![EQ Dataset](https://img.shields.io/badge/HuggingFace-EQ%20Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/msnowchanj/EQ)
[![CORA Experiments](https://img.shields.io/badge/GitHub-CORA%20Experiments-181717?logo=github)](https://github.com/EMNLP-2026/emnlp2026-CORA-diagnosis)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What this repository does

For each audio clip, EQ generates:

- `key_phrase`
- `statement`
- `question`
- `command`
- `indirect`
- `full_caption`

It then evaluates the original caption and generated queries against the corresponding audio with LAION CLAP, MS-CLAP, MGA-CLAP, or M2D-CLAP.

## Quick start

Requirements:

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for query generation

```bash
git clone https://github.com/welry9277/EQ.git
cd EQ
uv sync --frozen
cp .env.example .env
```

Set your key in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

## 1. Prepare data

### AudioCaps

Download the official AudioCaps test captions:

```bash
uv run python scripts/prepare_data.py --dataset audiocaps
```

This downloads [`cdjkim/audiocaps` test.csv](https://github.com/cdjkim/audiocaps/blob/master/dataset/test.csv) to:

```text
input/audiocaps/test.csv
```

The file contains 4,875 rows grouped into 975 clips. Every `(youtube_id, start_time)` group has exactly five captions. The loader validates this before generation.

### Clotho

Download an official caption CSV:

```bash
uv run python scripts/prepare_data.py \
  --dataset clotho \
  --clotho-split evaluation
```

The script writes:

```text
input/clotho/clotho_captions_evaluation.csv
```

Download the matching audio archive from the [official Clotho 2.1 record](https://zenodo.org/records/4783391), extract it, and place the audio files directly under:

```text
input/clotho/audio/
├── Santa Motor.wav
├── Radio Garble.wav
└── ...
```

The loader supports both:

- Official wide CSVs with `file_name`, `caption_1`, ..., `caption_5`
- Long CSVs with repeated `file_name`, `caption` rows

### MeCAT

Create the expected directories:

```bash
uv run python scripts/prepare_data.py --dataset mecat
```

Place MeCAT files as follows:

```text
input/mecat/
├── json_files/
│   ├── <audio_id>.json
│   └── ...
└── flac_files/
    ├── <audio_id>.flac
    └── ...
```

Each JSON file must contain a `short` field. It may be either a string or a list:

```json
{
  "short": [
    "Persistent engine noise with heavy distortion",
    "Continuous mechanical rumbling"
  ],
  "domain": "vehicle"
}
```

Nested directories under `json_files/` are supported. Unless the JSON contains `file_name` or `audio_file`, the matching audio file is assumed to be `<audio_id>.flac`.

## 2. Generate EQ queries

### AudioCaps full captions

AudioCaps defaults to the `full_caption` query type. The five captions for each clip are passed together to the model:

```bash
uv run python scripts/makeeq.py \
  --dataset audiocaps \
  --captions-path input/audiocaps/test.csv \
  --output-dir results/eq/audiocaps \
  --split test
```

The output is:

```text
results/eq/audiocaps/eq_full_caption.jsonl
```

To generate additional query types explicitly:

```bash
uv run python scripts/makeeq.py \
  --dataset audiocaps \
  --captions-path input/audiocaps/test.csv \
  --output-dir results/eq/audiocaps \
  --query-types full_caption question command
```

### Clotho

```bash
uv run python scripts/makeeq.py \
  --dataset clotho \
  --captions-path input/clotho/clotho_captions_evaluation.csv \
  --output-dir results/eq/clotho \
  --split evaluation
```

### MeCAT

```bash
uv run python scripts/makeeq.py \
  --dataset mecat \
  --captions-path input/mecat/json_files \
  --output-dir results/eq/mecat \
  --split default
```

Use `--num-queries 20` for a small generation test. Model settings are in [`configs/config.yaml`](configs/config.yaml).

Clotho and MeCAT create six files by default:

```text
eq_key_phrase.jsonl
eq_statement.jsonl
eq_question.jsonl
eq_command.jsonl
eq_indirect.jsonl
eq_full_caption.jsonl
```

API failures and empty generated queries cause validation to fail instead of silently producing incomplete data.

## 3. Merge query types

Merge the six files into one strict JSONL file per dataset:

```bash
uv run python scripts/merge_eq_by_clip.py \
  --input-dir results/eq/clotho
```

For MeCAT:

```bash
uv run python scripts/merge_eq_by_clip.py \
  --input-dir results/eq/mecat
```

The output is `<input-dir>/eq_by_clip.jsonl`.

## 4. Run the complete retrieval evaluation

The pipeline performs all stages:

1. Source-caption text embeddings
2. Generated-query text embeddings
3. Audio embeddings
4. Embedding merge by `audio_id`
5. Recall@1/5/10 calculation
6. Recall bar chart

### AudioCaps evaluation from Hugging Face

AudioCaps evaluation uses the `test` split of [`msnowchanj/EQ`](https://huggingface.co/datasets/msnowchanj/EQ/viewer/default/test), including its bundled audio and all six query types:

```bash
uv run python scripts/eval_pipeline.py \
  --dataset audiocaps \
  --config configs/config_laion.yaml
```

On the first run, the pipeline automatically:

1. Downloads the Hugging Face `test` split
2. Selects rows where `dataset == "audiocaps"`
3. Exports `eq_by_clip.jsonl` and WAV files under `input/hf_eq/test/audiocaps/`
4. Runs embedding extraction and Recall@1/5/10 evaluation

The Hugging Face test split contains 2,868 examples. The config limits downloading to the five Parquet shards that contain AudioCaps rows, roughly 0.93GB compressed, instead of scanning all 12 test shards. For a quick check:

```bash
uv run python scripts/eval_pipeline.py \
  --dataset audiocaps \
  --config configs/config_laion.yaml \
  --limit 100
```

Use `--refresh-hf` to replace an existing local export.

### Clotho with LAION CLAP

```bash
uv run python scripts/eval_pipeline.py \
  --dataset clotho \
  --config configs/config_laion.yaml
```

### MeCAT with LAION CLAP

```bash
uv run python scripts/eval_pipeline.py \
  --dataset mecat \
  --config configs/config_laion.yaml
```

### Evaluate all configured models

```bash
uv run python scripts/eval_pipeline.py \
  --dataset clotho \
  --config configs/config.yaml
```

Useful options:

```bash
# First 100 clips only
uv run python scripts/eval_pipeline.py \
  --dataset clotho \
  --config configs/config_laion.yaml \
  --limit 100

# Selected models and query pools
uv run python scripts/eval_pipeline.py \
  --dataset clotho \
  --config configs/config.yaml \
  --models laion msclap \
  --pools original full_caption question \
  --ks 1 5 10

# CPU execution without plotting
uv run python scripts/eval_pipeline.py \
  --dataset clotho \
  --config configs/config_laion.yaml \
  --device cpu \
  --no-plot
```

Dataset paths are read from the selected config. They can be overridden with `--caption-jsonl` and `--audio-dir`.

## Evaluation outputs

For dataset `{dataset}` and model `{model}`:

```text
results/
├── audioEmb/{dataset}/{model}/emb.jsonl
├── testEmb/{dataset}/{model}/
│   ├── source_emb.jsonl
│   └── generated_emb.jsonl
├── mergedEmb/{dataset}/{model}/merged.jsonl
└── retrieval/{dataset}/
    ├── text_to_audio_recall.json
    ├── text_to_audio_recall.csv
    ├── text_to_audio_recall_bar.png
    └── {model}_{query_type}_retrieval.jsonl
```

The CSV summary is the fastest result to inspect:

```text
model,pool,n_queries,n_audio,R@1,R@5,R@10
laion,original,975,975,...
laion,full_caption,975,975,...
```

`original` uses the dataset's `source_caption` field. The other pools use their corresponding generated query.

## Model configuration

Configuration files are grouped under [`configs/`](configs):

```text
configs/
├── config.yaml
├── config_laion.yaml
├── config_msclap.yaml
├── config_mga.yaml
└── config_m2d.yaml
```

- `config_laion.yaml`: downloads the LAION model through Transformers
- `config_msclap.yaml`: uses the installed `msclap` package
- `config_mga.yaml`: requires a local MGA-CLAP repository and checkpoint
- `config_m2d.yaml`: requires a local M2D repository and checkpoint
- `config.yaml`: enables all four models

Each config maps AudioCaps evaluation to the Hugging Face `msnowchanj/EQ` test split, while Clotho and MeCAT use their configured local paths.

For MGA and M2D, update `repo_path`, `checkpoint_path`, and model-specific settings in the YAML if your local paths differ.

## EQ output schema

Before merging, each generated JSONL record has this shape:

```json
{
  "audio_id": "clip-id",
  "dataset": "clotho",
  "dataset_slug": "clotho_evaluation",
  "query_type": "question",
  "generated_query": "Can you hear ...?",
  "original_captions": ["..."],
  "metadata": {
    "file_name": "clip-id.wav"
  },
  "source_model": "gpt-4o-mini",
  "regen_model": "gpt-4o-mini"
}
```

`full_caption` uses every caption for the clip. The other five types use the middle caption at index `len(captions) // 2`.

## Repository layout

```text
EQ/
├── configs/                  # Generation and evaluation settings
├── eq_generation/           # Data loaders and EQ generators
├── scripts/
│   ├── prepare_data.py
│   ├── prepare_hf_eq.py
│   ├── makeeq.py
│   ├── merge_eq_by_clip.py
│   ├── eval_pipeline.py
│   ├── eval_merge_embeddings.py
│   └── eval_text_to_audio_retrieval.py
├── src/clap_eval/            # CLAP model wrappers
├── tests/
└── docs/
```

## Validation

```bash
uv run ruff check eq_generation scripts src tests
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q eq_generation scripts src tests
```

The code is released under the MIT license. AudioCaps, Clotho, MeCAT, and Hugging Face-hosted audio retain their own dataset and audio licenses.
