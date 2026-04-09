# EQ Generation Pipeline

LLM-based toolkit for generating `EQ` (Extended Query) data for audio retrieval.

This repository is now organized around the EQ workflow only:

- build a top-1 mapping from VGGSound categories to AudioCaps captions
- generate six EQ query variants per matched target
- validate and save the outputs as JSONL

## Installation

Requirements:

- Python `>=3.9`
- `uv`
- OpenAI API key for generation

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

Default model settings live in [`config.yaml`](/home/essibae5/UIQ/config.yaml):

```yaml
model:
  source_model: gpt-5.4-mini
  regen_model: gpt-5.4-mini
  backend: gpt
  temperature: 0.7
  batch_size: 10
  max_tokens: 100
```

## Data Preparation

Use [`scripts/prepare_data.py`](/home/essibae5/UIQ/scripts/prepare_data.py) to create input directories and download supported metadata:

```bash
./scripts/prepare_data.py --dataset audiocaps
./scripts/prepare_data.py --dataset clotho
./scripts/prepare_data.py --dataset all
```

Behavior:

- `audiocaps`: downloads `input/audiocaps/train.csv`
- `clotho`: downloads `input/clotho/clotho_captions_development.csv`
- `mecat`: creates `input/mecat/metadata/` for manual placement

If you want a different AudioCaps split such as `test.csv`, place it manually and pass that path to the scripts below.

## EQ Query Types

The pipeline generates:

- `key_phrase`
- `statement`
- `question`
- `command`
- `indirect`
- `full_caption`

The first five use the matched top-1 caption. `full_caption` uses the full AudioCaps caption set for the same `audio_id`.

## Step 1. Build Top-1 Mapping

Use [`scripts/map_vggsound_to_audiocaps_top1.py`](/home/essibae5/UIQ/scripts/map_vggsound_to_audiocaps_top1.py):

```bash
./scripts/map_vggsound_to_audiocaps_top1.py \
  --captions-csv input/audiocaps/test.csv \
  --categories-json input/vggsound_categories.json \
  --output-json input/vggsound_to_audiocaps_top1.json
```

Expected mapping fields:

- `category`
- `matched_caption`
- `audio_id`
- `similarity`
- `top_k_captions`

## Step 2. Generate EQ

Use [`scripts/makeeq.py`](/home/essibae5/UIQ/scripts/makeeq.py):

```bash
./scripts/makeeq.py \
  --mapping-json input/vggsound_to_audiocaps_top1.json \
  --captions-csv input/audiocaps/test.csv \
  --output-dir results/eq/audiocaps_test
```

Optional arguments:

- `--split`: split label stored in metadata, default `test`
- `--num-queries`: limit mapping entries
- `--config`: use a custom config file

Generated files:

- `eq_key_phrase.jsonl`
- `eq_statement.jsonl`
- `eq_question.jsonl`
- `eq_command.jsonl`
- `eq_indirect.jsonl`
- `eq_full_caption.jsonl`

Validation logs are written to `eq_validation.log`.

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
  "source_model": "gpt-5.4-mini",
  "regen_model": "gpt-5.4-mini"
}
```

EQ records may additionally include:

```json
{
  "vgg": {
    "category": "...",
    "audio_id": "...",
    "similarity": 0.0
  }
}
```

## Repository Layout

```text
repo/
├── config.yaml
├── eq_generation/
├── scripts/
│   ├── prepare_data.py
│   ├── makeeq.py
│   └── map_vggsound_to_audiocaps_top1.py
└── results/
```
