# EQ Generation Pipeline

LLM-based toolkit for generating `EQ` (Extended Query) data for audio retrieval from an **AudioCaps-style caption CSV** (grouped by clip). Every query type is prompted with the **full caption set** per clip; outputs are validated and saved as JSONL.

## Installation

Requirements:

- Python `>=3.9`
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key for generation

Clone and install:

```bash
git clone <your-fork-or-upstream-url>
cd UIQ
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

Default model settings live in [`config.yaml`](config.yaml):

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

Use [`scripts/prepare_data.py`](scripts/prepare_data.py) to create input directories and download supported metadata:

<데이터 다운로드 audiocaps>
```bash
./scripts/prepare_data.py --dataset audiocaps
```
<데이터 다운로드 clotho>
https://zenodo.org/records/3490684
여기서 evaluation으로 끝나는 데이터 3개 다운 받으면 됨 (audio가 다운 개오래걸림)
Behavior:

- `audiocaps`: downloads official `train.csv`, `val.csv`, and `test.csv` into `input/audiocaps/` from [cdjkim/audiocaps](https://github.com/cdjkim/audiocaps/tree/master/dataset)
- `mecat`: creates `input/mecat/metadata/` for manual placement

### AudioCaps via Hugging Face (`datasets`)

For **official** train/val/test caption CSVs from the paper repo, use `prepare_data.py --dataset audiocaps` (no `datasets` install required).

The snippet below is **[Hugging Face Hub](https://huggingface.co/datasets)** + the [`datasets`](https://huggingface.co/docs/datasets) library. The repo id **`d0rj/audiocaps`** is a mirror of **AudioCaps**, not Clotho. (Clotho is a different dataset; use `prepare_data.py --dataset clotho` or another Hub id for Clotho.)

After `uv sync --extra hf`, export a split to the same CSV columns `makeeq` expects (`youtube_id`, `start_time`, `caption`):

```bash
./scripts/download_audiocaps_hf.py --split test --output input/audiocaps/test.csv
```

Splits on the Hub are named `train`, `validation`, and `test`.

### One row per clip (merged captions)

Official CSVs have several caption lines per `(youtube_id, start_time)`. To collapse them into a single row (first `audiocap_id`, captions joined with ` | ` by default):

```bash
# Defaults: input/audiocaps/test.csv -> input/audiocaps/test_merged.csv
./scripts/merge_audiocaps_captions.py

./scripts/merge_audiocaps_captions.py \
  --input input/audiocaps/val.csv \
  --output input/audiocaps/val_merged.csv
```

Use `--separator` to change the join string. Note: `makeeq` already groups by clip when reading multi-row CSVs; merging is for tools that need one row per clip.

## EQ Query Types

The pipeline generates six variants per clip:

- `key_phrase`
- `statement`
- `question`
- `command`
- `indirect`
- `full_caption`

All types see the **full per-clip caption list** in the prompt; `original_captions` in JSONL is that full list for every type.

## Generate EQ

Use [`scripts/makeeq.py`](scripts/makeeq.py):

```bash
./scripts/makeeq.py \
  --captions-csv input/audiocaps/test.csv \
  --output-dir results/eq/audiocaps_test \
  --split test
```

Optional arguments:

- `--split`: split label stored in metadata, default `test`
- `--num-queries`: cap number of clips (order after grouping)
- `--config`: custom config file

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

## Repository Layout

```text
repo/
├── config.yaml
├── eq_generation/
├── scripts/
│   ├── prepare_data.py
│   ├── download_audiocaps_hf.py
│   ├── merge_audiocaps_captions.py
│   └── makeeq.py
└── results/
```
