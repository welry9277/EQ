# UIQ (User Intent Query) Generation Toolkit

Standalone toolkit for generating User Intent Queries from audio captions. Evaluated and structured as described in **Omni-Embed-Audio (ACL 2026)**.

## Query Types

This tool generates 5 types of queries (and alternatively, 5 hard-negative variants) using `gpt-5.4-mini`:
- **Keyword**
- **Imperative**
- **Polite**
- **Question**
- **Paraphrase**

## Setup

We use `uv` for seamless package management.

```bash
uv sync
```

Create a `.env` file at the project root with your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

## Data Preparation

Before running the query generator, you can download and prepare the original captions into the `input/` folder automatically using the `prepare_data.py` script:

```bash
./scripts/prepare_data.py --dataset all
```
*(Note for MeCAT: Since it may not have a simple single download URL, the folder is created for you. You need to manually move its JSON metadata to `input/mecat/metadata/`)*

## Quick Start (Generating queries)

The CLI pipeline has been simplified to run through `./scripts/makeuiq.py`. 

### AudioCaps

```bash
./scripts/makeuiq.py \
    --dataset audiocaps \
    --captions-csv input/audiocaps/train.csv \
    --output-dir results/uiq/audiocaps \
    --num-queries 50 
```

*(Omit `--num-queries` to run on the entire dataset)*

### Clotho

```bash
./scripts/makeuiq.py \
    --dataset clotho \
    --captions-csv input/clotho/clotho_captions_development.csv \
    --output-dir results/uiq/clotho
```

### MeCAT

```bash
./scripts/makeuiq.py \
    --dataset mecat \
    --meta-dir input/mecat/metadata \
    --output-dir results/uiq/mecat
```

### Negative Queries

Negative queries require a compiled hard negatives JSONL file. Providing this file will automatically generate the 5 negative query categories corresponding to your base types.

```bash
./scripts/makeuiq.py \
    --dataset audiocaps \
    --captions-csv input/audiocaps/train.csv \
    --output-dir results/uiq/audiocaps_negative \
    --hard-neg-jsonl results/hard_negatives/audiocaps/stage2_filtered_negatives.jsonl
```
