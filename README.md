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

## Semantic Mapping: AudioCaps to VGGSound

This repository also includes a prototype-based semantic assignment pipeline using `BAAI/bge-large-en-v1.5`. It builds one embedding prototype per VGGSound category from prompt templates, then assigns each AudioCaps caption to exactly one category by top-1 cosine similarity.

Input files should be JSON lists of strings:

```json
["dog barking", "rain falling", "people cheering"]
```

```json
["A dog barking loudly in the distance", "Heavy rain falls on a roof"]
```

Run:

```bash
./scripts/map_audiocaps_to_vggsound.py \
    --categories-json input/vggsound_categories.json \
    --captions-json input/audiocaps_captions.json \
    --output-json results/semantic_mapping/audiocaps_to_vggsound.json
```

The output contains one result per caption with:
- `caption`
- `assigned_category`
- `similarity`
- `top_k`

If you want to reuse [`load_audiocaps()`](/home/essibae5/UIQ/scripts/makeuiq.py#L18) directly and map every caption in the CSV, use:

```bash
./scripts/map_audiocaps_csv_to_vggsound.py \
    --captions-csv input/audiocaps/test.csv \
    --categories-json input/vggsound_categories.json \
    --output-json results/semantic_mapping/audiocaps_test_all_captions.json
```

This version expands `original_captions` so each caption gets its own assignment result, while keeping the parent `audio_id`.

If you want the reverse direction, meaning:
- build embeddings for VGGSound categories first
- then, for each category, find the single best AudioCaps caption

use:

```bash
./scripts/map_vggsound_to_audiocaps_top1.py \
    --captions-csv input/audiocaps/test.csv \
    --categories-json input/vggsound_categories.json \
    --output-json results/semantic_mapping/vggsound_to_audiocaps_top1.json
```

This returns one result per category with:
- `category`
- `matched_caption`
- `audio_id`
- `similarity`
- `top_k_captions`
