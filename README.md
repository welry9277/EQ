# EQ (Extended Query) Generation Toolkit

This repository generates EQ (Extended Query) data for audio retrieval using `gpt-5.4-mini`.

EQ expands one matched AudioCaps target into 6 query styles so that the same audio can be expressed in multiple retrieval forms.

## Overview

EQ is built from two inputs:

1. `vggsound_to_audiocaps_top1.json`
2. AudioCaps caption CSV

For each entry in `vggsound_to_audiocaps_top1.json`:

- treat the matched AudioCaps `audio_id` as the target
- use `audio_id` as the authoritative key for caption retrieval
- generate 6 query variants

The first 5 query types use only the top-1 matched caption.

- `key_phrase`
- `statement`
- `question`
- `command`
- `indirect`

The last type uses the full caption set for the same `audio_id`.

- `full_caption`

## Query Types

### Key Phrase

Core words or a shortened phrase with omitted sentence elements.

Example:

```text
Dog barking in distance.
```

### Statement

Objective factual description without emotion or explicit user intent.

Example:

```text
Person is walking on crunchy dry leaves in a quiet park.
```

### Question

Direct query asking whether a sound or event is present.

Example:

```text
Is there any dog barking sound?
```

### Command

Direct instruction to retrieve a matching sound.

Example:

```text
Find a high-pitched metallic clinking sound.
```

### Indirect

Polite or indirect request such as `Please`, `Could you`, or `I would appreciate it if`.

Examples:

```text
I would appreciate it if you could find a sound effect for a door creaking slowly.
Could you please provide a list of audio files that match the crashing waves in this clip?
```

### Full-Caption

One representative query created by summarizing all captions for the same `audio_id`.

This type is intended to preserve the full information from the 5 AudioCaps captions.

Note:

- `full_caption` performance is good
- the other query types are relatively weaker

## Setup

Install dependencies with `uv`.

```bash
uv sync
```

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

The default model configuration is stored in [config.yaml](/home/essibae5/UIQ/config.yaml).

```yaml
model:
  source_model: gpt-5.4-mini
  regen_model: gpt-5.4-mini
  backend: gpt
  temperature: 0.7
  batch_size: 10
  max_tokens: 100
```

## Step 1. Build Top-1 Mapping

Use [`scripts/map_vggsound_to_audiocaps_top1.py`](/home/essibae5/UIQ/scripts/map_vggsound_to_audiocaps_top1.py) to match each VGGSound category to the single most similar AudioCaps caption.

Required inputs:

- `--captions-csv`
- `--categories-json`
- `--output-json`

Example:

```bash
./scripts/map_vggsound_to_audiocaps_top1.py \
    --captions-csv input/audiocaps/test.csv \
    --categories-json input/vggsound_categories.json \
    --output-json input/vggsound_to_audiocaps_top1.json
```

The output JSON contains one entry per VGGSound category with fields such as:

- `category`
- `matched_caption`
- `audio_id`
- `similarity`
- `top_k_captions`

## Step 2. Generate EQ

Use [`scripts/makeeq.py`](/home/essibae5/UIQ/scripts/makeeq.py) to generate the 6 EQ files.

Example:

```bash
./scripts/makeeq.py \
    --mapping-json input/vggsound_to_audiocaps_top1.json \
    --captions-csv input/audiocaps/test.csv \
    --output-dir results/eq/audiocaps_test
```

If the mapping file contains 310 entries and validation passes, each query type will produce 310 outputs.

- `eq_key_phrase.jsonl`
- `eq_statement.jsonl`
- `eq_question.jsonl`
- `eq_command.jsonl`
- `eq_indirect.jsonl`
- `eq_full_caption.jsonl`

Total outputs:

```text
310 x 6 = 1860
```

## Generation Rules

### For `key_phrase`, `statement`, `question`, `command`, `indirect`

- use only the top-1 matched caption
- `original_captions` must contain exactly one caption

### For `full_caption`

- retrieve all captions for the same AudioCaps `audio_id`
- generate one representative query summarizing the full caption set
- `original_captions` must contain all captions for that `audio_id`

## Output Schema

Each line in the output JSONL files follows this structure:

```json
{
  "audio_id": "...",
  "dataset": "...",
  "dataset_slug": "...",
  "query_type": "...",
  "generated_query": "...",
  "original_captions": ["..."],
  "vgg": {
    "category": "...",
    "audio_id": "...",
    "similarity": 0.0
  },
  "metadata": {},
  "source_model": "gpt-5.4-mini",
  "regen_model": "gpt-5.4-mini"
}
```

Rules:

- `generated_query` must contain only the final query text
- no explanations or extra formatting
- outputs must stay faithful to the source caption or caption set

## Validation

The EQ pipeline writes `eq_validation.log` and validates the following:

- each mapping entry must produce 6 outputs
- missing caption-set cases must be logged clearly
- `audio_id` is the authoritative key for caption retrieval
- top-1 caption must belong to the retrieved caption set
- `full_caption` must retain the full caption set
- all other types must retain exactly one caption

If validation fails, the script stops and reports the error in the log.

## Code Flow

Main entrypoint:

- [`scripts/makeeq.py`](/home/essibae5/UIQ/scripts/makeeq.py)

Important components:

- [`scripts/makeuiq.py`](/home/essibae5/UIQ/scripts/makeuiq.py): loads AudioCaps grouped by `audio_id`
- [`uiq_generation/query_types.py`](/home/essibae5/UIQ/uiq_generation/query_types.py): EQ query type definitions and output schema
- [`uiq_generation/generators/prompts.py`](/home/essibae5/UIQ/uiq_generation/generators/prompts.py): prompt templates for each query type
- [`uiq_generation/generators/gpt_generator.py`](/home/essibae5/UIQ/uiq_generation/generators/gpt_generator.py): GPT generation backend

Execution flow:

1. Load AudioCaps CSV and group captions by `audio_id`.
2. Load `vggsound_to_audiocaps_top1.json`.
3. Validate each mapping entry using `audio_id`.
4. Extract the top-1 caption for the first 5 query types.
5. Retrieve all 5 captions for `full_caption`.
6. Generate 6 EQ variants with `gpt-5.4-mini`.
7. Save one JSONL file per query type.
8. Run final validation and write `eq_validation.log`.
