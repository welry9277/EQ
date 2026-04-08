#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.makeuiq import load_audiocaps, load_config
from uiq_generation import QueryResult, QueryType, UIQGenerator


EQ_QUERY_TYPES = [
    QueryType.KEY_PHRASE,
    QueryType.STATEMENT,
    QueryType.QUESTION,
    QueryType.COMMAND,
    QueryType.INDIRECT,
    QueryType.FULL_CAPTION,
]

EQ_OUTPUT_FILENAMES = {
    QueryType.KEY_PHRASE: "eq_key_phrase.jsonl",
    QueryType.STATEMENT: "eq_statement.jsonl",
    QueryType.QUESTION: "eq_question.jsonl",
    QueryType.COMMAND: "eq_command.jsonl",
    QueryType.INDIRECT: "eq_indirect.jsonl",
    QueryType.FULL_CAPTION: "eq_full_caption.jsonl",
}


def load_mapping_entries(mapping_json_path: str) -> list[dict[str, Any]]:
    with open(mapping_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Mapping JSON must contain a top-level list.")
    return data


def build_record_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record["audio_id"]).strip(): record for record in records}


def extract_top1_caption(mapping_entry: dict[str, Any]) -> tuple[str, str]:
    matched_caption = str(mapping_entry.get("matched_caption", "")).strip()
    if matched_caption:
        return matched_caption, "matched_caption"

    top_k_captions = mapping_entry.get("top_k_captions")
    if isinstance(top_k_captions, list) and top_k_captions:
        top1_caption = str(top_k_captions[0].get("caption", "")).strip()
        if top1_caption:
            return top1_caption, "top_k_captions[0].caption"

    return "", ""


def format_caption_set(captions: list[str]) -> str:
    return "\n".join(f"- {caption}" for caption in captions)


def append_log(log_lines: list[str], message: str) -> None:
    print(message)
    log_lines.append(message)


def validate_and_prepare_entries(
    mapping_entries: list[dict[str, Any]],
    record_index: dict[str, dict[str, Any]],
    log_lines: list[str],
) -> list[dict[str, Any]]:
    prepared = []

    for idx, entry in enumerate(mapping_entries):
        audio_id = str(entry.get("audio_id", "")).strip()
        category = str(entry.get("category", "")).strip()

        if not audio_id:
            append_log(
                log_lines,
                f"[ERROR] Mapping entry {idx} is missing audio_id. category={category!r}",
            )
            continue

        record = record_index.get(audio_id)
        if record is None:
            append_log(
                log_lines,
                f"[ERROR] Missing caption set for audio_id={audio_id} category={category!r}. "
                "AudioCaps lookup uses audio_id as the authoritative key.",
            )
            continue

        top1_caption, top1_source = extract_top1_caption(entry)
        if not top1_caption:
            append_log(
                log_lines,
                f"[ERROR] Missing top-1 caption for audio_id={audio_id} category={category!r}. "
                "Expected matched_caption or top_k_captions[0].caption.",
            )
            continue

        caption_set = record["original_captions"]
        if top1_caption not in caption_set:
            append_log(
                log_lines,
                f"[ERROR] Top-1 caption mismatch for audio_id={audio_id} category={category!r}. "
                f"Caption from {top1_source} was not found in the AudioCaps caption set for that audio_id.",
            )
            continue

        prepared.append(
            {
                "audio_id": audio_id,
                "dataset": record["dataset"],
                "dataset_slug": record["dataset_slug"],
                "top1_caption": top1_caption,
                "caption_set": caption_set,
                "metadata": {
                    **record.get("metadata", {}),
                    "caption_count": len(caption_set),
                    "top1_caption_source": top1_source,
                },
                "vgg": {
                    "category": entry.get("category"),
                    "audio_id": audio_id,
                    "similarity": entry.get("similarity"),
                },
            }
        )

    return prepared


def build_results_for_query_type(
    prepared_entries: list[dict[str, Any]],
    generator: Any,
    query_type: QueryType,
    source_model: str,
    regen_model: str,
) -> list[QueryResult]:
    if query_type == QueryType.FULL_CAPTION:
        prompts = [format_caption_set(entry["caption_set"]) for entry in prepared_entries]
    else:
        prompts = [entry["top1_caption"] for entry in prepared_entries]

    raw_results = generator.generate(
        captions=prompts,
        query_type=query_type,
        clip_ids=[entry["audio_id"] for entry in prepared_entries],
        show_progress=True,
    )

    results = []
    for entry, raw_result in zip(prepared_entries, raw_results):
        if query_type == QueryType.FULL_CAPTION:
            original_captions = list(entry["caption_set"])
        else:
            original_captions = [entry["top1_caption"]]

        results.append(
            QueryResult(
                audio_id=entry["audio_id"],
                dataset=entry["dataset"],
                dataset_slug=entry["dataset_slug"],
                query_type=query_type,
                generated_query=raw_result.generated_query,
                original_captions=original_captions,
                vgg=entry["vgg"],
                metadata=entry["metadata"],
                source_model=source_model,
                regen_model=regen_model,
            )
        )

    return results


def validate_outputs(
    prepared_entries: list[dict[str, Any]],
    outputs_by_type: dict[QueryType, list[QueryResult]],
    log_lines: list[str],
) -> None:
    counts = Counter()
    expected_by_audio_id = {entry["audio_id"]: entry for entry in prepared_entries}

    for query_type, results in outputs_by_type.items():
        if len(results) != len(prepared_entries):
            append_log(
                log_lines,
                f"[ERROR] {query_type.value} produced {len(results)} outputs for {len(prepared_entries)} mapping entries.",
            )

        for result in results:
            counts[result.audio_id] += 1
            expected_entry = expected_by_audio_id.get(result.audio_id)
            if expected_entry is None:
                append_log(
                    log_lines,
                    f"[ERROR] Unexpected output audio_id={result.audio_id} in {query_type.value}.",
                )
                continue

            if query_type == QueryType.FULL_CAPTION:
                if result.original_captions != expected_entry["caption_set"]:
                    append_log(
                        log_lines,
                        f"[ERROR] Full-Caption output for audio_id={result.audio_id} "
                        "did not retain the complete caption set from AudioCaps.",
                    )
            elif result.original_captions != [expected_entry["top1_caption"]]:
                append_log(
                    log_lines,
                    f"[ERROR] {query_type.value} output for audio_id={result.audio_id} "
                    "did not retain exactly the matched top-1 caption.",
                )

    for entry in prepared_entries:
        produced = counts[entry["audio_id"]]
        if produced != len(EQ_QUERY_TYPES):
            append_log(
                log_lines,
                f"[ERROR] audio_id={entry['audio_id']} produced {produced} outputs; "
                f"expected {len(EQ_QUERY_TYPES)}.",
            )


def write_validation_log(output_dir: Path, log_lines: list[str]) -> Path:
    log_path = output_dir / "eq_validation.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="EQ (Extended Query) generation pipeline")
    parser.add_argument("--mapping-json", required=True, help="Path to vggsound_to_audiocaps_top1.json")
    parser.add_argument("--captions-csv", required=True, help="Path to the AudioCaps caption CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for per-type EQ JSONL files")
    parser.add_argument("--split", default="test", help="Split label passed into load_audiocaps(). Default: test")
    parser.add_argument("--num-queries", type=int, help="Optional limit on number of mapping entries")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.get("model", {})
    source_model = model_config.get("source_model", "gpt-5.4-mini")
    regen_model = model_config.get("regen_model", "gpt-5.4-mini")
    backend = model_config.get("backend", "gpt")
    temperature = model_config.get("temperature", 0.7)
    batch_size = model_config.get("batch_size", 10)
    max_tokens = model_config.get("max_tokens", 100)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    append_log(log_lines, f"[INFO] Loading AudioCaps records from {args.captions_csv}")
    records = load_audiocaps(args.captions_csv, split=args.split)
    record_index = build_record_index(records)
    append_log(log_lines, f"[INFO] Loaded {len(records)} unique AudioCaps audio_id groups")

    mapping_entries = load_mapping_entries(args.mapping_json)
    if args.num_queries and len(mapping_entries) > args.num_queries:
        mapping_entries = mapping_entries[: args.num_queries]
    append_log(log_lines, f"[INFO] Loaded {len(mapping_entries)} mapping entries from {args.mapping_json}")

    prepared_entries = validate_and_prepare_entries(mapping_entries, record_index, log_lines)
    if len(prepared_entries) != len(mapping_entries):
        append_log(
            log_lines,
            f"[ERROR] Validation failed for {len(mapping_entries) - len(prepared_entries)} mapping entries. "
            "EQ generation aborted.",
        )
        log_path = write_validation_log(output_dir, log_lines)
        raise SystemExit(f"See validation log: {log_path}")

    generator = UIQGenerator(
        backend=backend,
        model=source_model,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outputs_by_type: dict[QueryType, list[QueryResult]] = {}
    for query_type in EQ_QUERY_TYPES:
        append_log(log_lines, f"[INFO] Generating {query_type.value} for {len(prepared_entries)} entries")
        results = build_results_for_query_type(
            prepared_entries=prepared_entries,
            generator=generator,
            query_type=query_type,
            source_model=source_model,
            regen_model=regen_model,
        )
        outputs_by_type[query_type] = results
        generator.save_results(results, output_dir / EQ_OUTPUT_FILENAMES[query_type], format="jsonl")

    validate_outputs(prepared_entries, outputs_by_type, log_lines)

    error_count = sum(1 for line in log_lines if line.startswith("[ERROR]"))
    if error_count:
        append_log(log_lines, f"[ERROR] Validation finished with {error_count} error(s).")
        log_path = write_validation_log(output_dir, log_lines)
        raise SystemExit(f"See validation log: {log_path}")

    append_log(
        log_lines,
        f"[INFO] EQ generation complete. Produced {len(prepared_entries) * len(EQ_QUERY_TYPES)} outputs.",
    )
    log_path = write_validation_log(output_dir, log_lines)
    print(f"[INFO] Validation log saved to {log_path}")


if __name__ == "__main__":
    main()
