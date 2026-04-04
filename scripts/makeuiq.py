#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from uiq_generation import UIQGenerator, QueryType, QueryResult

def load_audiocaps(csv_path: str, split: str = "test") -> list[dict]:
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            youtube_id = row.get("youtube_id", "").strip()
            start_time = row.get("start_time", "").strip()
            caption = row.get("caption", "").strip()
            if not youtube_id or not start_time or not caption: continue
            
            try:
                start_time_float = float(start_time)
                start_time_int = int(start_time_float)
            except ValueError:
                start_time_int = start_time
                start_time_float = start_time

            audio_id = f"{youtube_id}_{start_time_int}"
            if audio_id not in data:
                data[audio_id] = {
                    "audio_id": audio_id,
                    "dataset": "audiocaps",
                    "dataset_slug": f"audiocaps_{split}",
                    "original_captions": [],
                    "metadata": {
                        "split": split,
                        "youtube_id": youtube_id,
                        "start_time": start_time_float
                    }
                }
            data[audio_id]["original_captions"].append(caption)
    return list(data.values())

def load_clotho(csv_path: str, split: str = "evaluation") -> list[dict]:
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("file_name", "").strip()
            if not filename: continue
            
            captions = []
            for i in range(1, 6):
                cap = row.get(f"caption_{i}", "").strip()
                if cap:
                    captions.append(cap)
            
            data.append({
                "audio_id": filename,
                "dataset": "clotho",
                "dataset_slug": f"clotho_{split}",
                "original_captions": captions,
                "metadata": {
                    "split": split,
                    "num_captions": len(captions)
                }
            })
    return data

def load_mecat(meta_dir: str) -> list[dict]:
    meta_dir = Path(meta_dir)
    data = []
    for json_path in sorted(meta_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            jdata = json.load(f)
        all_caps = jdata.get("short", [])
        if not all_caps: continue
        
        domain = jdata.get("domain", "00A")
        if "metadata" in jdata and "domain" in jdata["metadata"]:
            domain = jdata["metadata"]["domain"]

        data.append({
            "audio_id": json_path.stem,
            "dataset": "mecat",
            "dataset_slug": "mecat",
            "original_captions": all_caps,
            "metadata": {
                "domain": domain,
                "num_captions": len(all_caps)
            }
        })
    return data

def load_hard_negative_map(jsonl_path: str) -> Dict[str, str]:
    neg_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cid = record.get("clip_id", record.get("audio_id", ""))
            negatives = record.get("hard_negatives", [])
            if negatives:
                neg_map[cid] = negatives[0].get("caption", "")
    return neg_map

def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="UIQ Generation Pipeline")
    parser.add_argument("--dataset", required=True, choices=["audiocaps", "clotho", "mecat"])
    parser.add_argument("--captions-csv", type=str, help="Path to captions CSV")
    parser.add_argument("--meta-dir", type=str, help="Path to MeCAT metadata directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-queries", type=int, help="Number of queries to generate (optional)")
    parser.add_argument("--hard-neg-jsonl", type=str, help="Hard negatives JSONL for negative queries")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    model_config = config.get("model", {})
    
    source_model = model_config.get("source_model", "gpt-5.1")
    regen_model = model_config.get("regen_model", "gpt-5.1")
    backend = model_config.get("backend", "gpt")
    temperature = model_config.get("temperature", 0.7)
    batch_size = model_config.get("batch_size", 10)
    max_tokens = model_config.get("max_tokens", 100)

    print(f"[INFO] Using {backend} backend with model: {source_model}")

    if args.dataset == "audiocaps":
        if not args.captions_csv: parser.error("--captions-csv required for audiocaps")
        records = load_audiocaps(args.captions_csv)
    elif args.dataset == "clotho":
        if not args.captions_csv: parser.error("--captions-csv required for clotho")
        records = load_clotho(args.captions_csv)
    elif args.dataset == "mecat":
        if not args.meta_dir: parser.error("--meta-dir required for mecat")
        records = load_mecat(args.meta_dir)

    if args.num_queries and len(records) > args.num_queries:
        records = records[:args.num_queries]

    print(f"[INFO] Loaded {len(records)} audio records from {args.dataset}")

    generator = UIQGenerator(
        backend=backend, 
        model=source_model, 
        batch_size=batch_size, 
        max_tokens=max_tokens, 
        temperature=temperature
    )

    query_types_to_generate = [
        QueryType.KEYWORD, 
        QueryType.IMPERATIVE, 
        QueryType.POLITE, 
        QueryType.QUESTION, 
        QueryType.PARAPHRASE
    ]
    
    hard_negative_captions = None
    if args.hard_neg_jsonl:
        query_types_to_generate = [
            QueryType.KEYWORD_NEGATIVE, 
            QueryType.IMPERATIVE_NEGATIVE, 
            QueryType.POLITE_NEGATIVE, 
            QueryType.QUESTION_NEGATIVE, 
            QueryType.PARAPHRASE_NEGATIVE
        ]
        neg_map = load_hard_negative_map(args.hard_neg_jsonl)
        hard_negative_captions = [neg_map.get(rec["audio_id"]) for rec in records]
        print(f"[INFO] Loaded {len(neg_map)} hard negative mappings")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captions = []
    clip_ids = []
    for rec in records:
        caps = rec["original_captions"]
        tgt_idx = len(caps) // 2 if len(caps) > 1 else 0
        captions.append(caps[tgt_idx])
        clip_ids.append(rec["audio_id"])

    for qt in query_types_to_generate:
        print(f"[INFO] Generating {qt.value} queries for {len(captions)} records...")

        raw_results = generator.generate(
            captions=captions,
            query_type=qt,
            clip_ids=clip_ids,
            hard_negative_captions=hard_negative_captions,
            show_progress=True,
        )

        final_results = []
        for rec, raw_res in zip(records, raw_results):
            final_res = QueryResult(
                audio_id=rec["audio_id"],
                dataset=rec["dataset"],
                dataset_slug=rec["dataset_slug"],
                query_type=qt,
                generated_query=raw_res.generated_query,
                original_captions=rec["original_captions"],
                metadata=rec["metadata"],
                source_model=source_model,
                regen_model=regen_model
            )
            final_results.append(final_res)

        out_path = output_dir / f"uiq_{qt.value}.jsonl"
        generator.save_results(final_results, out_path, format="jsonl")

    print(f"[INFO] Done. Results saved to {output_dir}")

if __name__ == '__main__':
    main()