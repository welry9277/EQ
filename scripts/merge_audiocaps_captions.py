#!/usr/bin/env -S uv run python
"""Merge AudioCaps CSV rows that share the same (youtube_id, start_time).

Keeps the first row's audiocap_id; joins captions with a separator (default ' | ').
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    default_in = Path("input/audiocaps/test.csv")
    parser = argparse.ArgumentParser(
        description="Merge duplicate clip rows in AudioCaps-style CSV by youtube_id + start_time.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_in,
        help=f"Input CSV (default: {default_in})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: same folder as input, <stem>_merged.csv)",
    )
    parser.add_argument(
        "--separator",
        default=" | ",
        help="String between merged captions (default: ' | ').",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_merged.csv"

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input.resolve()}")

    # Preserve encounter order of groups; within each group preserve caption order.
    group_order: list[tuple[str, str]] = []
    groups: dict[tuple[str, str], dict] = {}

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"audiocap_id", "youtube_id", "start_time", "caption"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise SystemExit(f"CSV must have columns {sorted(required)}; missing: {sorted(missing)}")

        for row in reader:
            yt = str(row["youtube_id"]).strip()
            st = str(row["start_time"]).strip()
            cap = str(row["caption"]).strip()
            if not yt or not st or not cap:
                continue
            key = (yt, st)
            if key not in groups:
                group_order.append(key)
                groups[key] = {
                    "audiocap_id": str(row["audiocap_id"]).strip(),
                    "youtube_id": yt,
                    "start_time": st,
                    "captions": [cap],
                }
            else:
                groups[key]["captions"].append(cap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audiocap_id", "youtube_id", "start_time", "caption"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in group_order:
            g = groups[key]
            writer.writerow(
                {
                    "audiocap_id": g["audiocap_id"],
                    "youtube_id": g["youtube_id"],
                    "start_time": g["start_time"],
                    "caption": args.separator.join(g["captions"]),
                }
            )

    n_in = sum(len(g["captions"]) for g in groups.values())
    n_out = len(group_order)
    print(f"[INFO] Wrote {n_out} rows (from {n_in} caption rows) -> {args.output}")


if __name__ == "__main__":
    main()
