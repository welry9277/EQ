#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import urllib.request
import urllib.error
from pathlib import Path

# Common URLs for dataset metadata
AUDIOCAPS_URL = "https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv"
CLOTHO_URL = "https://zenodo.org/records/4783391/files/clotho_captions_development.csv"

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n -> to {dest} ...")
    try:
        # User-Agent header often required to bypass basic bot blockers
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✅ Successfully downloaded: {dest.name}")
    except urllib.error.URLError as e:
        print(f"❌ Failed to download {dest.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare dataset capsules for UIQ Generation")
    parser.add_argument("--dataset", required=True, choices=["audiocaps", "clotho", "mecat", "all"], 
                        help="Dataset to prepare (downloads to input/ directory)")
    args = parser.parse_args()

    datasets_to_process = ["audiocaps", "clotho", "mecat"] if args.dataset == "all" else [args.dataset]
    base_dir = Path("input")

    for ds in datasets_to_process:
        if ds == "audiocaps":
            out_path = base_dir / "audiocaps" / "train.csv"
            download_file(AUDIOCAPS_URL, out_path)
            
        elif ds == "clotho":
            out_path = base_dir / "clotho" / "clotho_captions_development.csv"
            download_file(CLOTHO_URL, out_path)
            
        elif ds == "mecat":
            mecat_dir = base_dir / "mecat" / "metadata"
            mecat_dir.mkdir(parents=True, exist_ok=True)
            print(f"⚠️  MeCAT metadata is usually restricted or complex to download automatically.")
            print(f"   Please manually place the MeCAT JSON files into: {mecat_dir}")

if __name__ == '__main__':
    main()
