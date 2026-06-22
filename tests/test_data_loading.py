from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

DATA_MODULE_PATH = Path(__file__).resolve().parents[1] / "eq_generation" / "data.py"
SPEC = importlib.util.spec_from_file_location("eq_data", DATA_MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {DATA_MODULE_PATH}")
DATA_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA_MODULE)
load_clotho = DATA_MODULE.load_clotho
load_audiocaps = DATA_MODULE.load_audiocaps
load_mecat = DATA_MODULE.load_mecat


class DataLoadingTests(unittest.TestCase):
    def test_load_audiocaps_groups_five_captions_per_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "audiocap_id",
                        "youtube_id",
                        "start_time",
                        "caption",
                    ],
                )
                writer.writeheader()
                for index in range(5):
                    writer.writerow(
                        {
                            "audiocap_id": str(index),
                            "youtube_id": "video",
                            "start_time": "20",
                            "caption": f"Caption {index}",
                        }
                    )

            rows = load_audiocaps(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["audio_id"], "video_20")
        self.assertEqual(len(rows[0]["original_captions"]), 5)
        self.assertEqual(rows[0]["metadata"]["audiocap_ids"], ["0", "1", "2", "3", "4"])

    def test_load_audiocaps_rejects_incomplete_caption_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            path.write_text(
                "audiocap_id,youtube_id,start_time,caption\n"
                "1,video,20,Only one caption\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Expected 5 caption rows"):
                load_audiocaps(path)

    def test_load_clotho_wide_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["file_name", "caption_1", "caption_2"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "file_name": "clip.wav",
                        "caption_1": "A bell rings.",
                        "caption_2": "A metal bell is ringing.",
                    }
                )

            rows = load_clotho(path, split="evaluation")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["audio_id"], "clip")
        self.assertEqual(rows[0]["metadata"]["file_name"], "clip.wav")
        self.assertEqual(len(rows[0]["original_captions"]), 2)

    def test_load_clotho_long_csv_groups_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["file_name", "caption"],
                )
                writer.writeheader()
                writer.writerow({"file_name": "clip.wav", "caption": "First."})
                writer.writerow({"file_name": "clip.wav", "caption": "Second."})

            rows = load_clotho(path)

        self.assertEqual(rows[0]["original_captions"], ["First.", "Second."])

    def test_load_mecat_recurses_and_accepts_short_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "nested"
            nested.mkdir()
            (nested / "clip.json").write_text(
                json.dumps({"short": "Engine rumbling", "domain": "vehicle"}),
                encoding="utf-8",
            )

            rows = load_mecat(tmp)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["original_captions"], ["Engine rumbling"])
        self.assertEqual(rows[0]["metadata"]["file_name"], "clip.flac")


if __name__ == "__main__":
    unittest.main()
