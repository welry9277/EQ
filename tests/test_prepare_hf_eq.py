from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_hf_eq import export_rows


class PrepareHfEqTests(unittest.TestCase):
    def test_export_rows_filters_dataset_and_writes_audio(self) -> None:
        rows = [
            {
                "audio_id": "clip_a",
                "dataset": "audiocaps",
                "source_caption": "Source A",
                "full_caption": "Full A",
                "key_phrase": "Key A",
                "statement": "Statement A",
                "question": "Question A?",
                "command": "Command A.",
                "indirect": "Indirect A.",
                "audio": {"bytes": b"RIFFfake-audio", "path": "audio.wav"},
            },
            {
                "audio_id": "clip_b",
                "dataset": "clotho",
                "source_caption": "Source B",
                "full_caption": "Full B",
                "key_phrase": "Key B",
                "statement": "Statement B",
                "question": "Question B?",
                "command": "Command B.",
                "indirect": "Indirect B.",
                "audio": {"bytes": b"RIFFother-audio", "path": "audio.wav"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            written = export_rows(
                rows=rows,
                output_dir=output_dir,
                dataset_filter="audiocaps",
            )
            payload = json.loads(
                (output_dir / "eq_by_clip.jsonl").read_text(encoding="utf-8")
            )
            audio_path = output_dir / "audio" / payload["metadata"]["file_name"]
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            audio_bytes = audio_path.read_bytes()

        self.assertEqual(written, 1)
        self.assertEqual(payload["audio_id"], "clip_a")
        self.assertEqual(payload["generated_queries"]["full_caption"], "Full A")
        self.assertEqual(audio_bytes, b"RIFFfake-audio")
        self.assertEqual(manifest["dataset_filter"], "audiocaps")


if __name__ == "__main__":
    unittest.main()
