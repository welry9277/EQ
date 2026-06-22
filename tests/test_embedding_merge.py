from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_merge_embeddings import merge_model


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class EmbeddingMergeTests(unittest.TestCase):
    def test_merge_model_joins_all_embedding_types(self) -> None:
        caption_rows = [
            {
                "audio_id": "clip",
                "generated_queries": {
                    "question": "Can you hear a bell?",
                    "statement": "A bell is ringing.",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_root = root / "audio"
            text_root = root / "text"
            output_root = root / "merged"
            write_jsonl(
                audio_root / "model" / "emb.jsonl",
                [{"audio_id": "clip", "file_name": "clip.wav", "emb": [1, 0]}],
            )
            write_jsonl(
                text_root / "model" / "source_emb.jsonl",
                [{"audio_id": "clip", "source_caption": "A bell.", "emb": [1, 0]}],
            )
            write_jsonl(
                text_root / "model" / "generated_emb.jsonl",
                [
                    {
                        "audio_id": "clip",
                        "query_type": "question",
                        "generated_text": "Can you hear a bell?",
                        "emb": [1, 0],
                    },
                    {
                        "audio_id": "clip",
                        "query_type": "statement",
                        "generated_text": "A bell is ringing.",
                        "emb": [1, 0],
                    },
                ],
            )

            written, skipped = merge_model(
                model_name="model",
                caption_rows=caption_rows,
                audio_root=audio_root,
                text_root=text_root,
                output_root=output_root,
                strict=True,
            )
            result = json.loads(
                (output_root / "model" / "merged.jsonl").read_text(encoding="utf-8")
            )

        self.assertEqual((written, skipped), (1, 0))
        self.assertEqual(set(result["generated_queries"]), {"question", "statement"})
        self.assertEqual(result["source"]["text"], "A bell.")


if __name__ == "__main__":
    unittest.main()
