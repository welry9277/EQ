from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.eval_text_to_audio_retrieval import evaluate_pool, main


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "audio_id": "a",
                "audio_emb": [1.0, 0.0],
                "source": {"text": "a", "emb": [1.0, 0.0]},
                "generated_queries": {
                    "question": {"text": "wrong a", "emb": [0.0, 1.0]}
                },
            },
            {
                "audio_id": "b",
                "audio_emb": [0.0, 1.0],
                "source": {"text": "b", "emb": [0.0, 1.0]},
                "generated_queries": {
                    "question": {"text": "wrong b", "emb": [1.0, 0.0]}
                },
            },
        ]

    def test_original_queries_retrieve_the_correct_audio(self) -> None:
        metrics, rows = evaluate_pool(
            self.rows,
            pool="original",
            ks=[1, 2],
            query_batch_size=1,
        )
        self.assertEqual(metrics["text_to_audio"]["R@1"], 1.0)
        self.assertEqual(metrics["text_to_audio"]["R@2"], 1.0)
        self.assertEqual([row["rank"] for row in rows], [1, 1])

    def test_swapped_queries_have_recall_at_two_only(self) -> None:
        metrics, _ = evaluate_pool(
            self.rows,
            pool="question",
            ks=[1, 2],
            query_batch_size=2,
        )
        self.assertEqual(metrics["text_to_audio"]["R@1"], 0.0)
        self.assertEqual(metrics["text_to_audio"]["R@2"], 1.0)

    def test_cli_writes_json_and_csv_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                "models:\n"
                "  - name: test_model\n"
                "    enabled: true\n"
                "evaluation:\n"
                "  top_k_list: [1, 2]\n",
                encoding="utf-8",
            )
            merged_path = root / "merged" / "test_model" / "merged.jsonl"
            merged_path.parent.mkdir(parents=True)
            with merged_path.open("w", encoding="utf-8") as handle:
                for row in self.rows:
                    handle.write(json.dumps(row) + "\n")

            output_dir = root / "retrieval"
            argv = [
                "eval_text_to_audio_retrieval.py",
                "--config",
                str(config_path),
                "--dataset",
                "test",
                "--merged-root",
                str(root / "merged"),
                "--output-dir",
                str(output_dir),
                "--pools",
                "original",
                "question",
            ]
            with patch("sys.argv", argv):
                main()

            summary = json.loads(
                (output_dir / "text_to_audio_recall.json").read_text(encoding="utf-8")
            )
            csv_exists = (output_dir / "text_to_audio_recall.csv").is_file()

        self.assertEqual(
            summary["models"]["test_model"]["original"]["text_to_audio"]["R@1"],
            1.0,
        )
        self.assertTrue(csv_exists)


if __name__ == "__main__":
    unittest.main()
