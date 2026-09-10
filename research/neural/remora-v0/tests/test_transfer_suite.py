import tempfile
import unittest
from pathlib import Path

from environments.transfer_suite import (
    ascii_sanitize,
    build_code_task_examples,
    build_math_examples,
    build_parity_examples,
    build_transfer_suite,
    task_stream,
)


class TransferSuiteTests(unittest.TestCase):
    def test_ascii_sanitization_is_deterministic_and_tokenizer_safe(self):
        value = ascii_sanitize("cafe\u0301 \u2014 line\n\x00")
        self.assertEqual(value, "cafe ? line\n")
        self.assertTrue(all(ord(character) < 128 for character in value))

    def test_verified_task_splits_use_distinct_ranges_and_are_encodable(self):
        train = build_math_examples(8, 11, low=0, high=20)
        valid = build_math_examples(8, 12, low=20, high=40)
        shifted = build_code_task_examples(8, 13, low=40, high=60, shifted=True)
        parity = build_parity_examples(8, 14, low=0, high=10)
        self.assertTrue({example.task_id for example in train}.isdisjoint(example.task_id for example in valid))
        self.assertTrue(all(example.response for example in train + valid + shifted))
        self.assertTrue(all(example.response in {"0", "1"} for example in parity))
        self.assertGreater(task_stream(train).numel(), 128)

    def test_unique_pair_capacity_is_explicit(self):
        with self.assertRaises(ValueError):
            build_parity_examples(5, 15, low=0, high=2)

    def test_suite_records_file_hashes_and_split_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wiki = root / "wiki"
            wiki.mkdir()
            for name in ("wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"):
                (wiki / name).write_text("A small held-out article about a model and a bus.\n" * 8)
            for directory in ("remora", "train", "experiments", "tests"):
                (root / directory).mkdir()
            (root / "remora" / "model.py").write_text("class Model:\n    pass\n" * 16)
            (root / "train" / "train.py").write_text("def train():\n    return 1\n" * 16)
            (root / "experiments" / "eval.py").write_text("def evaluate():\n    return 1\n" * 16)
            (root / "tests" / "test_eval.py").write_text("def test_eval():\n    assert True\n" * 16)
            suite = build_transfer_suite(root, wiki, code_task_count=4, math_task_count=4)
            self.assertEqual(suite.metadata["schema"], "remora-v1-transfer-suite")
            self.assertEqual(len(suite.metadata["code"]["train_files"]), 2)
            self.assertEqual(len(suite.metadata["code"]["valid_files"]), 2)
            self.assertEqual(len(suite.math_train), 4)
            self.assertEqual(len(suite.math_valid), 96)
            self.assertEqual(len(suite.math_shifted), 96)
            self.assertEqual(len(suite.parity_train), 4)
            self.assertEqual(len(suite.parity_valid), 96)
            self.assertEqual(len(suite.parity_shifted), 96)
            self.assertEqual(len(suite.metadata["wiki"]["splits"]), 3)
            self.assertTrue(all(value for value in suite.metadata["stream_sha256"].values()))


if __name__ == "__main__":
    unittest.main()
