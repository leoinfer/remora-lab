import unittest

import torch

from remora.memory import EvidenceStore
from remora.world_model import SurpriseTracker


class EvidenceTests(unittest.TestCase):
    def _store(self, independent: bool):
        store = EvidenceStore()
        store.add("inherited", "paper", "paper-cluster", {"z": "any"}, "Y", 0.8)
        for i in range(10):
            store.add("experienced", f"run-{i}" if independent else "same-run", f"cluster-{i}" if independent else "same-cluster", {"z": 1}, "X", 0.9)
        return store

    def test_duplicates_are_not_independent(self):
        duplicate = self._store(False).posterior({"z": 1})
        independent = self._store(True).posterior({"z": 1})
        self.assertEqual(duplicate["raw_observations"], 11)
        self.assertEqual(duplicate["effective_independent_clusters"], 2)
        self.assertEqual(independent["effective_independent_clusters"], 11)
        self.assertLess(duplicate["probability_x"], independent["probability_x"])

    def test_source_separation(self):
        store = self._store(False)
        separated = store.separated_posterior({"z": 1})
        self.assertEqual(separated["inherited"]["raw_observations"], 1)
        self.assertEqual(separated["experienced"]["raw_observations"], 10)

    def test_surprise_does_not_multiply_duplicate_clusters(self):
        duplicate = SurpriseTracker()
        for _ in range(12):
            duplicate.observe(torch.tensor([[0.95, 0.05]]), 1, "same-run")
        independent = SurpriseTracker()
        for i in range(12):
            independent.observe(torch.tensor([[0.95, 0.05]]), 1, f"run-{i}")
        self.assertEqual(duplicate.summary()["effective_independent_clusters"], 1)
        self.assertEqual(independent.summary()["effective_independent_clusters"], 12)
        self.assertLess(duplicate.summary()["compute_allocation"], independent.summary()["compute_allocation"])


if __name__ == "__main__":
    unittest.main()
