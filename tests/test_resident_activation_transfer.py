import unittest

import torch

from experiments.resident_activation_transfer import _fit_candidate, _fit_classifier


class ResidentActivationTransferTests(unittest.TestCase):
    def test_candidate_and_prompt_control_report_all_fixed_splits(self):
        features = torch.randn(6, 12)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        candidate = _fit_candidate(
            features,
            labels,
            2,
            2,
            2,
            device=torch.device("cpu"),
            steps=2,
            seed=4,
        )
        control = _fit_classifier(
            torch.randn(6, 8),
            labels,
            2,
            2,
            2,
            device=torch.device("cpu"),
            steps=2,
            seed=5,
        )
        self.assertEqual(set(candidate["accuracy"]), {"train", "valid", "shifted"})
        self.assertEqual(set(control["accuracy"]), {"train", "valid", "shifted"})
        self.assertIn("changed_parameters", candidate)


if __name__ == "__main__":
    unittest.main()
