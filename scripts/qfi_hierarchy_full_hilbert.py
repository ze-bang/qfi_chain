"""
Shared full-Hilbert-space machinery for the operator-subspace hierarchy.

This module exposes:

    apply_pauli_string(psi, N, sites, codes)
        Apply  prod_i P_{sites[i]}^{codes[i]}  to a state vector in the full
        2^N computational basis, where codes are 0=I, 1=X, 2=Y, 3=Z.

    apply_uniform_Z(psi, N) / apply_uniform_X(psi, N)
        Diagonal / off-diagonal application of  G = sum_j sigma^z_j  or
        G = sum_j sigma^x_j .

    exact_support_strings(window_size)
        Yield Pauli code-tuples exactly supported on a window
        (first and last entries non-identity).

    accumulate_K_from_window(K, psi, N, dim, w_start, w_size, chunk=256)
        Add Phi_real @ Phi_real^T  to the running 2 dim x 2 dim Gram matrix
        for every Hermitian Pauli string exactly supported on the connected
        window starting at w_start with size w_size.

    eta_from_K(K, dG, normdG2)
        Compute  eta^star = ||P_{col(K)} t||^2 / ||t||^2  where
        t = [Im dG ; -Re dG] in R^{2 dim} .

    compute_hierarchy(psi, N, k_max, generator='Z', verbose=False)
        Run both the connected single-support and the cumulative linear-span
        hierarchies for k = 1 ... k_max and return a dict.

The matched-filter formula and its real-form derivation are documented in
Section 4.6 (Practical computation via the Gram matrix) of the paper.
"""
from __future__ import annotations

from itertools import product as iproduct

import numpy as np


# ---------------------------------------------------------------------------
# Pauli string action on the full 2^N Hilbert space
# ---------------------------------------------------------------------------
# Codes: 0 = I, 1 = X, 2 = Y, 3 = Z.  All four Pauli matrices are Hermitian.
#
# Convention: bit_j = 0 means site j in state |0>, bit_j = 1 means |1>.
# Single-site action:
#   I :  |b> -> |b>,                       coefficient 1
#   X :  |b> -> |1-b>,                     coefficient 1
#   Y :  |0> -> i|1>,  |1> -> -i|0>,       coefficient (+i if b=0, -i if b=1)
#   Z :  |0> -> +|0>, |1> -> -|1>,         coefficient (+1 if b=0, -1 if b=1)
def apply_pauli_string(psi: np.ndarray, N: int, sites: tuple,
                       codes: tuple) -> np.ndarray:
    """Apply Pauli string  prod_i P_{sites[i]}^{codes[i]}  to psi (2^N basis)."""
    states = np.arange(1 << N, dtype=np.int64)
    coeff = np.ones(1 << N, dtype=np.complex128)
    new_states = states.copy()
    for site, code in zip(sites, codes):
        if code == 0:
            continue
        bit = (states >> site) & 1
        mask = np.int64(1 << site)
        if code == 1:  # X
            new_states ^= mask
        elif code == 2:  # Y
            new_states ^= mask
            coeff *= np.where(bit == 0, 1j, -1j)
        elif code == 3:  # Z
            coeff *= np.where(bit == 0, 1.0, -1.0)
        else:
            raise ValueError(f"unknown Pauli code {code}")
    out = np.zeros_like(psi)
    np.add.at(out, new_states, coeff * psi)
    return out


def apply_uniform_Z(psi: np.ndarray, N: int) -> np.ndarray:
    """Apply  G_Z = sum_j sigma^z_j  to psi.  Diagonal in computational basis,
    eigenvalue (#zeros - #ones) = N - 2*popcount."""
    states = np.arange(1 << N, dtype=np.int64)
    popc = np.zeros_like(states)
    x = states.copy()
    while x.any():
        popc += x & 1
        x >>= 1
    g_diag = (N - 2 * popc).astype(np.float64)
    return g_diag * psi


def apply_uniform_X(psi: np.ndarray, N: int) -> np.ndarray:
    """Apply  G_X = sum_j sigma^x_j  to psi.  Sum over single-site bit-flips."""
    out = np.zeros_like(psi)
    for j in range(N):
        sites = (j,)
        out += apply_pauli_string(psi, N, sites, (1,))
    return out


def apply_staggered_Z(psi: np.ndarray, N: int) -> np.ndarray:
    """Apply  G_{Zpi} = sum_j (-1)^j sigma^z_j  to psi.

    Diagonal in computational basis with eigenvalue
    sum_j (-1)^j (1 - 2 b_j) on bitstring (b_0, ..., b_{N-1}).
    This is the canonical staggered (k=pi) Pauli generator that probes
    Neel-like correlations on a bipartite chain.  In normalisation, this
    is 2 * S^z_pi where S^z_pi = sum_j (-1)^j S^z_j (so F_Q reported here
    is 4 * Var(2 S^z_pi) = 16 * Var(S^z_pi); the support hierarchy itself
    is invariant under that overall rescaling because it is a ratio).
    """
    states = np.arange(1 << N, dtype=np.int64)
    g_diag = np.zeros(states.size, dtype=np.float64)
    for j in range(N):
        bit = (states >> j) & 1
        sign = 1.0 if (j % 2 == 0) else -1.0
        g_diag += sign * (1.0 - 2.0 * bit.astype(np.float64))
    return g_diag * psi


def apply_generator(psi: np.ndarray, N: int, generator: str) -> np.ndarray:
    if generator == "Z":
        return apply_uniform_Z(psi, N)
    if generator == "X":
        return apply_uniform_X(psi, N)
    if generator in ("Zpi", "Zstag", "staggered_Z"):
        return apply_staggered_Z(psi, N)
    raise ValueError(
        f"unknown generator {generator!r}; expected 'Z', 'X', or 'Zpi'"
    )


# ---------------------------------------------------------------------------
# Enumerate Hermitian Pauli strings exactly supported on a window
# ---------------------------------------------------------------------------
def exact_support_strings(window_size: int):
    """Yield code-tuples of length `window_size` with codes in {0, 1, 2, 3}
    such that codes[0] != 0 and codes[-1] != 0 (exact support).  All Pauli
    strings are Hermitian individually, so no symmetrisation is needed."""
    for codes in iproduct(range(4), repeat=window_size):
        if codes[0] == 0 or codes[-1] == 0:
            continue
        yield codes


# ---------------------------------------------------------------------------
# Hilbert-space Gram accumulator and matched-filter solver
# ---------------------------------------------------------------------------
def _flush_chunk_into_K(K: np.ndarray, buf: list, dim: int) -> None:
    """K += Phi_real @ Phi_real.T  for chunk of complex vectors `buf`."""
    arr = np.asarray(buf, dtype=np.complex128)  # (m, dim)
    Phi_real = np.empty((2 * dim, arr.shape[0]), dtype=np.float64)
    Phi_real[:dim, :] = arr.real.T
    Phi_real[dim:, :] = arr.imag.T
    K += Phi_real @ Phi_real.T


def accumulate_K_from_window(K: np.ndarray, psi: np.ndarray, N: int,
                             dim: int, w_start: int, w_size: int,
                             chunk: int = 256) -> int:
    """Accumulate K += Phi_real @ Phi_real.T for every Hermitian Pauli string
    exactly supported on the connected window starting at w_start with size
    w_size (periodic in N).  Returns the number of operators added."""
    sites = tuple((w_start + i) % N for i in range(w_size))
    buf = []
    n_added = 0
    for codes in exact_support_strings(w_size):
        Opsi = apply_pauli_string(psi, N, sites, codes)
        ev = complex(np.vdot(psi, Opsi))
        phi = Opsi - ev * psi
        buf.append(phi)
        n_added += 1
        if len(buf) >= chunk:
            _flush_chunk_into_K(K, buf, dim)
            buf = []
    if buf:
        _flush_chunk_into_K(K, buf, dim)
    return n_added


def eta_from_K(K: np.ndarray, dG: np.ndarray, normdG2: float):
    """Return (eta, rank) where eta = ||P_{col(K)} t||^2 / ||t||^2,
    t = [Im dG ; -Re dG] in R^{2 dim}, rank = #(positive eigenvalues of K)."""
    if normdG2 <= 0:
        return 0.0, 0
    K_sym = 0.5 * (K + K.T)
    w_eig, V_eig = np.linalg.eigh(K_sym)
    if w_eig.size == 0:
        return 0.0, 0
    tol = K.shape[0] * np.finfo(w_eig.dtype).eps * max(1.0, float(w_eig.max()))
    keep = w_eig > tol
    if not np.any(keep):
        return 0.0, 0
    dim = dG.shape[0]
    t = np.empty(2 * dim, dtype=np.float64)
    t[:dim] = dG.imag
    t[dim:] = -dG.real
    proj = V_eig[:, keep].T @ t
    num = float(np.sum(proj * proj))
    return num / normdG2, int(np.sum(keep))


# ---------------------------------------------------------------------------
# Driver: cumulative hierarchies for one state with a given generator
# ---------------------------------------------------------------------------
def compute_hierarchy(psi: np.ndarray, N: int, k_max: int,
                      generator: str = "Z", verbose: bool = False):
    """Compute single-support and span hierarchies for psi up to k_max.

    For k = 1 .. k_max:
        eta_single[k-1] = max over connected windows W of size <= k of eta^*(W)
        eta_span[k-1]   = eta^*(span of all operators on windows of size <= k)
        eta_per_window[k-1, j] = eta^*(window of size k starting at j)

    Returns a dict with arrays plus F_Q and ||dG||^2.
    """
    dim = 1 << N
    Gpsi = apply_generator(psi, N, generator)
    Gmean = complex(np.vdot(psi, Gpsi))
    dG = Gpsi - Gmean * psi
    normdG2 = float(np.real(np.vdot(dG, dG)))
    F_Q = 4.0 * normdG2

    eta_single = np.zeros(k_max, dtype=np.float64)
    eta_span = np.zeros(k_max, dtype=np.float64)
    eta_per_window = np.zeros((k_max, N), dtype=np.float64)

    K_span = np.zeros((2 * dim, 2 * dim), dtype=np.float64)
    for k in range(1, k_max + 1):
        best_single_k = 0.0
        for w_start in range(N):
            K_w = np.zeros_like(K_span)
            accumulate_K_from_window(K_w, psi, N, dim, w_start, k)
            eta_w, _ = eta_from_K(K_w, dG, normdG2)
            eta_per_window[k - 1, w_start] = eta_w
            best_single_k = max(best_single_k, eta_w)
            K_span += K_w
        eta_single[k - 1] = max(
            best_single_k, eta_single[k - 2] if k > 1 else 0.0
        )
        eta_span_k, rank_k = eta_from_K(K_span, dG, normdG2)
        eta_span[k - 1] = eta_span_k
        if verbose:
            print(f"  k={k}: rank(K_span)={rank_k}, "
                  f"eta_single={eta_single[k-1]:.6f}, "
                  f"eta_span={eta_span[k-1]:.6f}")

    return {
        "k": np.arange(1, k_max + 1),
        "eta_single": eta_single,
        "eta_span": eta_span,
        "eta_per_window": eta_per_window,
        "F_Q": F_Q,
        "normdG2": normdG2,
        "Gmean": Gmean,
    }
