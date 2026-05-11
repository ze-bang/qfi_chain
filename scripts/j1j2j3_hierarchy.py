#!/usr/bin/env python3
"""
Operator-subspace (support) hierarchy for the J1-J2-J3 Heisenberg chain.

Hamiltonian (PBC, N=8 by default):

    H = sum_{j=0..N-1} J^{(j)} S_j . S_{j+1},

with the four-site bond pattern of qfi_heisenberg_chain_A.tex Eq. (model):

    J^{(j)} = J_1   if j mod 4 in {0, 2}     (dimer bonds)
              J_2   if j mod 4 == 1           (bridge bonds)
              J_3   if j mod 4 == 3           (inter-cluster bonds)

The chain interpolates three reference phases along a closed three-leg
loop:
    Leg 0  dimer -> cluster :  (J1, J2, J3) = (1, J2, eps),     J2 in [eps,1]
    Leg 1  cluster -> uniform: (J1, J2, J3) = (1, 1,  J3),      J3 in [eps,1]
    Leg 2  uniform -> dimer  : (J1, J2, J3) = (1, v,  v),       v  in [1, eps]

with eps = 0.05.

The support hierarchy of qfi_heisenberg_chain_B.tex (Sec. 4 / 5) is run
with the staggered generator G = sum_j (-1)^j sigma^z_j (= 2 S^z_pi)
that is the canonical low-body generator for staggered (Neel) order on
a bipartite chain (qfi_heisenberg_chain_A.tex Sec. 6).  At each path
point we compute, in the full 2^N Hilbert space:

    eta^star_{conn,<=k}(rho)   single connected window
    eta^star_{span,<=k}(rho)   linear span of all <=k windows
    eta^star_W                 each individual k-window (by start position)

via the Hilbert-space Gram trick of compute_hierarchy().

The ground state is computed in the S^z_tot = 0 sector via sparse
Lanczos (basis size C(N, N/2)) and embedded back into the full 2^N
Hilbert space for the matched-filter computation.

Outputs:
    QFI_CHAIN/data/j1j2j3_hierarchy_N{N}.npz
"""
from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qfi_hierarchy_full_hilbert import compute_hierarchy

HERE = Path(__file__).resolve().parent
QFI_DIR = HERE.parent
DATA_DIR = QFI_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# S^z = 0 sector of the spin-1/2 Heisenberg chain
# ---------------------------------------------------------------------------
def build_sz0_basis(N: int) -> np.ndarray:
    """Sorted bitstring basis of the S^z_tot = 0 sector of N spins-1/2."""
    half = N // 2
    if N % 2 != 0:
        raise ValueError(f"S^z = 0 sector needs even N, got N={N}")
    states = sorted(
        sum(1 << b for b in bits) for bits in combinations(range(N), half)
    )
    return np.array(states, dtype=np.int64)


def build_bond_hamiltonian_sz0(
    N: int, basis: np.ndarray, bonds: list[tuple[int, int]]
) -> sp.csr_matrix:
    """Heisenberg bond sum H_bond = sum_{(j,jp) in bonds} S_j . S_{jp}
    on the S^z = 0 sector.  S.S = 0.25 sigma^z sigma^z + 0.5 (S^+ S^- + S^- S^+).
    """
    dim = basis.size
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    diag = np.zeros(dim, dtype=np.float64)
    for (j, jp) in bonds:
        bj = ((basis >> j) & 1).astype(np.int64)
        bjp = ((basis >> jp) & 1).astype(np.int64)
        diag += 0.25 * (1 - 2 * bj).astype(np.float64) * (1 - 2 * bjp).astype(np.float64)
        mask = bj != bjp
        src = np.where(mask)[0]
        flip = np.int64((1 << j) | (1 << jp))
        flipped = basis[src] ^ flip
        tgt = np.searchsorted(basis, flipped)
        rows.append(tgt.astype(np.intp))
        cols.append(src.astype(np.intp))
        data.append(np.full(src.size, 0.5))
    rows.append(np.arange(dim, dtype=np.intp))
    cols.append(np.arange(dim, dtype=np.intp))
    data.append(diag)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    H = sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))
    return 0.5 * (H + H.T)


def bond_lists(N: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]],
                                list[tuple[int, int]]]:
    """Return three bond lists (PBC) following the J1-J2-J3 pattern."""
    dimer_bonds = [(j, (j + 1) % N) for j in range(N) if j % 4 in (0, 2)]
    bridge_bonds = [(j, (j + 1) % N) for j in range(N) if j % 4 == 1]
    inter_bonds = [(j, (j + 1) % N) for j in range(N) if j % 4 == 3]
    return dimer_bonds, bridge_bonds, inter_bonds


def ground_state_j1j2j3_full(
    N: int, J1: float, J2: float, J3: float,
    basis: np.ndarray, H_dimer: sp.csr_matrix, H_bridge: sp.csr_matrix,
    H_inter: sp.csr_matrix, v0: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Lanczos ground state in the S^z = 0 sector of the J1-J2-J3 chain.
    Returns (E0, psi_full, psi_sec) where psi_full is the embedding into
    the full 2^N Hilbert space and psi_sec is the sector vector.
    """
    H_op = J1 * H_dimer + J2 * H_bridge + J3 * H_inter
    e_arr, v_arr = spla.eigsh(H_op, k=1, which="SA", v0=v0,
                              maxiter=2000, tol=1e-12)
    psi_sec = v_arr[:, 0].real
    psi_sec /= np.linalg.norm(psi_sec)
    psi_full = np.zeros(1 << N, dtype=np.complex128)
    psi_full[basis] = psi_sec
    return float(e_arr[0]), psi_full, psi_sec


# ---------------------------------------------------------------------------
# Three-leg path of qfi_heisenberg_chain_A.tex Eq. (path)
# ---------------------------------------------------------------------------
def build_path(eps: float = 0.05, step: float = 0.05
               ) -> tuple[list[tuple[float, float, float]], np.ndarray,
                          list[str]]:
    """Closed three-leg loop dimer -> cluster -> uniform -> dimer.

    Defaults match qfi_heisenberg_chain_A.tex (eps=0.05, step=0.10 there;
    we use step=0.05 here to get a denser scan).
    """
    leg0 = [(1.0, float(J2), eps) for J2 in np.arange(eps, 1.0 + 1e-9, step)]
    leg1 = [(1.0, 1.0, float(J3))
            for J3 in np.arange(eps + step, 1.0 + 1e-9, step)]
    leg2_vals = np.concatenate(
        [np.arange(1.0 - step, eps + 1e-9, -step), [eps]]
    )
    leg2 = [(1.0, float(v), float(v)) for v in leg2_vals]
    path = leg0 + leg1 + leg2
    n0, n1, n2 = len(leg0), len(leg1), len(leg2)
    leg_ends = np.array([n0, n0 + n1, n0 + n1 + n2], dtype=np.int64)
    leg_labels = [
        rf"Leg 0: dimer$\to$cluster ({n0} pts)",
        rf"Leg 1: cluster$\to$uniform ({n1} pts)",
        rf"Leg 2: uniform$\to$dimer ({n2} pts)",
    ]
    return path, leg_ends, leg_labels


# ---------------------------------------------------------------------------
# Main: sweep the 3-leg path and compute the hierarchy at each path point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--generator", type=str, default="Zpi",
                   help="generator: 'Zpi' = sum_j (-1)^j sigma^z_j (default), "
                        "or 'Z' = sum_j sigma^z_j")
    args = p.parse_args()

    N = args.N
    k_max = min(args.k_max, N)
    eps = args.eps
    step = args.step

    print(f"\n=== J1-J2-J3 hierarchy ===  N={N}, k_max={k_max}, "
          f"eps={eps}, step={step}, generator={args.generator}")

    basis = build_sz0_basis(N)
    print(f"  S^z=0 sector dim = {basis.size}")
    dimer_bonds, bridge_bonds, inter_bonds = bond_lists(N)
    print(f"  bonds: dimer={dimer_bonds}, bridge={bridge_bonds}, "
          f"inter={inter_bonds}")
    H_dimer = build_bond_hamiltonian_sz0(N, basis, dimer_bonds)
    H_bridge = build_bond_hamiltonian_sz0(N, basis, bridge_bonds)
    H_inter = build_bond_hamiltonian_sz0(N, basis, inter_bonds)

    path, leg_ends, leg_labels = build_path(eps=eps, step=step)
    n_pts = len(path)
    print(f"  path: {n_pts} points, leg_ends = {leg_ends.tolist()}")
    for s in leg_labels:
        print(f"    {s}")

    J1_arr = np.zeros(n_pts, dtype=np.float64)
    J2_arr = np.zeros(n_pts, dtype=np.float64)
    J3_arr = np.zeros(n_pts, dtype=np.float64)
    E0_arr = np.zeros(n_pts, dtype=np.float64)
    F_Q_arr = np.zeros(n_pts, dtype=np.float64)
    eta_single_arr = np.zeros((n_pts, k_max), dtype=np.float64)
    eta_span_arr = np.zeros((n_pts, k_max), dtype=np.float64)
    eta_per_window_arr = np.zeros((n_pts, k_max, N), dtype=np.float64)

    psi_prev_sec: np.ndarray | None = None
    t_start = time.time()
    for ip, (J1, J2, J3) in enumerate(path):
        t0 = time.time()
        E0, psi_full, psi_sec = ground_state_j1j2j3_full(
            N, J1, J2, J3, basis, H_dimer, H_bridge, H_inter,
            v0=psi_prev_sec,
        )
        psi_prev_sec = psi_sec
        t_gs = time.time() - t0

        t1 = time.time()
        result = compute_hierarchy(psi_full, N, k_max,
                                   generator=args.generator, verbose=False)
        t_hi = time.time() - t1

        F_Q = result["F_Q"]
        eta_span = result["eta_span"]
        eta_single = result["eta_single"]

        J1_arr[ip] = J1
        J2_arr[ip] = J2
        J3_arr[ip] = J3
        E0_arr[ip] = E0
        F_Q_arr[ip] = F_Q
        eta_single_arr[ip] = eta_single
        eta_span_arr[ip] = eta_span
        eta_per_window_arr[ip] = result["eta_per_window"]

        elapsed = time.time() - t_start
        eta_per_pt = elapsed / (ip + 1)
        eta_remain = eta_per_pt * (n_pts - ip - 1)
        print(f"[{ip:2d}/{n_pts}] J=({J1:.3f},{J2:.3f},{J3:.3f}) "
              f"E0={E0:+8.4f} F_Q={F_Q:8.4f}  "
              f"eta_span={np.array2string(eta_span, precision=3, suppress_small=True)}  "
              f"[gs {t_gs:5.2f}s, hier {t_hi:6.2f}s, ETA {eta_remain/60:.1f}m]")

    out_path = DATA_DIR / f"j1j2j3_hierarchy_N{N}.npz"
    np.savez(
        out_path,
        N=np.int64(N),
        k_max=np.int64(k_max),
        eps=np.float64(eps),
        step=np.float64(step),
        generator=np.array(args.generator),
        leg_ends=leg_ends,
        J1=J1_arr,
        J2=J2_arr,
        J3=J3_arr,
        E0=E0_arr,
        F_Q=F_Q_arr,
        eta_single=eta_single_arr,
        eta_span=eta_span_arr,
        eta_per_window=eta_per_window_arr,
    )
    print(f"\nsaved {out_path}")
    print(f"total time: {(time.time() - t_start)/60:.2f} min")


if __name__ == "__main__":
    main()
