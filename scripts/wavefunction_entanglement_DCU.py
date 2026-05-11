#!/usr/bin/env python3
"""
Direct entanglement-structure analysis of the J1-J2-J3 ground state at the
three reference points (D), (C), (U) of the closed loop dimer -> cluster
-> uniform.

For each reference point we:
  1. Build the ground state in the S^z = 0 sector via Lanczos (using the
     bond Hamiltonians from j1j2j3_hierarchy.py).
  2. Embed it in the full 2^N Hilbert space.
  3. Compute, on the FULL wavefunction:
       - Bipartite von Neumann entanglement entropy S(L) for every
         contiguous cut L = 1, ..., N-1.
       - Two-site reduced density matrices  rho_{j,j+1}  for every nearest
         neighbour pair, and from them
              fidelity to the singlet,  F_s(j)   = <s| rho_{j,j+1} |s>
              two-site purity            Tr rho_{j,j+1}^2
              two-site entropy           S_2(j)  = -Tr rho log rho
       - Best-cluster-product fidelity for window sizes k = 1, 2, 3, 4:
              F_k = max_partition |<psi | prod_a |phi_a^*>|^2
         where each |phi_a^*> is taken to be the ground state of the
         REDUCED state rho_{B_a} on block B_a, i.e. its leading
         eigenvector.  This is the standard "best-2-producible-approx"
         lower bound on the partition fidelity.

The output is printed as a table and saved to NPZ for use in the
manuscript's Sec. 6.1.

Usage:
    python wavefunction_entanglement_DCU.py --N 8
    python wavefunction_entanglement_DCU.py --N 12
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from j1j2j3_hierarchy import (
    build_sz0_basis,
    bond_lists,
    build_bond_hamiltonian_sz0,
    ground_state_j1j2j3_full,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Reduced density matrices and entanglement diagnostics
# ---------------------------------------------------------------------------
def reshape_to_bipartition(psi: np.ndarray, N: int, A_sites: list[int]
                           ) -> np.ndarray:
    """Return matrix M of shape (2^|A|, 2^|B|) such that
       M[a, b] = <bitstring(A=a, B=b) | psi>.

    A_sites is the list of site indices in subsystem A; the rest go to B.
    Site indices in the full bitstring follow the convention used by
    apply_pauli_string: bit j of the basis index is the spin at site j.
    """
    A_sites = list(A_sites)
    B_sites = [j for j in range(N) if j not in A_sites]
    nA, nB = len(A_sites), len(B_sites)
    if nA + nB != N:
        raise ValueError("A_sites must be a subset of range(N)")
    # Build, for every basis index s, (a, b) where a, b are the
    # composite indices in A and B.
    states = np.arange(1 << N, dtype=np.int64)
    a_idx = np.zeros_like(states)
    for k, j in enumerate(A_sites):
        bit = (states >> j) & 1
        a_idx |= bit << k
    b_idx = np.zeros_like(states)
    for k, j in enumerate(B_sites):
        bit = (states >> j) & 1
        b_idx |= bit << k
    M = np.zeros((1 << nA, 1 << nB), dtype=psi.dtype)
    M[a_idx, b_idx] = psi
    return M


def schmidt_spectrum(psi: np.ndarray, N: int, A_sites: list[int]) -> np.ndarray:
    M = reshape_to_bipartition(psi, N, A_sites)
    # singular values
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-14]
    return s


def vn_entropy_from_schmidt(s: np.ndarray) -> float:
    p = s * s
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def reduced_density_matrix(psi: np.ndarray, N: int, A_sites: list[int]
                           ) -> np.ndarray:
    M = reshape_to_bipartition(psi, N, A_sites)
    rho = M @ M.conj().T
    return rho


def two_site_diagnostics(psi: np.ndarray, N: int, j: int) -> dict:
    """Diagnostics of the two-site reduced density matrix on sites (j, j+1)
    using PBC."""
    sites = [j % N, (j + 1) % N]
    rho = reduced_density_matrix(psi, N, sites)
    # singlet  |s> = (|01> - |10>) / sqrt(2)
    # bit indexing: site 0 = LSB.  Index = bit_{site 1}*2 + bit_{site 0}.
    singlet = np.zeros(4, dtype=np.complex128)
    singlet[0b01] = 1.0 / np.sqrt(2)   # bit_{site0}=1, bit_{site1}=0   -> |10>_{phys order}
    singlet[0b10] = -1.0 / np.sqrt(2)  # bit_{site0}=0, bit_{site1}=1   -> |01>
    # so |s> = (|10> - |01>)/sqrt2 in the (site0, site1) order; equivalent
    # up to global sign.
    F_s = float(np.real(singlet.conj() @ rho @ singlet))
    purity = float(np.real(np.trace(rho @ rho)))
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-14]
    S = float(-np.sum(evals * np.log(evals)))
    return dict(F_singlet=F_s, purity=purity, S2=S, eigs=evals)


# ---------------------------------------------------------------------------
# Best k-producible product fidelity (lower bound from local GS)
# ---------------------------------------------------------------------------
def kproducible_fidelity_lower_bound(psi: np.ndarray, N: int,
                                     blocks: list[list[int]]) -> float:
    """Take psi, and compute  |<psi | prod_a |phi_a*>>|^2  with each
    |phi_a*> the leading eigenvector of rho_{B_a}.  This is the squared
    overlap with the BEST product state of the form  prod_a |phi_a*>;
    it lower-bounds the maximal product-state fidelity over the partition
    by the choice of local GS, but for nearly-product states is essentially
    saturating.
    """
    # Build the product state  prod_a |phi_a*> in the full 2^N Hilbert space.
    psi_prod = np.ones(1, dtype=np.complex128)
    used_sites: list[int] = []
    block_states = []
    for B in blocks:
        rho_B = reduced_density_matrix(psi, N, B)
        evals, evecs = np.linalg.eigh(rho_B)
        # take the leading eigenvector
        phi = evecs[:, -1]
        block_states.append((B, phi))
        used_sites += list(B)
    # consistency: blocks must partition range(N)
    assert sorted(used_sites) == list(range(N)), \
        f"blocks must partition {list(range(N))}, got {sorted(used_sites)}"
    # Build the product state in the full 2^N basis.
    psi_full_prod = np.zeros(1 << N, dtype=np.complex128)
    states = np.arange(1 << N, dtype=np.int64)
    amps = np.ones(1 << N, dtype=np.complex128)
    for B, phi in block_states:
        nB = len(B)
        # local index of basis state on the block
        loc = np.zeros_like(states)
        for k, j in enumerate(B):
            bit = (states >> j) & 1
            loc |= bit << k
        amps *= phi[loc]
    psi_full_prod = amps
    psi_full_prod /= np.linalg.norm(psi_full_prod)
    return float(np.abs(np.vdot(psi_full_prod, psi)) ** 2)


def all_contiguous_partitions(N: int, k: int) -> list[list[list[int]]]:
    """All partitions of [0..N-1] into contiguous blocks of size <= k.
    PBC is ignored: the chain is cut linearly here.  For PBC ground states
    we enumerate all rotations to find the best partition.
    """
    out: list[list[list[int]]] = []

    def rec(start: int, acc: list[list[int]]):
        if start == N:
            out.append([list(b) for b in acc])
            return
        for size in range(1, min(k, N - start) + 1):
            block = list(range(start, start + size))
            acc.append(block)
            rec(start + size, acc)
            acc.pop()

    rec(0, [])
    return out


def best_kproducible_fidelity(psi: np.ndarray, N: int, k: int) -> tuple[float, list[list[int]]]:
    best_F = -1.0
    best_part = None
    # enumerate over rotations of the chain (PBC) to find best partition
    parts = all_contiguous_partitions(N, k)
    for shift in range(N):
        for part in parts:
            shifted = [[(j + shift) % N for j in B] for B in part]
            F = kproducible_fidelity_lower_bound(psi, N, shifted)
            if F > best_F:
                best_F = F
                best_part = shifted
    return best_F, best_part


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze_point(label: str, J1: float, J2: float, J3: float, N: int,
                  basis: np.ndarray, H_dimer, H_bridge, H_inter):
    print(f"\n=== Point ({label}) :   (J1, J2, J3) = ({J1}, {J2}, {J3}) ===")
    E0, psi_full, _ = ground_state_j1j2j3_full(
        N, J1, J2, J3, basis, H_dimer, H_bridge, H_inter
    )
    print(f"  E0/N = {E0 / N:.8f}")

    # --- Bipartite entanglement profile S(L) ---
    print(f"  Bipartite entanglement S(L) for contiguous cuts (PBC):")
    SL = np.zeros(N - 1)
    for L in range(1, N):
        A_sites = list(range(L))
        s = schmidt_spectrum(psi_full, N, A_sites)
        SL[L - 1] = vn_entropy_from_schmidt(s)
    line = "    " + "  ".join(f"L={L}:{SL[L-1]:.4f}" for L in range(1, N))
    print(line)

    # --- Two-site nearest-neighbour pair diagnostics ---
    print(f"  Pair (j, j+1) diagnostics:")
    F_s_arr = np.zeros(N)
    purity_arr = np.zeros(N)
    S2_arr = np.zeros(N)
    bond_class = np.empty(N, dtype="<U10")
    for j in range(N):
        d = two_site_diagnostics(psi_full, N, j)
        F_s_arr[j] = d["F_singlet"]
        purity_arr[j] = d["purity"]
        S2_arr[j] = d["S2"]
        # bond class follows j1j2j3_hierarchy.bond_lists pattern
        if j % 4 in (0, 2):
            bond_class[j] = "dimer"
        elif j % 4 == 1:
            bond_class[j] = "bridge"
        else:
            bond_class[j] = "inter"
    for j in range(N):
        print(f"    j={j}  ({bond_class[j]:6s})  F_singlet={F_s_arr[j]:.4f}  "
              f"purity={purity_arr[j]:.4f}  S2={S2_arr[j]:.4f}")

    # Class averages
    for cls in ("dimer", "bridge", "inter"):
        mask = bond_class == cls
        if mask.any():
            print(f"    AVG ({cls:6s}):  F_singlet={F_s_arr[mask].mean():.4f}"
                  f"  purity={purity_arr[mask].mean():.4f}"
                  f"  S2={S2_arr[mask].mean():.4f}")

    # --- Best k-producible product-state fidelity ---
    print(f"  Best k-producible product fidelity (block GS lower bound):")
    F_k = {}
    for k in range(1, min(5, N) + 1):
        F, part = best_kproducible_fidelity(psi_full, N, k)
        F_k[k] = F
        print(f"    k={k}:  F = {F:.6f}  best part = {part}")

    return dict(
        label=label, J=(J1, J2, J3), E0=E0, SL=SL,
        F_singlet=F_s_arr, purity=purity_arr, S2=S2_arr,
        bond_class=bond_class, F_k=F_k
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.05)
    args = p.parse_args()
    N = args.N
    eps = args.eps

    basis = build_sz0_basis(N)
    print(f"N={N}, S^z=0 sector dim = {basis.size}")
    dimer_bonds, bridge_bonds, inter_bonds = bond_lists(N)
    print(f"  dimer bonds  = {dimer_bonds}")
    print(f"  bridge bonds = {bridge_bonds}")
    print(f"  inter bonds  = {inter_bonds}")
    H_dimer = build_bond_hamiltonian_sz0(N, basis, dimer_bonds)
    H_bridge = build_bond_hamiltonian_sz0(N, basis, bridge_bonds)
    H_inter = build_bond_hamiltonian_sz0(N, basis, inter_bonds)

    points = [
        ("D", 1.0, eps, eps),    # dimer
        ("C", 1.0, 1.0, eps),    # cluster
        ("U", 1.0, 1.0, 1.0),    # uniform Heisenberg
    ]
    results = []
    for (lab, J1, J2, J3) in points:
        results.append(
            analyze_point(lab, J1, J2, J3, N, basis,
                          H_dimer, H_bridge, H_inter)
        )

    # Save
    out = DATA_DIR / f"wavefunction_entanglement_DCU_N{N}.npz"
    save_dict = {}
    for r in results:
        lab = r["label"]
        save_dict[f"{lab}_J"] = np.array(r["J"])
        save_dict[f"{lab}_E0"] = r["E0"]
        save_dict[f"{lab}_SL"] = r["SL"]
        save_dict[f"{lab}_F_singlet"] = r["F_singlet"]
        save_dict[f"{lab}_purity"] = r["purity"]
        save_dict[f"{lab}_S2"] = r["S2"]
        save_dict[f"{lab}_bond_class"] = r["bond_class"]
        save_dict[f"{lab}_F_k_keys"] = np.array(list(r["F_k"].keys()))
        save_dict[f"{lab}_F_k_vals"] = np.array(list(r["F_k"].values()))
    np.savez_compressed(out, **save_dict)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
