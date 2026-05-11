#!/usr/bin/env python3
"""
Crystal-clear demonstration of the operator-subspace hierarchy on
cluster-product GHZ states.

For a chain of N qubits with N divisible by p, define the p-cluster GHZ
product state

    |psi^(p)>  =  prod_{m=0}^{N/p - 1}  [ |0>^p + |1>^p ] / sqrt(2)
                                            (on sites p*m, ..., p*m+p-1).

With the uniform generator  G_Z = sum_j sigma^z_j , the operator-subspace
hierarchy is exactly Heaviside in the cluster size p:

    eta^star_{span, <=k}  =  0   for k < p,
                          =  1   for k >= p.

This script verifies the prediction numerically by direct matched-filter
computation in the full 2^N Hilbert space (no diagonalisation required;
the states are explicit).  Outputs:

    QFI_CHAIN/data/ghz_cluster_hierarchy_N{N}.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qfi_hierarchy_full_hilbert import compute_hierarchy

HERE = Path(__file__).resolve().parent
QFI_DIR = HERE.parent
DATA_DIR = QFI_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def cluster_ghz_state(N: int, p: int) -> np.ndarray:
    """Return  |psi^(p)>  in the full 2^N computational basis."""
    if N % p != 0:
        raise ValueError(f"N={N} not divisible by cluster size p={p}")
    psi = np.zeros(1 << N, dtype=np.complex128)
    n_clusters = N // p
    norm = (1.0 / np.sqrt(2.0)) ** n_clusters
    for c in range(1 << n_clusters):
        idx = 0
        for m in range(n_clusters):
            if (c >> m) & 1:
                base_site = m * p
                for j in range(p):
                    idx |= (1 << (base_site + j))
        psi[idx] = norm
    return psi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--cluster-sizes", type=str, default="2,4,8")
    p.add_argument("--k-max", type=int, default=8)
    args = p.parse_args()

    N = args.N
    k_max = min(args.k_max, N)
    sizes = [int(x) for x in args.cluster_sizes.split(",")]
    for ps in sizes:
        if N % ps != 0:
            raise ValueError(f"cluster size {ps} does not divide N={N}")

    out = {
        "N": np.int64(N),
        "k_max": np.int64(k_max),
        "cluster_sizes": np.array(sizes, dtype=np.int64),
    }

    for ps in sizes:
        print(f"--- cluster GHZ state with p = {ps} ---")
        psi = cluster_ghz_state(N, ps)
        h = compute_hierarchy(psi, N, k_max, generator="Z", verbose=True)
        print(f"  F_Q                                = {h['F_Q']:.4f}")
        print(f"  eta_span(k)   = {np.round(h['eta_span'], 5)}")
        print(f"  eta_single(k) = {np.round(h['eta_single'], 5)}")
        out[f"p{ps}_eta_single"] = h["eta_single"]
        out[f"p{ps}_eta_span"] = h["eta_span"]
        out[f"p{ps}_eta_per_window"] = h["eta_per_window"]
        out[f"p{ps}_F_Q"] = np.float64(h["F_Q"])

    out_path = DATA_DIR / f"ghz_cluster_hierarchy_N{N}.npz"
    np.savez(out_path, **out)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
