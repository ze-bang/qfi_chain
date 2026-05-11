#!/usr/bin/env python3
"""assemble_FQ_FEP.py --- Wiener solve → F_EP and η on the body-2..body-k basis.

Reads a JSON written by run_dmrg_T0.jl (T=0) or aggregate_metts.py (T>0)
and computes:
    F_Q                                    (already in input)
    F_EP[O*_{≤k}] = s^T Σ^+ s              (Wiener)
    η_conn_{≤k}  = F_EP / F_Q
for k in {2,4,6,8} (body-2 only here; body-4 augmentation requires a
four-point function which is not yet stored — see TODO).

Output: JSON with the summary plus the input metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def regularized_pinv(M: np.ndarray, lam_rel: float = 1e-10) -> np.ndarray:
    """Tikhonov-regularised pseudo-inverse. lam = lam_rel · tr(M)/dim."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    lam = lam_rel * (np.trace(M) / max(n, 1))
    return np.linalg.pinv(M + lam * np.eye(n))


def assemble(payload: dict, ks=(2, 4, 6, 8)) -> dict:
    obs = payload["observables"]
    is_thermal = "FQ_Szpi_mean" in obs
    if is_thermal:
        FQ = float(obs["FQ_Szpi_mean"])
        FQ_err = float(obs["FQ_Szpi_err"])
        s_full = np.asarray(obs["s_r_mean"], dtype=float)
        Sigma_diag = np.asarray(obs["Sigma_rr_mean"], dtype=float)
    else:
        FQ = float(obs["FQ_Szpi"])
        FQ_err = 0.0
        s_full = np.asarray(obs["s_r"], dtype=float)
        Sigma_diag = np.asarray(obs["Sigma_rr"], dtype=float)

    r_max = int(obs["r_max"])
    summary: dict[str, object] = {
        "FQ": FQ,
        "FQ_err": FQ_err,
        "r_max_available": r_max,
        "by_k": {},
    }

    for k in ks:
        # body-2 truncation at radius r ≤ k-1; only odd r contribute (selection rule).
        r_use = [r for r in range(1, min(k, r_max) + 1) if r % 2 == 1]
        if not r_use:
            summary["by_k"][str(k)] = {"FEP": 0.0, "eta": 0.0, "note": "no odd-r in range"}
            continue
        idx = [r - 1 for r in r_use]
        s_k = s_full[idx]
        # Diagonal-only Σ (body-2 only); off-diagonals are Wick-disconnected
        # in the seed pipeline. TODO: replace with full four-point.
        Sigma_k = np.diag(Sigma_diag[idx])
        Sigma_inv = regularized_pinv(Sigma_k)
        FEP = float(s_k @ Sigma_inv @ s_k)
        eta = FEP / FQ if FQ > 0 else float("nan")
        summary["by_k"][str(k)] = {
            "r_used": r_use,
            "s": s_k.tolist(),
            "Sigma_diag": Sigma_diag[idx].tolist(),
            "FEP": FEP,
            "eta": eta,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON from T0 or METTS aggregator")
    ap.add_argument("--ks", default="2,4,6,8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text())
    ks = tuple(int(x) for x in args.ks.split(","))
    summary = assemble(payload, ks=ks)
    out = {"metadata": payload["metadata"], "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}  FQ={summary['FQ']:.4g}  "
          + " ".join(f"η{k}={summary['by_k'][str(k)].get('eta', float('nan')):.3g}"
                     for k in ks))


if __name__ == "__main__":
    main()
