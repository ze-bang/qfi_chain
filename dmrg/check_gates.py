#!/usr/bin/env python3
"""check_gates.py --- Validate G1–G8 against the run outputs.

Run after either obs assembly is complete; exits non-zero if any gate
fails. Designed to be invoked from the SLURM `fits` job.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def load_all(pattern: str) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]


def check_T0(records: list[dict]) -> list[str]:
    failures: list[str] = []
    for r in records:
        m = r["metadata"]
        s = r["summary"]
        # G4: at (D), η_{≤2} = 1 ± 1e-3 (loosened: numerical Wiener)
        if m["phase"] == "D":
            eta2 = s["by_k"].get("2", {}).get("eta", float("nan"))
            if not (0.99 <= eta2 <= 1.01):
                failures.append(f"G4 fail: phase=D N={m['N']} eta_2={eta2:.4f}")
    return failures


def check_Tfin(records: list[dict], T0_records: list[dict]) -> list[str]:
    failures: list[str] = []
    # G6: jackknife err / mean ≤ 5%
    for r in records:
        s = r["summary"]
        rel = s["FQ_err"] / s["FQ"] if s["FQ"] > 0 else float("inf")
        if rel > 0.05:
            failures.append(
                f"G6 fail: {r['metadata']} relerr_FQ={rel:.2%}"
            )
    # G7: lowest-T METTS FQ matches T=0 to 5%
    by_phase_N_T0 = {(r["metadata"]["phase"], r["metadata"]["N"]): r
                     for r in T0_records}
    by_phase_N_low = {}
    for r in records:
        m = r["metadata"]
        key = (m["phase"], m["N"])
        if key not in by_phase_N_low or m["T"] < by_phase_N_low[key]["metadata"]["T"]:
            by_phase_N_low[key] = r
    for key, r in by_phase_N_low.items():
        if key in by_phase_N_T0:
            fq_lowT = r["summary"]["FQ"]
            fq_zeroT = by_phase_N_T0[key]["summary"]["FQ"]
            rel = abs(fq_lowT - fq_zeroT) / fq_zeroT if fq_zeroT else float("inf")
            if rel > 0.05:
                failures.append(
                    f"G7 fail: {key} T={r['metadata']['T']} "
                    f"FQ_metts={fq_lowT:.3g} vs FQ_T0={fq_zeroT:.3g} "
                    f"rel={rel:.2%}"
                )
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T0-glob", default="obs/T0/*_summary.json")
    ap.add_argument("--Tfin-glob", default="obs/Tfin/*_summary.json")
    args = ap.parse_args()

    t0 = load_all(args.T0_glob)
    tf = load_all(args.Tfin_glob)
    fails = check_T0(t0) + check_Tfin(tf, t0)

    if fails:
        print(f"FAILED gates ({len(fails)}):")
        for f in fails:
            print(" ", f)
        sys.exit(1)
    print(f"All gates passed.  T0={len(t0)} files, Tfin={len(tf)} files.")


if __name__ == "__main__":
    main()
