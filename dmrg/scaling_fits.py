#!/usr/bin/env python3
"""scaling_fits.py --- Aggregate summary JSONs and produce CSVs + figures.

Inputs: globs of `*_summary.json` from `assemble_FQ_FEP.py` for both
        T=0 (one per (phase,N)) and T>0 (one per (phase,N,T)).
Outputs:
   data/FQ_T0_scaling.csv              (D2)
   data/FEP_T0_scaling.csv             (D3)
   data/eta_T0_scaling.csv             (D3)
   data/FQ_Tfin.csv                    (D7)
   data/FEP_Tfin.csv, eta_Tfin.csv     (D8)
   figures/FQ_lnN_T0.pdf               (D10)
   figures/eta_T_collapse.pdf          (D10)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    return [json.loads(Path(p).read_text()) for p in files]


def to_df_T0(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        m = r["metadata"]
        s = r["summary"]
        for k_str, blk in s["by_k"].items():
            rows.append({
                "phase": m["phase"], "N": m["N"], "k": int(k_str),
                "FQ": s["FQ"], "FEP": blk.get("FEP", float("nan")),
                "eta": blk.get("eta", float("nan")),
            })
    return pd.DataFrame(rows)


def to_df_Tfin(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        m = r["metadata"]
        s = r["summary"]
        for k_str, blk in s["by_k"].items():
            rows.append({
                "phase": m["phase"], "N": m["N"], "T": m["T"],
                "k": int(k_str),
                "FQ": s["FQ"], "FQ_err": s["FQ_err"],
                "FEP": blk.get("FEP", float("nan")),
                "eta": blk.get("eta", float("nan")),
            })
    return pd.DataFrame(rows)


def fit_wzw(df: pd.DataFrame) -> dict:
    """Fit F_Q(N) = A (ln N)^{3/2} + B for the (U) phase, N≥32."""
    sub = df[(df["phase"] == "U") & (df["N"] >= 32) & (df["k"] == 2)].copy()
    if len(sub) < 3:
        return {"note": "not enough points"}
    x = np.log(sub["N"].to_numpy()) ** 1.5
    y = sub["FQ"].to_numpy()
    A, B = np.polyfit(x, y, 1)
    return {"A": float(A), "B": float(B), "predicted_A": 4.0 / 3.0,
            "fit_points": len(sub)}


def plot_FQ_lnN(df: pd.DataFrame, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for ph in ("D", "C", "U"):
        sub = df[(df["phase"] == ph) & (df["k"] == 2)].sort_values("N")
        if sub.empty:
            continue
        ax.plot(sub["N"], sub["FQ"], "o-", label=f"({ph})")
    ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_ylabel(r"$F_Q$")
    ax.set_title(r"$F_Q$ vs $N$ for $G=S^z_\pi$")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)


def plot_eta_T_collapse(df: pd.DataFrame, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for ph, marker in zip(("D", "C", "U"), ("o", "s", "^")):
        sub = df[(df["phase"] == ph) & (df["k"] == 2)]
        for N, sub_N in sub.groupby("N"):
            ax.plot(sub_N["T"], sub_N["eta"], marker=marker, ls="-",
                    alpha=0.7, label=f"({ph}) N={N}")
    ax.set_xscale("log"); ax.set_xlabel(r"$T/J_1$")
    ax.set_ylabel(r"$\eta^{\star,\rm conn}_{\le 2}$")
    ax.set_title(r"$\eta_{\le 2}(T,N)$")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-T0", default="obs/T0/*_summary.json")
    ap.add_argument("--inputs-Tfin", default="obs/Tfin/*_summary.json")
    ap.add_argument("--csv-dir", default="../data")
    ap.add_argument("--outdir", default="../figures")
    args = ap.parse_args()

    Path(args.csv_dir).mkdir(parents=True, exist_ok=True)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    recs_T0 = load(args.inputs_T0)
    recs_Tf = load(args.inputs_Tfin)

    if recs_T0:
        df0 = to_df_T0(recs_T0)
        df0.to_csv(f"{args.csv_dir}/FQ_T0_scaling.csv", index=False)
        df0[["phase", "N", "k", "FEP"]].to_csv(
            f"{args.csv_dir}/FEP_T0_scaling.csv", index=False)
        df0[["phase", "N", "k", "eta"]].to_csv(
            f"{args.csv_dir}/eta_T0_scaling.csv", index=False)
        plot_FQ_lnN(df0, f"{args.outdir}/FQ_lnN_T0.pdf")
        wzw = fit_wzw(df0)
        Path(f"{args.csv_dir}/wzw_fit.json").write_text(json.dumps(wzw, indent=2))
        print("WZW fit:", wzw)

    if recs_Tf:
        df1 = to_df_Tfin(recs_Tf)
        df1.to_csv(f"{args.csv_dir}/FQ_Tfin.csv", index=False)
        df1[["phase", "N", "T", "k", "FEP"]].to_csv(
            f"{args.csv_dir}/FEP_Tfin.csv", index=False)
        df1[["phase", "N", "T", "k", "eta"]].to_csv(
            f"{args.csv_dir}/eta_Tfin.csv", index=False)
        plot_eta_T_collapse(df1, f"{args.outdir}/eta_T_collapse.pdf")


if __name__ == "__main__":
    main()
