#!/usr/bin/env python3
"""Publication figures for the corrected period-4 J1-J2-J3 T=0 campaign.

Inputs:
  dmrg/obs/T0/*.json       anchor points D=(1,0,0), C=(1,1,0), U=(1,1,1)
  dmrg/obs/T0_path/*.json  light D->C->U->D path scan

Outputs:
  figures/corrected_*.pdf/.png
  figures/corrected_T0_anchor.csv
  figures/corrected_T0_path.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
})

ROOT = Path(__file__).resolve().parents[1]
DMRG = ROOT / "dmrg"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PHASES = ("D", "C", "U")
PHASE_LABELS = {
    "D": r"$(D)$ isolated dimers $(1,0,0)$",
    "C": r"$(C)$ four-site clusters $(1,1,0)$",
    "U": r"$(U)$ uniform Heisenberg $(1,1,1)$",
}
COLORS = {"D": "#444444", "C": "#1f77b4", "U": "#d62728"}
MARKERS = {"D": "s", "C": "^", "U": "o"}

PATH_ORDER = [
    "D_t000", "DC_t025", "DC_t050", "DC_t075", "C_t100",
    "CU_t025", "CU_t050", "CU_t075", "U_t100",
    "UD_t075", "UD_t050", "UD_t025", "D_close_t000",
]
PATH_LABELS = ["D", ".25", ".50", ".75", "C", ".25", ".50", ".75", "U", ".75", ".50", ".25", "D"]


def read_run(path: Path) -> dict:
    data = json.loads(path.read_text())
    meta = data["metadata"]
    conv = data["convergence"]
    obs = data["observables"]
    return {
        "file": path.name,
        "label": meta.get("label") or meta.get("phase"),
        "phase": meta.get("phase", ""),
        "N": int(meta["N"]),
        "J1": float(meta["J1"]),
        "J2": float(meta["J2"]),
        "J3": float(meta["J3"]),
        "E0": float(conv["E0"]),
        "chi": int(conv["final_chi"]),
        "FQ": float(obs["FQ_Szpi"]),
    }


def load_anchor() -> list[dict]:
    rows = []
    for p in sorted((DMRG / "obs" / "T0").glob("*.json")):
        if p.name.endswith("_summary.json"):
            continue
        rows.append(read_run(p))
    return sorted(rows, key=lambda r: (PHASES.index(r["phase"]), r["N"]))


def load_path() -> list[dict]:
    rows = []
    for p in sorted((DMRG / "obs" / "T0_path").glob("*.json")):
        rows.append(read_run(p))
    order = {label: i for i, label in enumerate(PATH_ORDER)}
    return sorted(rows, key=lambda r: (r["N"], order[r["label"]]))


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["label", "phase", "N", "J1", "J2", "J3", "E0", "E0_per_N", "chi", "FQ"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["E0_per_N"] = rr["E0"] / rr["N"]
            w.writerow({k: rr.get(k, "") for k in fields})


def save(fig, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=220)
    plt.close(fig)
    print(f"wrote figures/{stem}.pdf and .png")


def style_log_x(ax, ticks=(16, 24, 32, 48, 64, 96, 128)) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


def fig_anchor_fq(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for ph in PHASES:
        rr = [r for r in rows if r["phase"] == ph]
        ax.plot([r["N"] for r in rr], [r["FQ"] for r in rr],
                marker=MARKERS[ph], color=COLORS[ph], markerfacecolor="white",
                markeredgewidth=1.0, label=PHASE_LABELS[ph])
    style_log_x(ax)
    ax.set_xlabel(r"chain length $N$ (OBC)")
    ax.set_ylabel(r"$F_Q[S^z_\pi]$")
    ax.set_title(r"Corrected period-4 chain: anchor points")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    save(fig, "corrected_fig1_FQ_anchors")


def fig_u_wzw(rows: list[dict]) -> None:
    rr = [r for r in rows if r["phase"] == "U"]
    Ns = np.array([r["N"] for r in rr], dtype=float)
    FQ = np.array([r["FQ"] for r in rr], dtype=float)
    x = np.log(Ns) ** 1.5
    mask = Ns >= 32
    A, B = np.polyfit(x[mask], FQ[mask], 1)
    A_pred = 4.0 / 3.0
    B_ref = FQ[mask].mean() - A_pred * x[mask].mean()
    xx = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(x, FQ, "o", color=COLORS["U"], markerfacecolor="white",
            markeredgewidth=1.0, label="DMRG OBC, U=(1,1,1)")
    ax.plot(xx, A * xx + B, "-", color="black",
            label=fr"fit $N\geq32$: $A={A:.3f}$")
    ax.plot(xx, A_pred * xx + B_ref, "--", color="#888888",
            label=r"WZW reference slope $4/3$")
    for N, xi, yi in zip(Ns.astype(int), x, FQ):
        ax.annotate(rf"$N={N}$", (xi, yi), xytext=(4, -8),
                    textcoords="offset points", fontsize=7, color="#555555")
    ax.set_xlabel(r"$(\ln N)^{3/2}$")
    ax.set_ylabel(r"$F_Q(N)$")
    ax.set_title(r"Uniform point: WZW scaling diagnostic")
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="lower right", handlelength=2.1)
    save(fig, "corrected_fig2_U_WZW_fit")


def fig_path(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for N in (16, 32, 64):
        rr = [r for r in rows if r["N"] == N]
        ax.plot(range(len(rr)), [r["FQ"] for r in rr], marker="o", markerfacecolor="white",
                markeredgewidth=1.0, label=fr"$N={N}$")
    for x, name in [(0, "D"), (4, "C"), (8, "U"), (12, "D")]:
        ax.axvline(x, color="#888888", linestyle=":", linewidth=0.8)
        ax.text(x, ax.get_ylim()[1], name, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(PATH_LABELS)))
    ax.set_xticklabels(PATH_LABELS)
    ax.set_xlabel(r"path coordinate along $D\to C\to U\to D$")
    ax.set_ylabel(r"$F_Q[S^z_\pi]$")
    ax.set_title(r"Light path scan: corrected period-4 chain")
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.legend(frameon=False, loc="upper left", ncols=3)
    save(fig, "corrected_fig3_FQ_path_light")


def fig_diagnostics(rows: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ph in PHASES:
        rr = [r for r in rows if r["phase"] == ph]
        Ns = [r["N"] for r in rr]
        ax1.plot(Ns, [r["chi"] for r in rr], marker=MARKERS[ph], color=COLORS[ph],
                 markerfacecolor="white", markeredgewidth=1.0, label=ph)
        ax2.plot([1 / r["N"] for r in rr], [r["E0"] / r["N"] for r in rr],
                 marker=MARKERS[ph], color=COLORS[ph], markerfacecolor="white",
                 markeredgewidth=1.0, label=ph)
    style_log_x(ax1)
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel(r"$\chi_{\rm final}$")
    ax1.set_title(r"Bond dimension")
    ax1.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax1.legend(frameon=False, loc="center right")
    ax2.set_xlabel(r"$1/N$")
    ax2.set_ylabel(r"$E_0/N$")
    ax2.set_title(r"Energy per site")
    ax2.grid(True, alpha=0.25, linewidth=0.4)
    ax2.legend(frameon=False, loc="best")
    save(fig, "corrected_fig4_diagnostics")


def main() -> None:
    anchor = load_anchor()
    path = load_path()
    print(f"loaded anchors={len(anchor)} path={len(path)}")
    write_csv(anchor, FIG / "corrected_T0_anchor.csv")
    write_csv(path, FIG / "corrected_T0_path.csv")
    fig_anchor_fq(anchor)
    fig_u_wzw(anchor)
    fig_path(path)
    fig_diagnostics(anchor)


if __name__ == "__main__":
    main()
