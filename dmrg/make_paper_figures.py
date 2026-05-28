#!/usr/bin/env python3
"""make_paper_figures.py -- Publication-quality figures for the QFI_CHAIN
manuscript, built from the partial run that has finished so far (all 21
T=0 summaries; T>0 data still in flight).  Produces vector PDF + raster
PNG for each figure inside ../figures/.

Figures generated:
  Fig 1.  F_Q(N) vs N for the three reference phases (D), (C), (U).
  Fig 2.  WZW asymptotic test at (U):  F_Q vs (ln N)^{3/2} with the
          predicted slope 4/3 overlaid.
  Fig 3.  Asymptotic-coefficient diagnostic:  F_Q / (ln N)^{3/2} -> A.
  Fig 4.  Ground-state diagnostics: bond dim and energy per site.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")

# ---- publication style ------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,        # editable text in vector PDF
    "ps.fonttype":  42,
    "text.usetex": False,      # mathtext is enough; tex would need a TeX install
})

PHASES = ("D", "C", "U")
PHASE_LABELS = {
    "D": r"$(D)$ Majumdar--Ghosh  $(1,1/2,0)$",
    "C": r"$(C)$ Cluster product  $(0.241,0.451,0.308)$",
    "U": r"$(U)$ Heisenberg / WZW  $(1,0,0)$",
}
# Use distinct, print-friendly markers and a tight monochrome+accent palette.
MARKERS = {"D": "s", "C": "^", "U": "o"}
COLORS  = {"D": "#444444", "C": "#1f77b4", "U": "#d62728"}

DATA_DIR = Path(__file__).resolve().parent / "obs" / "T0"
FIG_DIR  = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_T0() -> dict:
    """Return {phase: {N: dict(FQ, E0, chi)}}."""
    out: dict = {p: {} for p in PHASES}
    for f in sorted(DATA_DIR.glob("*_summary.json")):
        s = json.loads(f.read_text())
        m = s["metadata"]
        raw = json.loads(Path(str(f).replace("_summary.json", ".json")).read_text())
        out[m["phase"]][m["N"]] = {
            "FQ":  s["summary"]["FQ"],
            "E0":  raw["convergence"]["E0"],
            "chi": raw["convergence"]["final_chi"],
        }
    return out


def save_pair(fig, stem: str) -> None:
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(FIG_DIR.parent)}  +  {png.name}")


def _style_log_x(ax, ticks=(16, 24, 32, 48, 64, 96, 128)) -> None:
    """Show only the chain-length ticks (no overlapping log minor labels)."""
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticks([], minor=True)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


# ---- Figure 1 -- F_Q vs N for D/C/U ----------------------------------------
def fig1_FQ_vs_N(D: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for ph in PHASES:
        Ns = sorted(D[ph].keys())
        ys = [D[ph][N]["FQ"] for N in Ns]
        ax.plot(Ns, ys,
                marker=MARKERS[ph], color=COLORS[ph],
                markerfacecolor="white", markeredgewidth=1.0,
                label=PHASE_LABELS[ph])
    _style_log_x(ax)
    ax.set_xlabel(r"chain length $N$  (open boundary)")
    ax.set_ylabel(r"$F_Q[\,|\psi_0\rangle;\,G\,]$")
    ax.set_title(r"Quantum Fisher information for $G=S^{z}_{\pi}$")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6)
    save_pair(fig, "fig1_FQ_vs_N_T0")


# ---- Figure 2 -- WZW asymptotic at (U) -------------------------------------
def fig2_wzw_test(D: dict) -> None:
    Ns = np.array(sorted(D["U"].keys()))
    Fq = np.array([D["U"][N]["FQ"] for N in Ns])
    x  = np.log(Ns) ** 1.5

    # Two-parameter linear fit on N >= 32.
    fit_mask = Ns >= 32
    A, B = np.polyfit(x[fit_mask], Fq[fit_mask], 1)

    # Reference: pure 4/3 prefactor with the same intercept.
    A_pred = 4.0 / 3.0
    B_ref  = Fq[fit_mask].mean() - A_pred * x[fit_mask].mean()
    x_grid = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.plot(x, Fq, "o", color=COLORS["U"], markerfacecolor="white",
            markeredgewidth=1.0, label="DMRG, $(U)$ phase")
    ax.plot(x_grid, A * x_grid + B, "-", color="black", linewidth=1.0,
            label=fr"fit  ($N\!\geq\!32$):  $A={A:.3f}$,  $B={B:+.2f}$")
    ax.plot(x_grid, A_pred * x_grid + B_ref, "--", color="#888888",
            linewidth=1.0,
            label=r"WZW prediction  $A=4/3$")
    # Annotate which point is which N.
    for N, xi, yi in zip(Ns, x, Fq):
        ax.annotate(rf"$N\!=\!{N}$", (xi, yi), xytext=(4, -8),
                    textcoords="offset points", fontsize=7, color="#555")
    ax.set_xlabel(r"$(\ln N)^{3/2}$")
    ax.set_ylabel(r"$F_{Q}(N)$")
    ax.set_title(r"WZW marginal scaling test at $(U)$")
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    save_pair(fig, "fig2_FQ_WZW_fit_U")


# ---- Figure 3 -- effective WZW coefficient vs N ----------------------------
def fig3_effective_A(D: dict) -> None:
    """A_eff(N) = F_Q(N) / (ln N)^{3/2}.  Should approach 4/3 from below as
    N -> infinity if the WZW prediction holds."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for ph in PHASES:
        Ns = np.array(sorted(D[ph].keys()))
        Fq = np.array([D[ph][N]["FQ"] for N in Ns])
        Aeff = Fq / np.log(Ns) ** 1.5
        ax.plot(Ns, Aeff,
                marker=MARKERS[ph], color=COLORS[ph],
                markerfacecolor="white", markeredgewidth=1.0,
                label=PHASE_LABELS[ph].split("  ")[0])
    ax.axhline(4.0 / 3.0, color="#888", linestyle="--", linewidth=1.0,
               label=r"$4/3$  (WZW)")
    _style_log_x(ax)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$F_{Q}(N) \,/\, (\ln N)^{3/2}$")
    ax.set_title(r"Effective WZW coefficient vs $N$")
    ax.set_ylim(0.0, 1.7)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, handlelength=1.6,
              fontsize=8)
    save_pair(fig, "fig3_Aeff_vs_N")


# ---- Figure 4 -- two-panel diagnostics: chi(N) and E_0/N ------------------
def fig4_diagnostics(D: dict) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    for ph in PHASES:
        Ns  = np.array(sorted(D[ph].keys()))
        chi = np.array([D[ph][N]["chi"] for N in Ns])
        e   = np.array([D[ph][N]["E0"] / N for N in Ns])
        axL.plot(Ns, chi,
                 marker=MARKERS[ph], color=COLORS[ph],
                 markerfacecolor="white", markeredgewidth=1.0,
                 label=PHASE_LABELS[ph].split("  ")[0])
        axR.plot(1.0 / Ns, e,
                 marker=MARKERS[ph], color=COLORS[ph],
                 markerfacecolor="white", markeredgewidth=1.0,
                 label=PHASE_LABELS[ph].split("  ")[0])

    axL.axhline(1200, color="#888", linestyle=":", linewidth=0.8)
    axL.text(110, 1280, r"$\chi_{\max}=1200$", fontsize=8, color="#555",
             ha="right")
    _style_log_x(axL)
    axL.set_yscale("log")
    axL.set_ylim(1.2, 2400)
    axL.set_xlabel(r"$N$")
    axL.set_ylabel(r"$\chi_{\mathrm{final}}$")
    axL.set_title(r"DMRG bond dimension at convergence")
    axL.grid(True, which="both", alpha=0.25, linewidth=0.4)
    axL.legend(loc="center right", frameon=False, handlelength=1.6)

    # Reference Heisenberg ground-state energy per site.
    e_inf_U = 0.25 - np.log(2.0)
    axR.axhline(e_inf_U, color=COLORS["U"], linestyle=":", linewidth=0.8)
    axR.text(0.0035, e_inf_U - 0.013,
             r"$1/4-\ln 2$",
             fontsize=8, color=COLORS["U"], ha="left")
    # Reference Majumdar--Ghosh exact energy per site E/N = -3/8.
    axR.axhline(-3.0 / 8.0, color=COLORS["D"], linestyle=":", linewidth=0.8)
    axR.text(0.0035, -3.0 / 8.0 + 0.005,
             r"$-3/8$",
             fontsize=8, color=COLORS["D"], ha="left")
    axR.set_xlabel(r"$1/N$")
    axR.set_ylabel(r"$E_{0}/N$")
    axR.set_title(r"Energy per site -- finite-size scaling")
    axR.set_ylim(-0.47, -0.17)
    axR.set_xlim(0.0, 0.07)
    axR.grid(True, which="both", alpha=0.25, linewidth=0.4)
    axR.legend(loc="upper left", frameon=False, handlelength=1.6, fontsize=8)

    fig.tight_layout()
    save_pair(fig, "fig4_diagnostics")


def dump_csv(D: dict) -> None:
    """Tidy CSV of every plotted point (for the SI / reproducibility)."""
    out = FIG_DIR / "T0_summary.csv"
    rows = ["phase,N,FQ,Aeff,E0,E0_per_N,chi_final"]
    for ph in PHASES:
        for N in sorted(D[ph].keys()):
            r = D[ph][N]
            Aeff = r["FQ"] / np.log(N) ** 1.5
            rows.append(f"{ph},{N},{r['FQ']:.10g},{Aeff:.10g},"
                        f"{r['E0']:.10g},{r['E0']/N:.10g},{r['chi']}")
    out.write_text("\n".join(rows) + "\n")
    print(f"  wrote {out.relative_to(FIG_DIR.parent)}")


def main() -> None:
    D = load_T0()
    counts = {p: len(D[p]) for p in PHASES}
    print(f"loaded T=0 summaries: {counts}  -> writing PDFs to {FIG_DIR}")
    fig1_FQ_vs_N(D)
    fig2_wzw_test(D)
    fig3_effective_A(D)
    fig4_diagnostics(D)
    dump_csv(D)
    print("done.")


if __name__ == "__main__":
    main()
