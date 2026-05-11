#!/usr/bin/env python3
"""Plot the wavefunction-entanglement diagnostics produced by
wavefunction_entanglement_DCU.py."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)


def load(N: int):
    return np.load(DATA / f"wavefunction_entanglement_DCU_N{N}.npz",
                   allow_pickle=True)


def plot_one(N: int):
    d = load(N)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    colors = {"D": "#1f77b4", "C": "#ff7f0e", "U": "#2ca02c"}
    pretty = {
        "D": r"(D) Dimer  $(1,\epsilon,\epsilon)$",
        "C": r"(C) Cluster $(1,1,\epsilon)$",
        "U": r"(U) Uniform $(1,1,1)$",
    }

    # Panel 1: bipartite entropy S(L)
    ax = axes[0]
    Ls = np.arange(1, N)
    for lab in ("D", "C", "U"):
        SL = d[f"{lab}_SL"]
        ax.plot(Ls, SL, "o-", color=colors[lab], lw=1.6, ms=5,
                label=pretty[lab])
    ax.axhline(np.log(2), ls=":", color="gray", lw=0.8)
    ax.text(N - 1.0, np.log(2) + 0.03, r"$\ln 2$", color="gray",
            fontsize=8, ha="right")
    ax.set_xlabel(r"contiguous cut $L$")
    ax.set_ylabel(r"$S(L)$  (von Neumann)")
    ax.set_title(rf"(a) Bipartite entropy, $N={N}$")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)

    # Panel 2: bond singlet fidelity by bond class
    ax = axes[1]
    classes = ("dimer", "bridge", "inter")
    x = np.arange(len(classes))
    width = 0.25
    for i, lab in enumerate(("D", "C", "U")):
        bc = d[f"{lab}_bond_class"]
        Fs = d[f"{lab}_F_singlet"]
        means = [Fs[bc == c].mean() if (bc == c).any() else np.nan
                 for c in classes]
        ax.bar(x + (i - 1) * width, means, width, color=colors[lab],
               label=pretty[lab])
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(r"$\langle s |\,\rho_{j,j+1}\,| s\rangle$")
    ax.set_title(r"(b) Singlet fidelity by bond class")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Panel 3: best k-producible product fidelity
    ax = axes[2]
    for lab in ("D", "C", "U"):
        ks = d[f"{lab}_F_k_keys"]
        Fk = d[f"{lab}_F_k_vals"]
        ax.plot(ks, Fk, "o-", color=colors[lab], lw=1.8, ms=6,
                label=pretty[lab])
    ax.set_xlabel(r"window size $k$")
    ax.set_ylabel(r"$\max_\Pi |\langle\psi|\prod_a|\phi_a^*\rangle|^2$")
    ax.set_title(r"(c) Best $k$-producible product fidelity")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.suptitle(rf"$J_1$-$J_2$-$J_3$ Heisenberg ground state, $N={N}$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_pdf = FIGS / f"wavefunction_entanglement_DCU_N{N}.pdf"
    out_png = FIGS / f"wavefunction_entanglement_DCU_N{N}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    for N in (8, 12):
        plot_one(N)
