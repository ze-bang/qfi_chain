#!/usr/bin/env python3
"""Plot the connected-subspace matched-filter saturation
$\\eta^{\\star,\\mathrm{conn}}_{\\le k} = F_{\\rm EP}/F_Q$ along the
closed loop dimer -> cluster -> uniform on the J1-J2-J3 chain.

On a 1D chain a connected k-subgraph is a contiguous window, so the
$\\mathcal V^{\\rm conn}_{\\le k}$ matched filter is exactly the
linear span of all Hermitian operators on contiguous windows of size
$\\le k$, which is `eta_span` in j1j2j3_hierarchy_N{N}.npz.

We display k = 2, 4, 6, 8.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)


def plot_one(N: int, ks_to_show=(2, 4, 6, 8)):
    d = np.load(DATA / f"j1j2j3_hierarchy_N{N}.npz")
    eta = d["eta_span"]                  # (n_pts, k_max)
    leg_ends = d["leg_ends"]
    k_max = int(d["k_max"])
    n_pts = eta.shape[0]
    x = np.arange(n_pts)

    ks = [k for k in ks_to_show if k <= k_max]

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(ks) - 1)) for i in range(len(ks))]
    # distinct markers/linestyles + tiny vertical offsets so overlapping
    # saturated curves remain individually visible
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]
    offsets = np.linspace(-0.012, 0.012, len(ks))

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for k, c, mk, ls, off in zip(ks, colors, markers, linestyles, offsets):
        ax.plot(x, eta[:, k - 1] + off, marker=mk, linestyle=ls,
                color=c, lw=1.7, ms=5.0, mfc="none", mew=1.4,
                label=rf"$k={k}$")

    for xp in leg_ends[:-1]:
        ax.axvline(xp - 0.5, color="0.7", lw=0.7, ls="--")
    label_pts = {0: "(D)", int(leg_ends[0]): "(C)",
                 int(leg_ends[1]): "(U)", n_pts - 1: "(D)"}
    for xp, lab in label_pts.items():
        ax.text(xp, 1.06, lab, ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(1.0, color="0.4", lw=0.6, ls=":")
    ax.set_xlabel(r"path index along  $D\to C\to U\to D$")
    ax.set_ylabel(
        r"$\eta^{\star,\mathrm{conn}}_{k}\;=\;F_{\mathrm{EP}}/F_Q$"
    )
    ax.set_title(
        rf"Connected-subspace matched-filter saturation, "
        rf"$J_1$-$J_2$-$J_3$ chain ($N={N}$)"
    )
    ax.set_ylim(-0.02, 1.13)
    ax.set_xlim(-0.5, n_pts - 0.5)
    ax.legend(loc="lower right", ncol=len(ks), fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_pdf = FIGS / f"eta_conn_strongest_path_N{N}.pdf"
    out_png = FIGS / f"eta_conn_strongest_path_N{N}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")

    # Also print the numerical values for the three reference points.
    print("\nNumerical values at reference points:")
    print(f"{'pt':>4s}  " + "  ".join(f"k={k:>2d}" for k in ks))
    for label, ip in [("(D)", 0), ("(C)", int(leg_ends[0])),
                      ("(U)", int(leg_ends[1])), ("(D')", n_pts - 1)]:
        vals = "  ".join(f"{eta[ip, k - 1]:5.3f}" for k in ks)
        print(f"{label:>4s}  {vals}")


if __name__ == "__main__":
    plot_one(8)
