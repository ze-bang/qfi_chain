#!/usr/bin/env python3
"""Plot of the connected-window matched-filter saturation
$\\eta^{\\star,\\mathrm{conn}}_{\\le k}$ on the cluster-GHZ family
$|\\psi^{(p)}\\rangle$ for $p\\in\\{2,4,8\\}$, $N=8$.

The signature is a sharp Heaviside step at $k=p$:
$\\eta^{\\star,\\mathrm{conn}}_{\\le k}(\\psi^{(p)})=\\Theta(k-p)$,
which is the operator-side proof of the Body-Order Gap theorem.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)


def main(N: int = 8, sizes=(2, 4, 8)):
    d = np.load(DATA / f"ghz_cluster_hierarchy_N{N}.npz")
    k_max = int(d["k_max"])
    k = np.arange(1, k_max + 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / max(1, len(sizes) - 1)) for i in range(len(sizes))]
    for ps, c in zip(sizes, colors):
        eta = d[f"p{ps}_eta_span"]
        ax.plot(k, eta, "o-", color=c, lw=1.8, ms=6,
                label=rf"$|\psi^{{({ps})}}\rangle$")
    ax.axhline(1.0, color="0.4", lw=0.6, ls=":")
    ax.set_xlabel(r"window size $k$")
    ax.set_ylabel(r"$\eta^{\star,\mathrm{conn}}_{k}\;=\;F_{\mathrm{EP}}/F_Q$")
    ax.set_title(rf"Cluster-GHZ states ($N={N}$, $G=\sum_j\sigma^z_j$)")
    ax.set_xticks(k)
    ax.set_ylim(-0.05, 1.10)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", frameon=False, fontsize=10)
    fig.tight_layout()
    out_pdf = FIGS / f"eta_conn_ghz_cluster_N{N}.pdf"
    out_png = FIGS / f"eta_conn_ghz_cluster_N{N}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved -> {out_pdf}\nSaved -> {out_png}")


if __name__ == "__main__":
    main()
