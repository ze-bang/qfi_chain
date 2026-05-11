# QFI_CHAIN — DMRG Extension Project Plan

## 0. Context

The LaTeX manuscript [`latex/qfi_geometric_extraction.tex`](latex/qfi_geometric_extraction.tex)
proves and numerically verifies (at $N=8,12$ via ED) the geometric
matched-filter hierarchy
$$\eta^{\star,\rm conn}_{\le k}=\frac{\mathbf s^\top \Sigma^+ \mathbf s}{F_Q}$$
on the staggered probe $G=S^z_\pi$ for the $J_1$–$J_2$–$J_3$ spin-1/2
chain along the closed loop $(D)\to(C)\to(U)\to(D)$:

- $(D)$ Majumdar–Ghosh dimer point $(J_1,J_2,J_3)=(1,1/2,0)$
- $(C)$ Cluster product $(J_1,J_2,J_3)\approx(0.241,0.451,0.308)$
- $(U)$ Uniform Heisenberg point $(J_1,J_2,J_3)=(1,0,0)$

The paper makes three sharp **scaling predictions** that ED at
$N=8,12$ cannot test:

1. **WZW marginal scaling at $(U)$:**
   $F_Q(N)\sim \tfrac{4}{3}(\ln N)^{3/2}$ — Eq. (FQ_WZW).
2. **Body-truncation collapse:** $\eta^{\star,\rm conn}_{\le k}(U)\to 0$
   as $N\to\infty$ for any fixed $k$, with rate $(\ln N)^{-3/2}$ —
   Eq. (eta_fixed_k_to_zero).
3. **Required body order grows extensively:** $k_\star(N)\sim N$ —
   Eq. (kstar_WZW).

The dimer/cluster phases also have predictions:
- $\eta^{\star,\rm conn}_{\le 2}(D)=1$ for **all** $N$ (single-bond
  saturation, Theorem `thm:saturation`).
- $\eta^{\star,\rm conn}_{\le 4}(C)\to 1$ with $4$-cluster fidelity
  $F_4\to 1$ as the gap protects clustering.

This project uses **DMRG / MPS-based methods** (ITensor.jl) to test
predictions 1–3 at $N\in\{16,24,32,48,64,96,128\}$ (open boundary)
at both **$T=0$** (standard DMRG) and **$T>0$** (METTS / purification),
for both the **quantum Fisher information $F_Q$** and the **error-propagation
Fisher $F_{\rm EP}[O]=|\partial_\theta\langle O\rangle|^2/\Var(O)$** of the
matched filter $O=O^\star_{\le k}$.  At finite $T$ the relevant
inequality is
$$F_{\rm EP}[O;\rho_T]\;\le\;F_Q[\rho_T]\;\le\;F_Q[\rho_{T=0}],$$
so the same hierarchy $\eta^{\star,\rm conn}_{\le k}=F_{\rm EP}/F_Q$
is well-defined at every $T$ and tests how thermal mixing degrades
the matched-filter saturation.

We also confirm $\eta^{\star,\rm conn}_{\le 2}(D)=1$ at $T=0$
for every $N$ (Theorem `thm:saturation`).

---

## 1. Deliverables

### 1.1 $T=0$ track (ground-state DMRG)

| # | Deliverable | Quantity | Sizes | Output |
|---|---|---|---|---|
| D1 | Ground state $|\psi_0(N)\rangle$ at $(D),(C),(U)$ | 3 phases × 7 sizes | $N=16,24,32,48,64,96,128$ | `states/T0/<phase>_N<N>.h5` (MPS) |
| D2 | $F_Q^{T=0}(N)$ for $G=S^z_\pi$ | 3 phases × 7 sizes | as D1 | `data/FQ_T0_scaling.csv` |
| D3 | $F_{\rm EP}^{T=0}[O^\star_{\le k}]$ and $\eta^{\star,\rm conn}_{\le k}(N)$ for $k\in\{2,4,6,8\}$ | 3 phases × 7 sizes × 4 cutoffs | as D1 | `data/FEP_T0_scaling.csv`, `data/eta_T0_scaling.csv` |
| D4 | Cluster fidelities $F_2,F_4,F_8$ at $(C)$ | 1 phase × 7 sizes | as D1 | `data/cluster_fidelity_C.csv` |
| D5 | (D) sanity check: $\eta^{\star,\rm conn}_{\le 2}=1$ exactly | 1 phase × 7 sizes | as D1 | `data/D_saturation_check.csv` |

### 1.2 $T>0$ track (METTS finite-temperature MPS)

| # | Deliverable | Quantity | Sizes / temperatures | Output |
|---|---|---|---|---|
| D6 | METTS ensemble $\{|\phi^{(s)}_T\rangle\}$ at $(D),(C),(U)$ | 3 phases × 4 sizes × 6 temperatures × $M$ samples | $N=16,32,48,64$; $T/J_1\in\{0.05,0.1,0.2,0.4,0.8,1.6\}$; $M=200$ | `states/Tfin/<phase>_N<N>_T<T>.h5` |
| D7 | $F_Q(T,N)$ from the convex-roof / SLD on the METTS ensemble | as D6 | as D6 | `data/FQ_Tfin.csv` |
| D8 | $F_{\rm EP}[O^\star_{\le k};\rho_T]$ and $\eta^{\star,\rm conn}_{\le k}(T,N)$ for $k\in\{2,4,6,8\}$ | as D6 × 4 cutoffs | as D6 | `data/FEP_Tfin.csv`, `data/eta_Tfin.csv` |
| D9 | Cross-over temperature $T_\star(N)$ where $\eta_{\le k=2}(T,N)$ falls below 0.5 | 3 phases × 4 sizes | as D6 | `data/Tstar_scaling.csv` |

### 1.3 Synthesis

| # | Deliverable | Output |
|---|---|---|
| D10 | Figures: $F_Q$ vs $\ln N$ at $T=0$; $F_Q(T)/F_Q(0)$ vs $T/J_1$; $\eta^{\star,\rm conn}_{\le k}(T,N)$ heatmaps; $\eta_{\le 2}(T)$ collapse plots | `figures/FQ_lnN_T0.pdf`, `figures/FQ_T_decay.pdf`, `figures/eta_T_heatmap_<k>.pdf`, `figures/eta_T_collapse.pdf` |
| D11 | New LaTeX section §6.6 (“Finite-temperature scaling”) | inline edit of `qfi_geometric_extraction.tex` |

**Stretch (only if D1–D11 finish on time):**

| D12 | Continuous loop scan across $(D)\to(C)\to(U)\to(D)$ at $N=32$, $T\in\{0,0.1,0.4\}$ | 24 path points | `data/loop_scan_TN32.csv` + figure |

---

## 2. Methodology

### 2.1 Software stack

- **Production library**: **ITensor.jl** (Julia) for both $T=0$ DMRG
  *and* $T>0$ METTS. Same `MPS` data structure, same `OpSum`
  Hamiltonian builder, identical bond-dimension and truncation
  controls.
- **Post-processing**: Python (`numpy`, `scipy`, `h5py`) reads the
  MPS files written by ITensor's `HDF5.h5open`. Optional `quimb`
  loaded only as a sanity-check backend on small $N$.
- **Why ITensor.jl over `quimb`**: U(1)-symmetric block sparse MPS
  is mature in ITensor.jl, METTS sampling has a reference
  implementation in `ITensorMPS.jl`, and Julia threading scales
  well to $\chi=1200$ on cluster nodes.

### 2.2 $T=0$ ground-state DMRG

- **Boundary conditions**: open (OBC). The WZW $(\ln N)^{3/2}$ is
  unaffected by BC up to subleading corrections; OBC dramatically
  reduces bond dimension.
- **Bond dimension** $\chi_\max$ schedule
  - $N\le 32$: $\chi_\max=400$
  - $N=48,64$: $\chi_\max=800$
  - $N=96,128$: $\chi_\max=1200$
- **Sweeps**: 30 with adaptive truncation $\epsilon_{\rm trunc}=10^{-10}$.
- **Convergence**: $|E_n-E_{n-1}|<10^{-9}$ AND truncation error
  $<10^{-9}$ for last 3 sweeps.
- **Symmetries**: enforce $S^z_{\rm tot}=0$ (U(1) block), parity if free.

### 2.3 $T>0$ METTS finite-temperature MPS

Minimally entangled typical thermal states (METTS, White 2009;
Stoudenmire & White 2010) sample
$\rho_T=Z^{-1}e^{-\beta H}$ as a Markov chain over product-state
“classical” configurations $|n\rangle$, propagated by imaginary-time
evolution.

- **Algorithm** per Markov step:
  1. start from a random product state $|n\rangle$ in the
     $S^z_{\rm tot}=0$ sector;
  2. apply $e^{-\beta H/2}$ by TDVP with $\Delta\tau=0.025/J_1$ to
     produce $|\phi^{(s)}\rangle\propto e^{-\beta H/2}|n\rangle$;
  3. measure observables on $|\phi^{(s)}\rangle$;
  4. collapse $|\phi^{(s)}\rangle$ in a randomly chosen product
     basis ($S^z$ or $S^x$ alternating) to get the next
     $|n\rangle$.
- **Burn-in**: 30 steps discarded.
- **Production**: $M=200$ samples per $(N,T,\text{phase})$ triple.
- **Bond dimension**: $\chi_\max=600$ for $N\le 48$,
  $\chi_\max=900$ for $N=64$ (METTS bond dimension is typically
  *smaller* than ground-state DMRG at the same $N$).
- **Temperature ladder**: $T/J_1\in\{0.05,0.1,0.2,0.4,0.8,1.6\}$,
  i.e.\ $\beta J_1\in\{20,10,5,2.5,1.25,0.625\}$.
- **Symmetries**: U(1) preserved through the TDVP (start state and
  collapse basis are $S^z_{\rm tot}=0$ projected).
- **Statistical error**: report $1/\sqrt M$ jackknife on every
  ensemble average; require $\delta F_Q/F_Q<5\%$ before reporting.

The per-sample cost is one TDVP sweep across imaginary time
$\beta/2$; total wall time per $(N,T)$ point is
$M\times O(\beta\,N\,\chi^2)$ — see §5.

### 2.4 Observables (uniform interface $T=0$ ↔ $T>0$)

For each MPS $|\psi\rangle$ — either the DMRG ground state
$|\psi_0\rangle$ or a single METTS sample $|\phi^{(s)}_T\rangle$ —
compute the same set of MPS expectation values:

1. **One- and two-point**: $\langle S^a_i\rangle$ and
   $C^{ab}_{ij}=\langle S^a_i S^b_j\rangle$ for $a,b\in\{x,y,z\}$,
   $i<j$. Cost $O(N^2\chi^3)$.
2. **Three-point**: $\langle S^a_i S^b_j S^c_l\rangle$ — needed for
   the matched-filter signal $s_r=-i\langle[\mathcal O^{XY,r}_\pi,G]\rangle$.
3. **Four-point**: $\langle S^a_iS^b_jS^c_lS^d_m\rangle$ for
   contiguous and short-range disjoint quadruples — needed for the
   covariance $\Sigma_{rs}$.
4. **Body-4 augmentation** ($k\ge 4$): extra contractions for the
   staggered double-bond hop $\mathcal Q_\pi$ (Eq. `Qpi`).
5. **Cluster fidelities** at $(C)$ ($T=0$ only):
   $F_p=|\langle\psi_0|\psi^{\rm prod}_p\rangle|^2$ via boundary-MPS
   overlap.

### 2.5 Assembly: $T=0$ vs $T>0$

**$T=0$ ($|\psi_0\rangle$ pure):**
$$F_Q[|\psi_0\rangle;G]=4\,\Var_{\psi_0}(G)=4\sum_{ij}(-1)^{i+j}
   \bigl(C^{zz}_{ij}-\langle S^z_i\rangle\langle S^z_j\rangle\bigr).$$

**$T>0$ ($\rho_T$ mixed):** the SLD-based QFI is
$$F_Q[\rho_T;G]=2\sum_{m,n}\frac{(p_m-p_n)^2}{p_m+p_n}\,|\langle m|G|n\rangle|^2,$$
which is **not** directly accessible from pure-state METTS samples.
We estimate it via the **convex-roof / lower-bound** route:
- Each METTS sample gives $F_Q^{(s)}\equiv 4\Var_{\phi^{(s)}}(G)$.
- The METTS-averaged variance $4\langle\Var_\phi(G)\rangle_M$ is a
  *strict upper bound* on the symmetric-logarithmic QFI of
  $\rho_T$ in the limit $M\to\infty$ (White 2009; Tsang 2013).
- The matched-filter Fisher
  $F_{\rm EP}[O;\rho_T]=|\partial_\theta\langle O\rangle_T|^2/\Var_T(O)$
  is computed *directly* on the ensemble (no convex roof needed):
  $\langle O\rangle_T=\langle\langle O\rangle_\phi\rangle_M$ and
  $\Var_T(O)=\langle\langle O^2\rangle_\phi\rangle_M-\langle O\rangle_T^2$.
- The signal slope $\partial_\theta\langle O\rangle_T$ is computed by
  finite-difference METTS at three values of $\theta=\theta_0\pm\epsilon$
  *or*, equivalently and more cheaply, as
  $\partial_\theta\langle O\rangle_T=-i\langle[O,G]\rangle_T$
  (linear-response identity at $\theta=0$).

**Summary of the saturation ratio**:
$$\eta^{\star,\rm conn}_{\le k}(T,N)=
   \frac{F_{\rm EP}[O^\star_{\le k};\rho_T]}{F_Q[\rho_T]},$$
with both numerator and denominator estimated on the same METTS
ensemble; the ratio is more stable than either separately.

### 2.6 Wiener solve

At every $(T,N,\text{phase})$ assemble $\mathbf s, \Sigma$ in the
symmetry-adapted basis $\{\mathcal O^{XY,r}_\pi\}_{r=1}^{k-1}\cup\{\mathcal Q_\pi\}$,
solve $\Sigma\mathbf c=\mathbf s$ by pseudo-inverse (regularize
$\Sigma\to\Sigma+\lambda I$ with $\lambda=10^{-10}\,\mathrm{tr}\,\Sigma/\dim$),
and evaluate
$$F_{\rm EP}[O^\star_{\le k}]=\mathbf s^\top\Sigma^+\mathbf s,\qquad
  \eta^{\star,\rm conn}_{\le k}=\frac{F_{\rm EP}[O^\star_{\le k}]}{F_Q}.$$

### 2.7 Scaling fits

- **WZW $T=0$**: fit $F_Q^{T=0}(N)=A(\ln N)^{3/2}+B$ on $N\ge 32$.
  Predict $A=4/3$.
- **Body collapse $T=0$**: fit
  $\eta^{\star,\rm conn}_{\le k}(N)=a_k(\ln N)^{-3/2}+b_k$ on
  $N\ge 32$ for each $k\in\{2,4,6,8\}$. Predict $b_k=0$.
- **Required $k_\star(N)$**: smallest $k$ with $\eta\ge 0.9$;
  predict $k_\star\sim N$.
- **Thermal decay**: at $(U)$ fit
  $F_Q(T)/F_Q(0)=g(T\ln N)$ for some collapse function $g$
  (heuristic from the CFT $\beta\to L$ duality); look for the
  expected $T$-vanishing of the WZW log enhancement.
- **$\eta(T)$ collapse**: at fixed $k$ test whether
  $\eta^{\star,\rm conn}_{\le k}(T,N)$ collapses as a function of
  the dimensionless ratio $T/T_\star(N)$ where $T_\star(N)$ is the
  half-saturation temperature.

---

## 3. Validation gates

### $T=0$ track
- **G1**: Truncation error $<10^{-8}$ at $\chi_\max$ for all sites.
- **G2**: $E_0$ converged to $10^{-7}$ relative across last 3 sweeps.
- **G3**: At $N=16$, DMRG observables agree with ED reference
  ([`data/j1j2j3_hierarchy_N8.npz`](data/j1j2j3_hierarchy_N8.npz)
  re-run for $N=16$) to $10^{-6}$.
- **G4**: At $(D)$ all $N$: $\eta^{\star,\rm conn}_{\le 2}=1\pm 10^{-6}$
  (analytic, Theorem `thm:saturation`).

### $T>0$ track
- **G5**: METTS energy estimator agrees with TPQ/exact at $N=16$
  for all $T$ in the ladder to $1\%$ relative.
- **G6**: METTS jackknife error on $F_Q(T,N)$ below $5\%$ for every
  reported point.
- **G7**: $T\to 0$ limit of METTS-$F_Q$ matches the $T=0$ DMRG value
  (D2) at the lowest $T=0.05J_1$ point to $5\%$.
- **G8**: Two estimators of $\partial_\theta\langle O\rangle_T$ agree
  within statistical error: finite-difference METTS at
  $\theta=\pm 10^{-3}$ vs.\ the linear-response identity
  $-i\langle[O,G]\rangle_T$.

If any gate fails: $T=0$ → increase $\chi_\max$ by $\times 1.5$;
$T>0$ → first double $M$ (more samples), then increase $\chi_\max$.

---

## 4. Code layout (target after the run)

```
QFI_CHAIN/
├── dmrg/
│   ├── hamiltonian.jl           # OpSum builder for J1-J2-J3 (D),(C),(U)
│   ├── run_dmrg_T0.jl           # ITensor.jl ground-state DMRG (D1)
│   ├── run_metts_Tfin.jl        # ITensor.jl METTS sampler (D6)
│   ├── compute_observables.py   # MPS HDF5 → one/two/three/four-point
│   ├── assemble_FQ_FEP.py       # build F_Q, F_EP, η at (T=0 or T>0)
│   ├── solve_wiener.py          # Σc=s with regularization
│   ├── scaling_fits.py          # WZW + thermal collapse fits + figures
│   ├── check_gates.py           # G1–G8 automated checks
│   └── README.md
├── data/                        # CSV outputs from D2–D9
├── figures/                     # PDF outputs (D10)
├── scripts/                     # ED scripts (kept, used as reference + G3,G7)
└── latex/qfi_geometric_extraction.tex   # extended with §6.6 (D11)
```

Implementation order:

1. `hamiltonian.jl` + `run_dmrg_T0.jl` — produce D1.
2. `compute_observables.py` + `assemble_FQ_FEP.py` — produce D2–D5.
3. `run_metts_Tfin.jl` — produce D6.
4. Re-use steps 2 on METTS HDF5 — produce D7–D9.
5. `scaling_fits.py` — produce D10 figures.
6. Hand-edit §6.6 of LaTeX — D11.
7. (Optional) loop scan — D12.

---

## 5. Compute budget estimate

### 5.1 $T=0$ DMRG (per phase)

| $N$ | $\chi_\max$ | mem/state | DMRG wall | Obs wall | Walltime |
|---|---|---|---|---|---|
| 16 | 400 | 0.05 GB | 5 min | 1 min | 6 min |
| 32 | 400 | 0.2 GB | 30 min | 5 min | 35 min |
| 48 | 800 | 1 GB | 2 h | 20 min | 2.5 h |
| 64 | 800 | 1.5 GB | 4 h | 1 h | 5 h |
| 96 | 1200 | 5 GB | 12 h | 4 h | 16 h |
| 128 | 1200 | 8 GB | 24 h | 8 h | 32 h |

Subtotal: ≈60 h per phase; 3 phases × 60 h ≈ **180 CPU-h ($T=0$)**.

### 5.2 $T>0$ METTS (per phase)

$M=200$ samples per $(N,T)$ point; 6 temperatures; 4 sizes
$N\in\{16,32,48,64\}$.

Wall time per sample $\approx \beta\,\tau_{\rm TDVP}$ where
$\tau_{\rm TDVP}\approx 5\,\mathrm{s}$ at $N=32,\chi=600$, scaling
$\propto N\chi^2$:

| $N$ | $\chi_\max$ | $\tau_{\rm TDVP}$ (s/step) | wall / sample @ $\beta J_1=20$ | wall / $(N,T)$ point ($M=200$) |
|---|---|---|---|---|
| 16 | 600 | 2 | 80 s | 4.4 h |
| 32 | 600 | 5 | 200 s | 11 h |
| 48 | 600 | 10 | 400 s | 22 h |
| 64 | 900 | 25 | 1000 s | 56 h |

(Higher $T$ is cheaper because $\beta$ is smaller; sum over the 6
temperatures is roughly $1.5\times$ the lowest-$T$ row.)
Total $T>0$ per phase ≈ $1.5\times(4.4+11+22+56)$ h $\approx 140$ h;
across 3 phases and parallelized over $M$ samples and $T$
independently ≈ **420 CPU-h ($T>0$)** but very embarrassingly
parallel: 200 samples × 6 temps × 4 sizes × 3 phases = 14400
independent jobs, max wall per job ≤ 1 h.

### 5.3 Grand total

$\approx$ **600 CPU-h** total; $T>0$ jobs are almost all
$\le 1\,\text{h}$ wall, so a 256-core allocation finishes the
METTS production in $\approx 6$ h elapsed.  Memory peak ≈ 16 GB
/ $T=0$ job at $N=128$, $\le 8$ GB / METTS job.

See [`CLUSTER_SUBMISSION.md`](CLUSTER_SUBMISSION.md) for SLURM details.
