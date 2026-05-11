# QFI_CHAIN

DMRG (T=0) + METTS (T>0) validation of the geometric matched-filter
hierarchy
$$\eta^{\star,\rm conn}_{\le k}=F_{\rm EP}[O^\star_{\le k}]/F_Q$$
on the staggered probe $G=S^z_\pi$ for the $J_1$–$J_2$–$J_3$
spin-1/2 chain.

Manuscript: [`latex/qfi_geometric_extraction.pdf`](latex/qfi_geometric_extraction.pdf)
(source: [`latex/qfi_geometric_extraction.tex`](latex/qfi_geometric_extraction.tex)).

Plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md).
Cluster ops: [`CLUSTER_SUBMISSION.md`](CLUSTER_SUBMISSION.md).
DMRG/METTS implementation: [`dmrg/README.md`](dmrg/README.md).

## Layout

```
QFI_CHAIN/
├── PROJECT_PLAN.md           # Deliverables, methodology, gates
├── CLUSTER_SUBMISSION.md     # SLURM + env setup
├── latex/                    # Manuscript (pdflatex)
├── scripts/                  # ED reference scripts (small N)
├── data/                     # ED reference .npz + DMRG CSV outputs
├── figures/                  # Manuscript and DMRG figures
└── dmrg/                     # ITensor.jl + Python pipeline
    ├── Project.toml          # Julia env
    ├── requirements.txt      # Python env
    ├── hamiltonian.jl        # J1J2J3 OpSum builder
    ├── observables.jl        # Shared MPS expectation values
    ├── run_dmrg_T0.jl        # Ground-state DMRG (D1)
    ├── run_metts_Tfin.jl     # METTS sampler (D6)
    ├── aggregate_metts.py    # Average samples + jackknife (D7-D9)
    ├── assemble_FQ_FEP.py    # Wiener solve → η, F_Q, F_EP
    ├── scaling_fits.py       # WZW + thermal collapse fits
    ├── check_gates.py        # G1–G8 automated checks
    └── slurm/                # Submission scripts
```

## Quick start (cluster)

```bash
git clone <YOUR_REMOTE>/QFI_CHAIN.git
cd QFI_CHAIN
# follow CLUSTER_SUBMISSION.md for env setup, then:
cd dmrg/slurm
JID_GS=$(sbatch --parsable submit_gs.slurm)
JID_OBS_T0=$(sbatch --parsable --dependency=afterok:$JID_GS submit_obs_T0.slurm)
JID_METTS=$(sbatch --parsable submit_metts.slurm)
JID_OBS_TFIN=$(sbatch --parsable --dependency=afterok:$JID_METTS submit_obs_Tfin.slurm)
sbatch --dependency=afterok:$JID_OBS_T0:$JID_OBS_TFIN submit_fits.slurm
```
