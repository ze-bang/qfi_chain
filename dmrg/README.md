# dmrg/ — runnable QFI / F_EP pipeline

Two-track DMRG + METTS workflow that produces the deliverables D1–D11 from
[`../PROJECT_PLAN.md`](../PROJECT_PLAN.md).

## Layout

| File                      | Role                                                  |
|---------------------------|-------------------------------------------------------|
| `Project.toml`            | Julia env (ITensor, ITensorMPS, ITensorTDVP, JSON3)   |
| `requirements.txt`        | Python env (numpy, scipy, h5py, pandas, matplotlib)   |
| `hamiltonian.jl`          | J1-J2-J3 OpSum/MPO + phase presets D, C, U            |
| `observables.jl`          | One-/two-point + body-2 filter + raw FQ, s_r, Σ_rr    |
| `run_dmrg_T0.jl`          | T=0 ground-state DMRG → JSON                          |
| `run_metts_Tfin.jl`       | METTS sample @ (phase,N,T,s) → JSON                   |
| `aggregate_metts.py`      | Average M samples; jackknife errors                   |
| `assemble_FQ_FEP.py`      | Wiener solve → F_EP, η for k ∈ {2,4,6,8}              |
| `scaling_fits.py`         | CSVs + figures (WZW fit, η(T) collapse)               |
| `check_gates.py`          | G1–G8 validation                                      |
| `slurm/submit_*.slurm`    | Cluster job templates                                 |

## Local quick test (before pushing to cluster)

```bash
# Julia env
julia --project=dmrg -e 'using Pkg; Pkg.instantiate()'
# Python env
python -m venv .venv && . .venv/bin/activate && pip install -r dmrg/requirements.txt

cd dmrg
# Tiny T=0 run
julia --project=. run_dmrg_T0.jl --phase D --N 16 --chi-max 200 --sweeps 20 \
    --r-max 5 --out obs/T0/D_N16.json
python assemble_FQ_FEP.py --input obs/T0/D_N16.json --ks 2,4 \
    --out obs/T0/D_N16_summary.json
```

For cluster usage see [`../CLUSTER_SUBMISSION.md`](../CLUSTER_SUBMISSION.md).

## Output structure

```
dmrg/
├─ obs/T0/<phase>_N<N>.json                       # raw observables
├─ obs/T0/<phase>_N<N>_summary.json               # FQ, FEP, η table
├─ obs/Tfin/raw/<phase>_N<N>_T<T>_s<s>.json       # METTS samples
├─ obs/Tfin/<phase>_N<N>_T<T>.json                # ensemble-averaged
├─ obs/Tfin/<phase>_N<N>_T<T>_summary.json
└─ logs/                                          # SLURM stdout/stderr
data/    # CSVs from scaling_fits.py
figures/ # PDFs from scaling_fits.py
```
