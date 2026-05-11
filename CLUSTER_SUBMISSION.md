# QFI_CHAIN DMRG Cluster Submission Notes

Target cluster: SLURM (Cedar / Niagara / Rorqual or equivalent).
Project: `QFI_CHAIN` — **ITensor.jl-based** DMRG / METTS validation
of both the **quantum Fisher information $F_Q$** and the
**error-propagation Fisher $F_{\rm EP}[O^\star_{\le k}]$** (and their
ratio $\eta^{\star,\rm conn}_{\le k}$) at **$T=0$** *and* at
**$T>0$** for the $J_1$–$J_2$–$J_3$ chain at $(D),(C),(U)$.

> Read [`PROJECT_PLAN.md`](PROJECT_PLAN.md) first — this file is the
> operational checklist for getting a run on the cluster.

---

## 1. Pre-flight checklist (local machine)

```bash
cd $HOME/exact_diagonalization_clean/QFI_CHAIN

# Sanity: project tree
tree -L 2 -I '__pycache__'

# Sanity: LaTeX compiles
cd latex && pdflatex -interaction=nonstopmode qfi_geometric_extraction.tex \
  && pdflatex -interaction=nonstopmode qfi_geometric_extraction.tex
cd ..

# Sanity: ED reference at N=8 reproduces the published numbers
python scripts/j1j2j3_hierarchy.py --N 8 --out data/j1j2j3_hierarchy_N8.npz
```

If anything above fails: do not submit. Fix locally first.

---

## 2. Cluster environment

### 2.1 Modules (Compute Canada / Alliance example)

```bash
module purge
module load StdEnv/2023
module load julia/1.10.4
module load python/3.11
module load hdf5/1.14
module load openblas/0.3.24
```

### 2.2 Python venv (one-time)

```bash
cd $SCRATCH
python -m venv qfi_venv
source qfi_venv/bin/activate
pip install --upgrade pip
pip install numpy scipy h5py quimb tqdm matplotlib pandas
```

### 2.3 Julia / ITensor.jl (one-time)

```bash
julia --project=$HOME/qfi_julia -e '
  using Pkg
  Pkg.add(["ITensors", "ITensorMPS", "ITensorTDVP", "HDF5", "JSON3", "ArgParse"])
  Pkg.precompile()
'
```

`ITensorMPS` and `ITensorTDVP` are required for METTS
(real/imaginary-time TDVP and projected-collapse helpers). All
$T=0$ and $T>0$ jobs share the same `hamiltonian.jl` MPO builder.

### 2.4 Stage the project on $SCRATCH

```bash
RSYNC_OPTS="-av --exclude=__pycache__ --exclude=*.pdf --exclude=*.aux --exclude=*.log"
rsync $RSYNC_OPTS $HOME/exact_diagonalization_clean/QFI_CHAIN/ $SCRATCH/QFI_CHAIN/
mkdir -p $SCRATCH/QFI_CHAIN/dmrg/{logs,states/T0,states/Tfin,obs/T0,obs/Tfin}
```

---

## 3. SLURM job templates

Four job arrays. Save as
`$SCRATCH/QFI_CHAIN/dmrg/submit_<name>.slurm`:
- `submit_gs.slurm`     — $T=0$ DMRG ground states (D1)
- `submit_obs_T0.slurm` — observables + $F_Q$, $F_{\rm EP}$, $\eta$ on D1 (D2–D5)
- `submit_metts.slurm`  — $T>0$ METTS sampler (D6)
- `submit_obs_Tfin.slurm` — observables + thermal $F_Q$, $F_{\rm EP}$, $\eta$ on D6 (D7–D9)
- `submit_fits.slurm`   — final fits + figures (D10)

### 3.1 $T=0$ ground-state production (D1)

```bash
#!/bin/bash
#SBATCH --job-name=qfi_dmrg_gs
#SBATCH --account=def-YOURACCT
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-20             # 7 sizes × 3 phases
#SBATCH --output=logs/gs_%A_%a.out
#SBATCH --error=logs/gs_%A_%a.err

set -euo pipefail
module load StdEnv/2023 julia/1.10.4 hdf5/1.14 openblas/0.3.24

PHASES=(D C U)
SIZES=(16 24 32 48 64 96 128)
PIDX=$((SLURM_ARRAY_TASK_ID / 7))
SIDX=$((SLURM_ARRAY_TASK_ID % 7))
PHASE=${PHASES[$PIDX]}
N=${SIZES[$SIDX]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SCRATCH/QFI_CHAIN/dmrg
julia --project=$HOME/qfi_julia run_dmrg_T0.jl \
  --phase $PHASE --N $N \
  --chi-max $(if (( N <= 32 )); then echo 400; \
              elif (( N <= 64 )); then echo 800; \
              else echo 1200; fi) \
  --sweeps 30 --tol 1e-10 \
  --out states/T0/${PHASE}_N${N}.h5
```

### 3.2 $T=0$ observable assembly (D2–D5)

```bash
#!/bin/bash
#SBATCH --job-name=qfi_obs_T0
#SBATCH --account=def-YOURACCT
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-20
#SBATCH --dependency=afterok:<JOBID_OF_GS_ARRAY>
#SBATCH --output=logs/obs_T0_%A_%a.out

set -euo pipefail
module load StdEnv/2023 python/3.11 hdf5/1.14
source $SCRATCH/qfi_venv/bin/activate

PHASES=(D C U)
SIZES=(16 24 32 48 64 96 128)
PIDX=$((SLURM_ARRAY_TASK_ID / 7))
SIDX=$((SLURM_ARRAY_TASK_ID % 7))
PHASE=${PHASES[$PIDX]}
N=${SIZES[$SIDX]}

cd $SCRATCH/QFI_CHAIN/dmrg
python compute_observables.py \
  --state states/T0/${PHASE}_N${N}.h5 \
  --phase $PHASE --N $N --temperature 0 \
  --ks 2,4,6,8 \
  --out obs/T0/${PHASE}_N${N}.npz
python assemble_FQ_FEP.py \
  --obs obs/T0/${PHASE}_N${N}.npz \
  --out obs/T0/${PHASE}_N${N}_summary.json
```

### 3.3 $T>0$ METTS production (D6)

METTS is **embarrassingly parallel over samples**: each sample is
an independent imaginary-time TDVP run from a random product state.
We shard 200 samples × 6 temperatures × 4 sizes × 3 phases =
**14 400 short jobs** as a single SLURM array.

```bash
#!/bin/bash
#SBATCH --job-name=qfi_metts
#SBATCH --account=def-YOURACCT
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14399%256       # cap concurrency at 256
#SBATCH --output=logs/metts_%A_%a.out

set -euo pipefail
module load StdEnv/2023 julia/1.10.4 hdf5/1.14 openblas/0.3.24

PHASES=(D C U)
SIZES=(16 32 48 64)
TEMPS=(0.05 0.10 0.20 0.40 0.80 1.60)
M=200    # samples per (phase, N, T)

NPHASE=${#PHASES[@]}; NSIZE=${#SIZES[@]}; NTEMP=${#TEMPS[@]}
idx=$SLURM_ARRAY_TASK_ID
s=$((idx % M));                       idx=$((idx / M))
ti=$((idx % NTEMP));                  idx=$((idx / NTEMP))
ni=$((idx % NSIZE));                  idx=$((idx / NSIZE))
pi=$((idx % NPHASE))
PHASE=${PHASES[$pi]}; N=${SIZES[$ni]}; T=${TEMPS[$ti]}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $SCRATCH/QFI_CHAIN/dmrg
CHI=$(( N <= 48 ? 600 : 900 ))
julia --project=$HOME/qfi_julia run_metts_Tfin.jl \
  --phase $PHASE --N $N --T $T \
  --chi-max $CHI --dtau 0.025 \
  --burn-in 30 --sample-id $s \
  --out states/Tfin/${PHASE}_N${N}_T${T}_s${s}.h5
```

> Notes:
> - `--sample-id` seeds the random product start; samples can be
>   re-run independently if any time-out.
> - Each job is short ($\le 2$ h wall); if a node has many small
>   jobs queued, prefer `--array=...%256` to throttle.

### 3.4 $T>0$ observable assembly + ensemble averaging (D7–D9)

One job per `(phase,N,T)` triple, *after* all $M$ samples for that
triple are done. We use job-array indexing matched to $3\times 4\times 6=72$.

```bash
#!/bin/bash
#SBATCH --job-name=qfi_obs_Tfin
#SBATCH --account=def-YOURACCT
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-71
#SBATCH --dependency=afterok:<JOBID_OF_METTS_ARRAY>
#SBATCH --output=logs/obs_Tfin_%A_%a.out

set -euo pipefail
module load StdEnv/2023 python/3.11 hdf5/1.14
source $SCRATCH/qfi_venv/bin/activate

PHASES=(D C U)
SIZES=(16 32 48 64)
TEMPS=(0.05 0.10 0.20 0.40 0.80 1.60)
NPHASE=${#PHASES[@]}; NSIZE=${#SIZES[@]}; NTEMP=${#TEMPS[@]}
idx=$SLURM_ARRAY_TASK_ID
ti=$((idx % NTEMP)); idx=$((idx / NTEMP))
ni=$((idx % NSIZE)); idx=$((idx / NSIZE))
pi=$idx
PHASE=${PHASES[$pi]}; N=${SIZES[$ni]}; T=${TEMPS[$ti]}

cd $SCRATCH/QFI_CHAIN/dmrg
python compute_observables.py \
  --state "states/Tfin/${PHASE}_N${N}_T${T}_s*.h5" \
  --phase $PHASE --N $N --temperature $T \
  --ks 2,4,6,8 --jackknife \
  --out obs/Tfin/${PHASE}_N${N}_T${T}.npz
python assemble_FQ_FEP.py \
  --obs obs/Tfin/${PHASE}_N${N}_T${T}.npz \
  --thermal --jackknife \
  --out obs/Tfin/${PHASE}_N${N}_T${T}_summary.json
```

### 3.5 Aggregation + figures (D10)

Single job, runs after all of `obs_Tfin` finish:

```bash
#!/bin/bash
#SBATCH --job-name=qfi_fits
#SBATCH --account=def-YOURACCT
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --dependency=afterok:<JOBID_OF_OBS_TFIN_ARRAY>
#SBATCH --output=logs/fits_%j.out

module load StdEnv/2023 python/3.11
source $SCRATCH/qfi_venv/bin/activate
cd $SCRATCH/QFI_CHAIN/dmrg
python scaling_fits.py \
  --inputs-T0 'obs/T0/*_summary.json' \
  --inputs-Tfin 'obs/Tfin/*_summary.json' \
  --outdir ../figures \
  --csv-dir ../data
```

---

## 4. Submission sequence

```bash
cd $SCRATCH/QFI_CHAIN/dmrg

# 1. T=0 ground states
JID_GS=$(sbatch --parsable submit_gs.slurm)
echo "T=0 GS array:        $JID_GS"

# 2. T=0 observables (depends on GS)
sed -i "s/<JOBID_OF_GS_ARRAY>/$JID_GS/" submit_obs_T0.slurm
JID_OBS_T0=$(sbatch --parsable submit_obs_T0.slurm)
echo "T=0 obs array:       $JID_OBS_T0"

# 3. T>0 METTS production (independent of T=0; may launch in parallel)
JID_METTS=$(sbatch --parsable submit_metts.slurm)
echo "METTS array:         $JID_METTS"

# 4. T>0 observables (depends on METTS)
sed -i "s/<JOBID_OF_METTS_ARRAY>/$JID_METTS/" submit_obs_Tfin.slurm
JID_OBS_TFIN=$(sbatch --parsable submit_obs_Tfin.slurm)
echo "T>0 obs array:       $JID_OBS_TFIN"

# 5. Fits (depends on BOTH obs arrays)
sed -i "s/<JOBID_OF_OBS_TFIN_ARRAY>/$JID_OBS_TFIN/" submit_fits.slurm
JID_FITS=$(sbatch --parsable \
  --dependency=afterok:$JID_OBS_T0:$JID_OBS_TFIN submit_fits.slurm)
echo "Fits/figures:        $JID_FITS"

squeue -u $USER --format='%.10i %.9P %.30j %.8T %.10M %.6D %R'
```

---

## 5. Monitoring

```bash
# Live output for one task
tail -f $SCRATCH/QFI_CHAIN/dmrg/logs/gs_${JID_GS}_5.out

# Quick QFI snapshot from a finished state
python -c "
import h5py, numpy as np
with h5py.File('$SCRATCH/QFI_CHAIN/dmrg/states/U_N32.h5','r') as f:
    print('E0=', f['E0'][()], 'maxchi=', f['chi_max'][()])
"

# Aggregate residuals against gates G1–G4
python $SCRATCH/QFI_CHAIN/dmrg/check_gates.py --states states/
```

---

## 6. Pulling results back

```bash
# After fits job is done
rsync -av $SCRATCH/QFI_CHAIN/data/    $HOME/exact_diagonalization_clean/QFI_CHAIN/data/
rsync -av $SCRATCH/QFI_CHAIN/figures/ $HOME/exact_diagonalization_clean/QFI_CHAIN/figures/

# Recompile LaTeX with the new Sec 6.6 figures
cd $HOME/exact_diagonalization_clean/QFI_CHAIN/latex
pdflatex -interaction=nonstopmode qfi_geometric_extraction.tex
pdflatex -interaction=nonstopmode qfi_geometric_extraction.tex
```

---

## 7. Failure modes & remedies

| Symptom | Likely cause | Action |
|---|---|---|
| DMRG stuck above $E_0$ tolerance | Bond dimension cap too low | Resubmit failed `gs` array element with `--chi-max` ×1.5 |
| Truncation error $>10^{-7}$ | Same | Same |
| Gate G3 fails (DMRG≠ED at $N=16$) | Wrong $S^z$ sector or boundary | Force `--sztot 0` and re-run; verify Hamiltonian sign |
| Gate G4 fails ($\eta(D)\ne 1$) | Wiener regularization too strong | Reduce $\lambda$ to $10^{-12}\mathrm{tr}\Sigma/\dim$ |
| Gate G5 fails (METTS energy off) | TDVP $\Delta\tau$ too coarse | Halve `--dtau` to 0.0125 and double `--burn-in` |
| Gate G6 fails ($\delta F_Q/F_Q>5\%$) | Insufficient samples | Resubmit `submit_metts.slurm --array=14400-21599` to add 100 more samples / triple |
| Gate G7 fails (METTS$\to 0$ disagrees with $T=0$) | Bond dim too small at low $T$ | Bump METTS `CHI` to $1.5\times$ for the $T=0.05$ ladder |
| Gate G8 fails (FD vs linear-response slope mismatch) | $\theta=\pm 10^{-3}$ outside linear regime | Drop $\epsilon$ to $10^{-4}$ |
| Fits job fails | Missing `obs/T*/<phase>_N<N>...` | Re-run failed `obs_T0` or `obs_Tfin` element only |
| Out-of-memory at $N=128$ | $\chi=1200$ too aggressive on node | Drop to $\chi=900$ AND request `--mem=64G` |

---

## 8. Reproducibility log

After every successful production run, append to
`data/RUN_LOG.md` (one row per submission batch):

```
| Date | JobID(GS) | JobID(Obs) | JobID(Fits) | Phases | Sizes | chi_max | Notes |
|------|-----------|------------|-------------|--------|-------|---------|-------|
```

This row plus the `slurm-*.out` files in `dmrg/logs/` is the
ground-truth reproducibility trail; do not delete `logs/` until the
manuscript is submitted.

---

## 9. Estimated wall-clock to figures-in-paper

| Stage | Wall (h) |
|-------|---|
| GS array (longest task = $U,N=128$) | 32 |
| Obs $T=0$ array (longest = $U,N=128$) | 8 |
| METTS array (with `--array=...%256`, longest = $U,N=64,T=0.05$) | 8 |
| Obs $T>0$ array (longest = $U,N=64,T=0.05$) | 4 |
| Fits | <1 |
| **Total elapsed (with parallel submission of $T=0$ and $T>0$)** | **≈ 41 h** |

GS and METTS arrays are independent and should be launched in
parallel; the critical path is set by the slowest $T=0$ DMRG job.
Full cumulative core-hour budget ≈ 600 core-h (§5 of
[`PROJECT_PLAN.md`](PROJECT_PLAN.md)).
