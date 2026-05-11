#!/bin/bash
#SBATCH --job-name=qfi_gs
#SBATCH --account=def-YOURACCT
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-20            # 3 phases × 7 sizes
#SBATCH --output=../logs/gs_%A_%a.out
#SBATCH --error=../logs/gs_%A_%a.err

set -euo pipefail
module load StdEnv/2023 julia/1.10.4 hdf5/1.14 openblas/0.3.24

PHASES=(D C U)
SIZES=(16 24 32 48 64 96 128)
PIDX=$((SLURM_ARRAY_TASK_ID / 7))
SIDX=$((SLURM_ARRAY_TASK_ID % 7))
PHASE=${PHASES[$PIDX]}
N=${SIZES[$SIDX]}

if   (( N <= 32 )); then CHI=400
elif (( N <= 64 )); then CHI=800
else                     CHI=1200
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR/.."     # → dmrg/
mkdir -p obs/T0 logs
julia --project=. run_dmrg_T0.jl \
    --phase $PHASE --N $N \
    --chi-max $CHI --sweeps 30 --tol 1e-10 --r-max 7 \
    --out obs/T0/${PHASE}_N${N}.json
