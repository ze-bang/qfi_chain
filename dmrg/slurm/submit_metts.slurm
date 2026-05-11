#!/bin/bash
#SBATCH --job-name=qfi_metts
#SBATCH --account=def-YOURACCT
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-14399%256       # 3 phases × 4 sizes × 6 T × 200 samples
#SBATCH --output=../logs/metts_%A_%a.out

set -euo pipefail
module load StdEnv/2023 julia/1.10.4 hdf5/1.14 openblas/0.3.24

PHASES=(D C U)
SIZES=(16 32 48 64)
TEMPS=(0.05 0.10 0.20 0.40 0.80 1.60)
M=200

NPHASE=${#PHASES[@]}; NSIZE=${#SIZES[@]}; NTEMP=${#TEMPS[@]}
idx=$SLURM_ARRAY_TASK_ID
s=$((idx % M));      idx=$((idx / M))
ti=$((idx % NTEMP)); idx=$((idx / NTEMP))
ni=$((idx % NSIZE)); idx=$((idx / NSIZE))
pi=$((idx % NPHASE))
PHASE=${PHASES[$pi]}; N=${SIZES[$ni]}; T=${TEMPS[$ti]}

CHI=$(( N <= 48 ? 600 : 900 ))

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR/.."
mkdir -p obs/Tfin/raw logs
julia --project=. run_metts_Tfin.jl \
    --phase $PHASE --N $N --T $T \
    --chi-max $CHI --dtau 0.025 \
    --burn-in 30 --sample-id $s --r-max 7 \
    --out obs/Tfin/raw/${PHASE}_N${N}_T${T}_s${s}.json
