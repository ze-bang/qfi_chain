#!/bin/bash
# qfi_path_T0.slurm -- light T=0 scan along D -> C -> U -> D.
#
# Correct period-4 bond-pattern model:
#   D -> C: (J1,J2,J3) = (1,t,0)
#   C -> U: (J1,J2,J3) = (1,1,t)
#   U -> D: (J1,J2,J3) = (1,t,t)
#
# Light grid:
#   t = 0, 0.25, 0.50, 0.75, 1.00
# with duplicate C/U endpoints removed; the final D is kept to close the loop.
# 13 path points x N in {16,32,64} = 39 array jobs.
#SBATCH --job-name=qfi_pathT0
#SBATCH --account=def-ybkim
#SBATCH --time=11:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-38
#SBATCH --output=/scratch/zhouzb79/qfi_chain/dmrg/logs/pathT0_%A_%a.out
#SBATCH --error=/scratch/zhouzb79/qfi_chain/dmrg/logs/pathT0_%A_%a.err

set -euo pipefail
source /scratch/zhouzb79/qfi_setup_env.sh

SIZES=(16 32 64)

# label|J1|J2|J3
POINTS=(
  "D_t000|1.0|0.00|0.00"
  "DC_t025|1.0|0.25|0.00"
  "DC_t050|1.0|0.50|0.00"
  "DC_t075|1.0|0.75|0.00"
  "C_t100|1.0|1.00|0.00"
  "CU_t025|1.0|1.00|0.25"
  "CU_t050|1.0|1.00|0.50"
  "CU_t075|1.0|1.00|0.75"
  "U_t100|1.0|1.00|1.00"
  "UD_t075|1.0|0.75|0.75"
  "UD_t050|1.0|0.50|0.50"
  "UD_t025|1.0|0.25|0.25"
  "D_close_t000|1.0|0.00|0.00"
)

NSIZE=${#SIZES[@]}
PIDX=$((SLURM_ARRAY_TASK_ID / NSIZE))
SIDX=$((SLURM_ARRAY_TASK_ID % NSIZE))
N=${SIZES[$SIDX]}

IFS='|' read -r LABEL J1 J2 J3 <<< "${POINTS[$PIDX]}"

if (( N <= 32 )); then
    CHI=500
else
    CHI=900
fi

cd "$QFI_CHAIN_DIR/dmrg"
mkdir -p obs/T0_path

echo "[$(date)] path_T0 task=$SLURM_ARRAY_TASK_ID label=$LABEL N=$N J=($J1,$J2,$J3) chi=$CHI"

julia --project="$QFI_JULIA_PROJECT" run_dmrg_T0.jl \
    --J1 "$J1" --J2 "$J2" --J3 "$J3" --label "$LABEL" --N "$N" \
    --chi-max "$CHI" --sweeps 30 --tol 1e-10 --r-max 7 \
    --out "obs/T0_path/${LABEL}_N${N}.json"

echo "[$(date)] path_T0 task=$SLURM_ARRAY_TASK_ID DONE"
