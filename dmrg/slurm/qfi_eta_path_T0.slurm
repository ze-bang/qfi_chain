#!/bin/bash
# qfi_eta_path_T0.slurm -- matched-filter eta_{<=2}, eta_{<=4}
# for the corrected light T=0 path scan. One path/N checkpoint per array task.
#SBATCH --job-name=qfi_etaPath
#SBATCH --account=def-ybkim
#SBATCH --time=11:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-38
#SBATCH --output=/scratch/zhouzb79/qfi_chain/dmrg/logs/etaPath_%A_%a.out
#SBATCH --error=/scratch/zhouzb79/qfi_chain/dmrg/logs/etaPath_%A_%a.err

set -euo pipefail
source /scratch/zhouzb79/qfi_setup_env.sh

SIZES=(16 32 64)
POINTS=(
  "D_t000"
  "DC_t025"
  "DC_t050"
  "DC_t075"
  "C_t100"
  "CU_t025"
  "CU_t050"
  "CU_t075"
  "U_t100"
  "UD_t075"
  "UD_t050"
  "UD_t025"
  "D_close_t000"
)

NSIZE=${#SIZES[@]}
PIDX=$((SLURM_ARRAY_TASK_ID / NSIZE))
SIDX=$((SLURM_ARRAY_TASK_ID % NSIZE))
LABEL=${POINTS[$PIDX]}
N=${SIZES[$SIDX]}

cd "$QFI_CHAIN_DIR/dmrg"
mkdir -p obs/T0_path_eta

IN="obs/T0_path/${LABEL}_N${N}.json"
OUT="obs/T0_path_eta/${LABEL}_N${N}_eta.json"

echo "[$(date)] eta_path task=$SLURM_ARRAY_TASK_ID label=$LABEL N=$N"
julia --project="$QFI_JULIA_PROJECT" compute_eta_path_T0.jl \
    --input-file "$IN" \
    --out "$OUT" \
    --kmax 4 \
    --eig-rtol 1e-10
echo "[$(date)] eta_path task=$SLURM_ARRAY_TASK_ID DONE"
