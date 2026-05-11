#!/bin/bash
#SBATCH --job-name=qfi_obs_Tfin
#SBATCH --account=def-YOURACCT
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-71              # 3 × 4 × 6
#SBATCH --output=../logs/obs_Tfin_%A_%a.out

set -euo pipefail
module load StdEnv/2023 python/3.11
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

cd "$SLURM_SUBMIT_DIR/.."
TAG="${PHASE}_N${N}_T${T}"

python aggregate_metts.py \
    --samples "obs/Tfin/raw/${TAG}_s*.json" \
    --out obs/Tfin/${TAG}.json

python assemble_FQ_FEP.py \
    --input obs/Tfin/${TAG}.json \
    --ks 2,4,6,8 \
    --out obs/Tfin/${TAG}_summary.json
