#!/bin/bash
#SBATCH --job-name=qfi_obs_T0
#SBATCH --account=def-YOURACCT
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-20
#SBATCH --output=../logs/obs_T0_%A_%a.out

set -euo pipefail
module load StdEnv/2023 python/3.11
source $SCRATCH/qfi_venv/bin/activate

PHASES=(D C U)
SIZES=(16 24 32 48 64 96 128)
PIDX=$((SLURM_ARRAY_TASK_ID / 7))
SIDX=$((SLURM_ARRAY_TASK_ID % 7))
PHASE=${PHASES[$PIDX]}
N=${SIZES[$SIDX]}

cd "$SLURM_SUBMIT_DIR/.."     # → dmrg/
python assemble_FQ_FEP.py \
    --input obs/T0/${PHASE}_N${N}.json \
    --ks 2,4,6,8 \
    --out obs/T0/${PHASE}_N${N}_summary.json
