#!/bin/bash
#SBATCH --job-name=qfi_fits
#SBATCH --account=def-YOURACCT
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=../logs/fits_%j.out

set -euo pipefail
module load StdEnv/2023 python/3.11
source $SCRATCH/qfi_venv/bin/activate

cd "$SLURM_SUBMIT_DIR/.."
python check_gates.py \
    --T0-glob 'obs/T0/*_summary.json' \
    --Tfin-glob 'obs/Tfin/*_summary.json'

python scaling_fits.py \
    --inputs-T0 'obs/T0/*_summary.json' \
    --inputs-Tfin 'obs/Tfin/*_summary.json' \
    --csv-dir ../data \
    --outdir ../figures
