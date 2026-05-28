#!/bin/bash
# qfi_submit_eta_path_T0.sh -- submit eta_{<=2}, eta_{<=4} post-processing
# for the corrected light T=0 path scan.

set -euo pipefail
cd /scratch/zhouzb79

mkdir -p qfi_chain/dmrg/logs qfi_chain/dmrg/obs/T0_path_eta

echo "=== QFI_CHAIN eta path T=0 submission $(date) ==="
jid=$(sbatch --parsable qfi_eta_path_T0.slurm)

cat <<EOF
Submitted qfi_eta_path_T0.slurm as $jid

Computes eta_{<=2}, eta_{<=4} for:
  path: D -> C -> U -> D
  sizes: N=16,32,64

Outputs:
  /scratch/zhouzb79/qfi_chain/dmrg/obs/T0_path_eta/<label>_N<N>_eta.json
EOF
