#!/bin/bash
# qfi_submit_path_T0.sh -- submit the light D -> C -> U -> D T=0 path scan.
# Run from /scratch/zhouzb79 after syncing the repo to scratch.

set -euo pipefail
cd /scratch/zhouzb79

echo "=== QFI_CHAIN light T=0 path submission $(date) ==="

mkdir -p qfi_chain/dmrg/logs \
         qfi_chain/dmrg/obs/T0_path

jid=$(sbatch --parsable qfi_path_T0.slurm)

cat <<EOF
Submitted qfi_path_T0.slurm as $jid

Path:
  D->C: (1,t,0),  t=0,0.25,0.50,0.75,1
  C->U: (1,1,t),  t=0.25,0.50,0.75,1
  U->D: (1,t,t),  t=0.75,0.50,0.25,0

Sizes:
  N=16,32,64

Outputs:
  /scratch/zhouzb79/qfi_chain/dmrg/obs/T0_path/<label>_N<N>.json
  /scratch/zhouzb79/qfi_chain/dmrg/obs/T0_path/<label>_N<N>_psi.h5

Monitor:
  squeue -u \$USER -o '%.12i %.9P %.20j %.8T %.10M %.10l %R'
EOF
