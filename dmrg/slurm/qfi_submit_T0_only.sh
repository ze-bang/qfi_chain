#!/bin/bash
# qfi_submit_T0_only.sh -- submit only the corrected T=0 DMRG campaign.
# Run from /scratch/zhouzb79 after syncing the repository to scratch.
#
# This intentionally does not submit METTS or the old Python F_EP assembly.
# The DMRG jobs write raw observable JSON plus *_psi.h5 checkpoints; eta_k
# post-processing should run after the corrected matched-filter basis is in
# place.

set -euo pipefail
cd /scratch/zhouzb79

echo "=== QFI_CHAIN corrected T=0 submission $(date) ==="

mkdir -p qfi_chain/dmrg/logs \
         qfi_chain/dmrg/obs/T0 \
         qfi_chain/states/T0 \
         qfi_chain/data \
         qfi_chain/figures

submit() {
    local desc="$1"; shift
    local jid
    jid=$(sbatch --parsable "$@")
    printf '  %-24s jid=%s   (%s)\n' "$desc" "$jid" "$*" >&2
    echo "$jid"
}

JID_GS_S=$(submit "gs_small corrected" qfi_gs_small.slurm)
JID_GS_L=$(submit "gs_large corrected" qfi_gs_large.slurm)

cat <<EOF

=== QFI_CHAIN corrected T=0 summary ===
  GS small array : $JID_GS_S   (D,C,U x N=16,24,32,48,64)
  GS large array : $JID_GS_L   (D,C,U x N=96,128)

Hamiltonian:
  period-4 nearest-neighbor pattern J1,J2,J1,J3,...
  D=(1,0,0), C=(1,1,0), U=(1,1,1), OBC

Monitor:
  squeue -u \$USER -o '%.12i %.9P %.20j %.8T %.10M %.10l %R'

Outputs:
  /scratch/zhouzb79/qfi_chain/dmrg/obs/T0/<phase>_N<N>.json
  /scratch/zhouzb79/qfi_chain/dmrg/obs/T0/<phase>_N<N>_psi.h5
EOF
