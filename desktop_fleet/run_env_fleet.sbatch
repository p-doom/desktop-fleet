#!/usr/bin/env bash
#SBATCH --job-name=env_fleet
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
allocated_nodes="${ENV_FLEET_ALLOCATED_NODES:?ENV_FLEET_ALLOCATED_NODES is required}"
slurm_nodes="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-}}"
if [[ -z "$slurm_nodes" || "$slurm_nodes" != "$allocated_nodes" ]]; then
    echo "fleet allocation has ${slurm_nodes:-unknown} nodes; expected $allocated_nodes" >&2
    exit 2
fi

exec srun \
    --kill-on-bad-exit=1 \
    --nodes="$allocated_nodes" \
    --ntasks="$allocated_nodes" \
    --ntasks-per-node=1 \
    uv run --no-sync python -m desktop_fleet.supervise launch
