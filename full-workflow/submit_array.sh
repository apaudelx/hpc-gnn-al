#!/bin/bash
#SBATCH --job-name=small_molecules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -p checkpt
#SBATCH -A loni_zwitterion
#SBATCH --time=72:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

module purge
module load gromacs/2021.7/intel-2021.5.0-intel-mpi-2021.5.1

# ---- Packmol (conda install) ----
: "${HPC_CONDA_PREFIX:=${WORK_ROOT:-/work/$USER}/conda}"
: "${HPC_CONDA_ENV:=hpc-gnn-al}"
if [ -f "$HPC_CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
  source "$HPC_CONDA_PREFIX/etc/profile.d/conda.sh"
  conda activate "$HPC_CONDA_ENV"
else
  echo "Warning: conda.sh not found at $HPC_CONDA_PREFIX/etc/profile.d/conda.sh; using PATH fallback."
  export PATH="$HPC_CONDA_PREFIX/bin:$PATH"
fi

SCRIPT_DIR="${SIM_WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR" || exit 1

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with sbatch --array=0-(N-1)."
  exit 1
fi

shopt -s nullglob
BASE_DIR="$SCRIPT_DIR/out_beads"
mapfile -t MOL_DIRS < <(printf '%s\n' "$BASE_DIR"/*/ | sed 's:/$::')

if [ ${#MOL_DIRS[@]} -eq 0 ]; then
  echo "Error: No molecule directories found in out_beads/"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge ${#MOL_DIRS[@]} ]; then
  echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is out of range (0..$((${#MOL_DIRS[@]}-1)))."
  exit 1
fi

bash "$SCRIPT_DIR/workflow_v2.sh" "${MOL_DIRS[$SLURM_ARRAY_TASK_ID]}"
