#!/bin/bash
#SBATCH --job-name=al_simulate
#SBATCH --partition=checkpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH -A loni_zwitterion
#SBATCH --requeue
#SBATCH --output=active_learning/logs/simulate_%A_%a.out
#SBATCH --error=active_learning/logs/simulate_%A_%a.err

# NOTE: No set -e — we handle errors manually to prevent one failed
# molecule from killing the entire array and blocking dependent jobs.

set -uo pipefail

module purge
module load gromacs/2021.7/intel-2021.5.0-intel-mpi-2021.5.1
: "${HPC_CONDA_PREFIX:=${WORK_ROOT:-/work/$USER}/conda}"
: "${HPC_CONDA_ENV:=hpc-gnn-al}"
if [ -f "$HPC_CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    source "$HPC_CONDA_PREFIX/etc/profile.d/conda.sh"
    conda activate "$HPC_CONDA_ENV"
else
    echo "Warning: conda.sh not found at $HPC_CONDA_PREFIX/etc/profile.d/conda.sh; using PATH fallback."
    export PATH="$HPC_CONDA_PREFIX/bin:$PATH"
fi

PROJECT_ROOT="${PROJECT_ROOT:?Error: PROJECT_ROOT not set}"
ITERATION="${ITERATION:-0}"
# Match PipelineConfig.sim_working_dir (override on the submit node if needed).
SIM_DIR="${SIM_WORKING_DIR:-$(cd "$PROJECT_ROOT/.." && pwd)/full-workflow}"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID not set."
    exit 1
fi

LIST_FILE="$PROJECT_ROOT/active_learning/iterations/iter_${ITERATION}/simulate_list.txt"
if [ ! -f "$LIST_FILE" ]; then
    echo "Error: missing simulate_list.txt at $LIST_FILE"
    exit 1
fi

# One compound id per line; must match filter_uncertainty.py output.
mapfile -t SIM_LIST < <(grep -v '^[[:space:]]*$' "$LIST_FILE" || true)

if [ "${#SIM_LIST[@]}" -eq 0 ]; then
    echo "======================================================================"
    echo "  ACTIVE LEARNING: simulate_list.txt is EMPTY for iteration ${ITERATION}."
    echo "  No GROMACS workflows will run (all batch molecules deferred / below threshold)."
    echo "  Exiting successfully so postprocess can record this iteration."
    echo "======================================================================"
    exit 0
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SIM_LIST[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: no assigned molecule (only ${#SIM_LIST[@]} in simulate_list). Exit 0."
    exit 0
fi

MOL_NAME="${SIM_LIST[$SLURM_ARRAY_TASK_ID]}"
BASE_DIR="$SIM_DIR/out_beads"
MOL_DIR="$BASE_DIR/$MOL_NAME"

cd "$SIM_DIR"

if [ ! -d "$MOL_DIR" ]; then
    echo "ERROR: expected molecule dir missing: $MOL_DIR (compound $MOL_NAME)"
    mkdir -p "$PROJECT_ROOT/active_learning/iterations/iter_${ITERATION}"
    echo "$MOL_NAME" >> "$PROJECT_ROOT/active_learning/iterations/iter_${ITERATION}/failed_simulations.txt"
    exit 0
fi

echo "=== Task $SLURM_ARRAY_TASK_ID: simulating $MOL_NAME ==="
echo "Date: $(date)"

bash "$SIM_DIR/workflow_v2.sh" "$MOL_DIR" || {
    echo "ERROR: simulation failed for $MOL_NAME"
    mkdir -p "$PROJECT_ROOT/active_learning/iterations/iter_${ITERATION}"
    echo "$MOL_NAME" >> "$PROJECT_ROOT/active_learning/iterations/iter_${ITERATION}/failed_simulations.txt"
    exit 0
}

echo "=== Simulation complete for $MOL_NAME — $(date) ==="
