#!/bin/bash
#SBATCH --job-name=al_bootstrap
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -A loni_zwitterion
#SBATCH --output=active_learning/logs/bootstrap_%j.out
#SBATCH --error=active_learning/logs/bootstrap_%j.err

set -euo pipefail

# ==========================================
# Initialize and Activate Conda Environment
# ==========================================
: "${HPC_CONDA_PREFIX:=${WORK_ROOT:-/work/$USER}/conda}"
: "${HPC_CONDA_ENV:=hpc-gnn-al}"
if [ -f "$HPC_CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    source "$HPC_CONDA_PREFIX/etc/profile.d/conda.sh"
    conda activate "$HPC_CONDA_ENV"
else
    echo "Warning: conda.sh not found at $HPC_CONDA_PREFIX/etc/profile.d/conda.sh; using current Python."
fi
# ==========================================

PROJECT_ROOT="${PROJECT_ROOT:?Error: PROJECT_ROOT not set}"
cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo "ACTIVE LEARNING — BOOTSTRAP"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Date:         $(date)"
echo "Python path:  $(which python)" # Logs which python executable is being used
echo ""

# Step 1: Initialize pipeline (discover molecules, create tracker)
echo "--- Step 1: Initializing pipeline ---"
python active_learning/scripts/init_pipeline.py
echo ""

# Step 2: Train initial ensemble from base 739 dataset
echo "--- Step 2: Training initial ensemble ---"
python active_learning/scripts/train_ensemble.py \
    --training-csv data/ee_values_739.csv \
    --data-dir data/ee_itp_739 \
    --iteration 0
echo ""

echo "============================================================"
echo "Bootstrap complete — $(date)"
echo "============================================================"
echo ""
echo "Next: run an iteration with"
echo "  bash active_learning/submit_iteration.sh"
