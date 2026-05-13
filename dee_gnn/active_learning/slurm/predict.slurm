#!/bin/bash
#SBATCH --job-name=al_predict
#SBATCH --partition=checkpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -A loni_zwitterion
#SBATCH --output=active_learning/logs/predict_%j.out
#SBATCH --error=active_learning/logs/predict_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Error: PROJECT_ROOT not set}"
ITERATION="${ITERATION:?Error: ITERATION not set}"

cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "=== Stage 1: Ensemble Prediction (Iteration $ITERATION) ==="
echo "Date: $(date)"

python active_learning/scripts/predict_ensemble.py --iteration "$ITERATION"

echo "=== Prediction complete — $(date) ==="
