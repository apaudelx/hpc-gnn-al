#!/bin/bash
#SBATCH --job-name=al_filter
#SBATCH --partition=checkpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH -A loni_zwitterion
#SBATCH --output=active_learning/logs/filter_%j.out
#SBATCH --error=active_learning/logs/filter_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?Error: PROJECT_ROOT not set}"
ITERATION="${ITERATION:?Error: ITERATION not set}"

cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "=== Stage 2: Uncertainty Filter (Iteration $ITERATION) ==="
echo "Date: $(date)"

python active_learning/scripts/filter_uncertainty.py --iteration "$ITERATION"

echo "=== Filter complete — $(date) ==="
