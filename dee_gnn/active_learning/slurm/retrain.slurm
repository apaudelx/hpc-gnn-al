#!/bin/bash
#SBATCH --job-name=al_bootstrap
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH -A loni_zwitterion
#SBATCH --output=active_learning/logs/retrain_%j.out
#SBATCH --error=active_learning/logs/retrain_%j.err

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
ITERATION="${ITERATION:?Error: ITERATION not set}"

cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "=== Stage 5: Retrain Ensemble (Iteration $ITERATION) ==="
echo "Date: $(date)"

python active_learning/scripts/train_ensemble.py \
    --training-csv active_learning/updated_training.csv \
    --data-dir active_learning/combined_data \
    --iteration "$ITERATION" \
    --skip-if-unchanged

echo "=== Retraining complete — $(date) ==="

# ── Print pipeline summary ───────────────────────────────────────────────
python -c "
import json
tracker = json.load(open('active_learning/tracker.json'))
iters = tracker.get('iterations', [])
if iters:
    print()
    print('Pipeline progress:')
    print(f\"{'Iter':>4} {'Simulated':>10} {'Deferred':>9} {'Train Size':>11} {'Mean Unc':>9}\")
    print('-' * 47)
    for it in iters:
        print(f\"{it['iteration']:>4} {it['simulated_count']:>10} \"
              f\"{it.get('deferred_count', it.get('skipped_count', 0)):>9} \"
              f\"{it['training_set_size']:>11} \"
              f\"{it['mean_uncertainty']:>9.4f}\")
"

# ── Check if more molecules remain ──────────────────────────────────────
HAS_MORE=$(python -c "
import json
tracker = json.load(open('active_learning/tracker.json'))
processed = {m for m, i in tracker['molecules'].items() if i.get('processed')}
unprocessed = [m for m in tracker['batch_order'] if m not in processed]
print('yes' if unprocessed else 'no')
")

echo ""
if [ "$HAS_MORE" = "yes" ]; then
    NEXT=$((ITERATION + 1))
    echo "More molecules to process. Next iteration: $NEXT"
    echo "  Run: bash active_learning/submit_iteration.sh $NEXT"
else
    echo "All molecules processed. Pipeline complete."
fi
