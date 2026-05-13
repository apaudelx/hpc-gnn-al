#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Submit multiple active learning iterations as one long SLURM dependency chain.
#
# Usage:
#   bash active_learning/submit_multi_iteration.sh 5      # submit 5 iterations
#   bash active_learning/submit_multi_iteration.sh         # defaults to 5
#
# Each iteration's predict job depends on the previous iteration's retrain job,
# so all N iterations run back-to-back with no manual intervention.
#
# All results are saved per iteration (models, predictions, tracker), so you
# can inspect progress at any point even while later iterations are queued.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

NUM_ITERATIONS=${1:-5}

parse_jobid() {
    grep -oE '^[0-9]+$' | head -1
}

# Read config values
BATCH_SIZE=$(python -c "
import json
cfg = json.load(open('active_learning/pipeline_config.json'))
print(cfg.get('batch_size', 100))
")

MAX_SIMS=$(python -c "
import json
cfg = json.load(open('active_learning/pipeline_config.json'))
print(cfg.get('max_simulations', 20))
")

ARRAY_MAX=$((MAX_SIMS - 1))

START_ITER=$(python -c "
import json
t = json.load(open('active_learning/tracker.json'))
print(t['current_iteration'] + 1)
")

END_ITER=$((START_ITER + NUM_ITERATIONS - 1))

export SIM_WORKING_DIR=$(python -c "
import json
cfg = json.load(open('active_learning/pipeline_config.json'))
print(cfg.get('sim_working_dir', str(__import__('pathlib').Path.cwd().parent / 'full-workflow')))
")

mkdir -p active_learning/logs

echo "============================================================"
echo "ACTIVE LEARNING — MULTI-ITERATION SUBMISSION"
echo "============================================================"
echo "Project root:     $PROJECT_ROOT"
echo "Iterations:       $START_ITER through $END_ITER ($NUM_ITERATIONS total)"
echo "Batch size:       $BATCH_SIZE  (predict per iteration)"
echo "Max simulations:  $MAX_SIMS  (top uncertain → simulate)"
echo ""

PREV_RETRAIN_JOB=""

for ITER in $(seq $START_ITER $END_ITER); do
    export ITERATION=$ITER

    echo "────────────────────────────────────────────────────────────"
    echo "  ITERATION $ITER"
    echo "────────────────────────────────────────────────────────────"

    # Predict — depends on previous iteration's retrain (if any)
    if [ -n "$PREV_RETRAIN_JOB" ]; then
        PRED_JOB=$(sbatch --parsable \
            --dependency=afterok:$PREV_RETRAIN_JOB \
            --export=ALL \
            active_learning/slurm/predict.slurm 2>&1 | parse_jobid)
    else
        PRED_JOB=$(sbatch --parsable \
            --export=ALL \
            active_learning/slurm/predict.slurm 2>&1 | parse_jobid)
    fi
    if [ -z "$PRED_JOB" ]; then echo "Error: failed to submit predict job (iter $ITER)"; exit 1; fi
    echo "  1. Predict:      $PRED_JOB"

    # Filter
    FILTER_JOB=$(sbatch --parsable \
        --dependency=afterok:$PRED_JOB \
        --export=ALL \
        active_learning/slurm/filter.slurm 2>&1 | parse_jobid)
    if [ -z "$FILTER_JOB" ]; then echo "Error: failed to submit filter job (iter $ITER)"; exit 1; fi
    echo "  2. Filter:       $FILTER_JOB"

    # Simulate (array)
    SIM_JOB=$(sbatch --parsable \
        --dependency=afterok:$FILTER_JOB \
        --array=0-$ARRAY_MAX \
        --export=ALL \
        active_learning/slurm/simulate.slurm 2>&1 | parse_jobid)
    if [ -z "$SIM_JOB" ]; then echo "Error: failed to submit simulate job (iter $ITER)"; exit 1; fi
    echo "  3. Simulate:     $SIM_JOB (array 0-$ARRAY_MAX)"

    # Postprocess
    POST_JOB=$(sbatch --parsable \
        --dependency=afterok:$SIM_JOB \
        --export=ALL \
        active_learning/slurm/postprocess.slurm 2>&1 | parse_jobid)
    if [ -z "$POST_JOB" ]; then echo "Error: failed to submit postprocess job (iter $ITER)"; exit 1; fi
    echo "  4. Postprocess:  $POST_JOB"

    # Retrain
    TRAIN_JOB=$(sbatch --parsable \
        --dependency=afterok:$POST_JOB \
        --export=ALL \
        active_learning/slurm/retrain.slurm 2>&1 | parse_jobid)
    if [ -z "$TRAIN_JOB" ]; then echo "Error: failed to submit retrain job (iter $ITER)"; exit 1; fi
    echo "  5. Retrain:      $TRAIN_JOB"

    PREV_RETRAIN_JOB=$TRAIN_JOB
    echo ""
done

echo "============================================================"
echo "ALL $NUM_ITERATIONS ITERATIONS SUBMITTED"
echo "============================================================"
echo "Total SLURM jobs: $((NUM_ITERATIONS * 5)) + array tasks"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls active_learning/logs/"
echo ""
echo "Final retrain job: $PREV_RETRAIN_JOB"
echo "When everything finishes, check:"
echo "  cat active_learning/logs/retrain_${PREV_RETRAIN_JOB}.out"
