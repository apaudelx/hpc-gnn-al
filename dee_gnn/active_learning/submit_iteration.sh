#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Submit one active learning iteration as a SLURM dependency chain.
#
# Run from the dee_gnn project root on the head node:
#   bash active_learning/submit_iteration.sh          # auto-detect iteration
#   bash active_learning/submit_iteration.sh 3        # explicit iteration
#
# This submits 5 dependent SLURM jobs:
#   Predict → Filter → Simulate (array) → Post-process → Retrain
#
# Each job only starts after the previous one succeeds (afterok).
# All heavy computation runs on compute nodes — nothing on the head node.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Extract numeric job ID from sbatch output.
# SLURM lua plugins on some clusters print extra messages (SU accounting,
# core warnings) alongside the job ID. This filters to just the number.
parse_jobid() {
    grep -oE '^[0-9]+$' | head -1
}

# Auto-detect iteration number from tracker if not provided
if [ -n "${1:-}" ]; then
    ITERATION=$1
else
    ITERATION=$(python -c "
import json
t = json.load(open('active_learning/tracker.json'))
print(t['current_iteration'] + 1)
")
fi
export ITERATION

export SIM_WORKING_DIR=$(python -c "
import json
cfg = json.load(open('active_learning/pipeline_config.json'))
print(cfg.get('sim_working_dir', str(__import__('pathlib').Path.cwd().parent / 'full-workflow')))
")

# Read config values
BATCH_SIZE=$(python -c "
import json
cfg = json.load(open('active_learning/pipeline_config.json'))
print(cfg.get('batch_size', 20))
")

MAX_SIMS=$(python -c "
import json
cfg = json.load(open('active_learning/pipeline_config.json'))
print(cfg.get('max_simulations', 5))
")

ARRAY_MAX=$((MAX_SIMS - 1))

mkdir -p active_learning/logs

echo "============================================================"
echo "ACTIVE LEARNING — ITERATION $ITERATION"
echo "============================================================"
echo "Project root:     $PROJECT_ROOT"
echo "Batch size:       $BATCH_SIZE  (predict per iteration)"
echo "Max simulations:  $MAX_SIMS  (top uncertain → simulate)"
echo ""

# ── Stage 1: Ensemble Prediction ─────────────────────────────────────────────
PRED_JOB=$(sbatch --parsable \
    --export=ALL \
    active_learning/slurm/predict.slurm 2>&1 | parse_jobid)
if [ -z "$PRED_JOB" ]; then echo "Error: failed to submit predict job"; exit 1; fi
echo "1. Predict job:     $PRED_JOB"

# ── Stage 2: Uncertainty Filter + Stage Molecules ────────────────────────────
FILTER_JOB=$(sbatch --parsable \
    --dependency=afterok:$PRED_JOB \
    --export=ALL \
    active_learning/slurm/filter.slurm 2>&1 | parse_jobid)
if [ -z "$FILTER_JOB" ]; then echo "Error: failed to submit filter job"; exit 1; fi
echo "2. Filter job:      $FILTER_JOB"

# ── Stage 3: GROMACS Simulation (array, only uncertain molecules) ────────────
SIM_JOB=$(sbatch --parsable \
    --dependency=afterok:$FILTER_JOB \
    --array=0-$ARRAY_MAX \
    --export=ALL \
    active_learning/slurm/simulate.slurm 2>&1 | parse_jobid)
if [ -z "$SIM_JOB" ]; then echo "Error: failed to submit simulate job"; exit 1; fi
echo "3. Simulate job:    $SIM_JOB (array 0-$ARRAY_MAX, max $MAX_SIMS)"

# ── Stage 4: Collect Results + Update Training Data ──────────────────────────
POST_JOB=$(sbatch --parsable \
    --dependency=afterok:$SIM_JOB \
    --export=ALL \
    active_learning/slurm/postprocess.slurm 2>&1 | parse_jobid)
if [ -z "$POST_JOB" ]; then echo "Error: failed to submit postprocess job"; exit 1; fi
echo "4. Postprocess job: $POST_JOB"

# ── Stage 5: Retrain Ensemble ────────────────────────────────────────────────
TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:$POST_JOB \
    --export=ALL \
    active_learning/slurm/retrain.slurm 2>&1 | parse_jobid)
if [ -z "$TRAIN_JOB" ]; then echo "Error: failed to submit retrain job"; exit 1; fi
echo "5. Retrain job:     $TRAIN_JOB"

echo ""
echo "Dependency chain:"
echo "  Predict ($PRED_JOB)"
echo "    → Filter ($FILTER_JOB)"
echo "      → Simulate ($SIM_JOB)"
echo "        → Postprocess ($POST_JOB)"
echo "          → Retrain ($TRAIN_JOB)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls active_learning/logs/"
echo ""
echo "When complete, check retrain log for next steps:"
echo "  cat active_learning/logs/retrain_${TRAIN_JOB}.out"
