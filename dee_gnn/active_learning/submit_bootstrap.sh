#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# One-time bootstrap: initialize pipeline + train initial ensemble.
#
# Run from the dee_gnn project root on the head node:
#   bash active_learning/submit_bootstrap.sh
#
# This submits a single SLURM job that:
#   1. Discovers molecules in ee_itp_100_new/
#   2. Creates tracker.json and pipeline_config.json
#   3. Trains an ensemble of 5 GNN models from the base 739 dataset
#
# After it completes, run iterations with:
#   bash active_learning/submit_iteration.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Extract numeric job ID from sbatch output (filters SLURM plugin messages)
parse_jobid() {
    grep -oE '^[0-9]+$' | head -1
}

mkdir -p active_learning/logs

echo "============================================================"
echo "ACTIVE LEARNING — SUBMITTING BOOTSTRAP"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

JOB_ID=$(sbatch --parsable --export=ALL \
    active_learning/slurm/bootstrap.slurm 2>&1 | parse_jobid)
if [ -z "$JOB_ID" ]; then echo "Error: failed to submit bootstrap job"; exit 1; fi

echo "Bootstrap job submitted: $JOB_ID"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f active_learning/logs/bootstrap_${JOB_ID}.out"
echo ""
echo "After completion, start the first iteration:"
echo "  bash active_learning/submit_iteration.sh"

