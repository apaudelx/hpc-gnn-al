#!/usr/bin/env bash
#
# Reset active learning pipeline artifacts.
#
# Usage:
#   bash reset_pipeline.sh          # soft reset (keep ensemble + tracker)
#   bash reset_pipeline.sh --hard   # full reset (wipe everything, re-bootstrap needed)
#
# Soft reset: clears iteration outputs, logs, and simulation staging so you
#   can re-run an iteration. Keeps ensemble models, tracker, and training CSV.
#
# Hard reset: wipes ALL pipeline artifacts for a complete fresh start.
#   You'll need to run submit_bootstrap.sh again.
#

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SIM_DIR="${SIM_WORKING_DIR:-$(cd "$PROJECT_ROOT/.." && pwd)/full-workflow}"
HARD=false

if [ "${1:-}" = "--hard" ]; then
    HARD=true
fi

if [ "$HARD" = true ]; then
    echo "=== Active Learning Pipeline — HARD RESET ==="
else
    echo "=== Active Learning Pipeline — SOFT RESET ==="
fi
echo "Project root: $PROJECT_ROOT"
echo ""

# ── 1. Iteration outputs (both modes) ──────────────────────────────────────
echo "[1/6] Cleaning iteration outputs ..."
rm -rf "$PROJECT_ROOT/active_learning/iterations"
echo "  Done."

# ── 2. SLURM logs (both modes) ─────────────────────────────────────────────
echo "[2/6] Cleaning logs ..."
rm -rf "$PROJECT_ROOT/active_learning/logs"
rm -f "$PROJECT_ROOT"/small_molecules_*.out "$PROJECT_ROOT"/small_molecules_*.err
echo "  Done."

# ── 3. Simulation staging (both modes) ─────────────────────────────────────
echo "[3/6] Cleaning out_beads/ molecule directories ..."
OUT_BEADS="$SIM_DIR/out_beads"
if [ -d "$OUT_BEADS" ]; then
    mol_removed=0
    for item in "$OUT_BEADS"/*/; do
        [ -d "$item" ] || continue
        rm -rf "$item"
        mol_removed=$((mol_removed + 1))
    done
    rm -f "$OUT_BEADS/lj_encapsulation.csv"
    rm -f "$OUT_BEADS/combined_lj.dat"
    rm -f "$OUT_BEADS/encapsulation_mean_vs_logP_plape_res.dat"
    echo "  Removed $mol_removed molecule dir(s) and data files."
else
    echo "  $OUT_BEADS not found — skipping."
fi

echo "[4/6] Cleaning archived batches ..."
BATCHES_DIR="$SIM_DIR/batches"
if [ -d "$BATCHES_DIR" ]; then
    rm -rf "$BATCHES_DIR"
    echo "  Done."
else
    echo "  No batches/ directory — skipping."
fi

# ── 5. Hard-only: ensemble, tracker, training data ─────────────────────────
if [ "$HARD" = true ]; then
    echo "[5/6] Removing ensemble models, tracker, and training data ..."
    rm -f  "$PROJECT_ROOT/active_learning/tracker.json"
    rm -f  "$PROJECT_ROOT/active_learning/pipeline_config.json"
    rm -f  "$PROJECT_ROOT/active_learning/updated_training.csv"
    rm -rf "$PROJECT_ROOT/active_learning/combined_data"
    rm -rf "$PROJECT_ROOT/active_learning/ensemble"
    echo "  Done."
else
    echo "[5/6] Keeping ensemble, tracker, and training data (soft reset)."
fi

# ── 6. Summary ─────────────────────────────────────────────────────────────
echo "[6/6] Checking original data integrity ..."
for f in "data/ee_values_739.csv" "data/ee_itp_739" "pool_1"; do
    if [ -e "$PROJECT_ROOT/$f" ]; then
        echo "  ✓ $f exists"
    else
        echo "  ✗ $f NOT FOUND"
    fi
done

echo ""
echo "=== Reset complete. ==="
echo ""
if [ "$HARD" = true ]; then
    echo "Hard reset done. Next steps:"
    echo "  1. bash active_learning/submit_bootstrap.sh"
    echo "  2. bash active_learning/submit_iteration.sh"
else
    echo "Soft reset done. Ensemble and tracker preserved."
    echo "  Re-run: bash active_learning/submit_iteration.sh"
fi
