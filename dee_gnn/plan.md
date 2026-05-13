# Active Learning Pipeline — Design Document

## Overview

Uncertainty-based active learning pipeline for molecular encapsulation prediction.
A **deep ensemble** of 5 GNN models provides uncertainty estimates. Only molecules
where the ensemble **disagrees** (high prediction variance) are sent to expensive
GROMACS simulations. This avoids simulating molecules the model already understands,
saving significant HPC compute time.

All computation runs on **compute nodes** via SLURM job dependency chains.
The head node is only used to submit jobs.

---

## Architecture

```
Head Node (user runs once):
  bash submit_iteration.sh

Compute Nodes (SLURM dependency chain):
  Predict → Filter → Simulate (array) → Postprocess → Retrain
    |         |          |                  |             |
   CPU       CPU       CPU/MPI            CPU           CPU
```

### Stage-by-Stage Flow

| Stage | SLURM Job | Script | What It Does |
|-------|-----------|--------|--------------|
| 1 | `predict.slurm` | `predict_ensemble.py` | Load 5 ensemble models, predict all batch molecules, compute mean + std |
| 2 | `filter.slurm` | `filter_uncertainty.py` | Split by uncertainty threshold: high → simulate, low → skip; stage molecules in out_beads/ |
| 3 | `simulate.slurm` (array) | `workflow_v2.sh` | GROMACS MD simulation for each high-uncertainty molecule |
| 4 | `postprocess.slurm` | `collect_results.py` | Read lj_encapsulation.csv, compare with predictions, update training CSV |
| 5 | `retrain.slurm` | `train_ensemble.py` | Retrain all 5 ensemble members on updated dataset |

### Key Difference from Previous Pipeline

**Before (error-based):** Simulate ALL molecules → predict → compare → add wrong ones to training.
- Problem: 100% of molecules are simulated. No compute savings.

**Now (uncertainty-based):** Predict FIRST → only simulate uncertain ones → add results to training.
- Benefit: Confident predictions skip simulation entirely. Actual compute savings.

---

## File Structure

```
active_learning/
├── __init__.py
├── config.py                   # PipelineConfig dataclass
├── tracker.py                  # JSON-backed state tracker
│
├── scripts/                    # Python scripts (called by SLURM jobs)
│   ├── init_pipeline.py        # Discover molecules, create tracker
│   ├── predict_ensemble.py     # Ensemble inference with uncertainty
│   ├── filter_uncertainty.py   # Split high/low uncertainty, stage molecules
│   ├── collect_results.py      # Post-simulation: collect, compare, update CSV
│   └── train_ensemble.py       # Train/retrain all ensemble members
│
├── slurm/                      # SLURM job scripts (run on compute nodes)
│   ├── bootstrap.slurm         # One-time: init + train initial ensemble
│   ├── predict.slurm           # Stage 1
│   ├── filter.slurm            # Stage 2
│   ├── simulate.slurm          # Stage 3 (array job)
│   ├── postprocess.slurm       # Stage 4
│   └── retrain.slurm           # Stage 5
│
├── submit_bootstrap.sh         # Head-node script: submit bootstrap job
└── submit_iteration.sh         # Head-node script: submit one iteration chain
```

---

## How to Run

### First Time Setup

```bash
# 1. Reset any previous artifacts
bash reset_pipeline.sh

# 2. Submit bootstrap (trains initial 5-model ensemble from 667 base data)
bash active_learning/submit_bootstrap.sh

# 3. Wait for bootstrap to complete
squeue -u $USER
```

### Each Iteration

```bash
# Submit iteration (auto-detects iteration number from tracker)
bash active_learning/submit_iteration.sh

# Or specify explicitly
bash active_learning/submit_iteration.sh 1

# Monitor
squeue -u $USER
tail -f active_learning/logs/*.out

# When retrain completes, check the log for next steps
cat active_learning/logs/retrain_<JOB_ID>.out
```

### Multiple Iterations

Iterations are submitted manually one at a time. After each retrain
completes, the log will tell you if more molecules remain and what
command to run next. This gives you control to inspect results between
iterations.

---

## Configuration

All parameters in `active_learning/config.py` (saved to `pipeline_config.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ensemble_size` | 5 | Number of GNN models in the ensemble |
| `ensemble_seeds` | [121-125] | Random seeds for each member |
| `uncertainty_threshold` | 0.05 | Prediction std above which to simulate |
| `batch_size` | 20 | Molecules per iteration |
| `training_epochs` | 500 | Epochs per ensemble member per training |

### Tuning the Uncertainty Threshold

- **Lower threshold** (e.g., 0.02): More molecules simulated, better model, slower
- **Higher threshold** (e.g., 0.10): Fewer simulations, faster, but may miss edge cases
- The right value depends on your ensemble's typical disagreement level
- Check `predictions.json` after the first iteration to calibrate

---

## Ensemble Uncertainty

The deep ensemble approach trains 5 identical GNN architectures with different
random seeds. At inference:

- Each model produces a prediction for each molecule
- **Mean** = best estimate of encapsulation
- **Std** = uncertainty (ensemble disagreement)

When std is high, the models disagree → the molecule is in a region of chemical
space the model doesn't understand well → simulate it to get ground truth.

When std is low, all models agree → the model is confident → skip simulation.

---

## SLURM Dependency Chain

```
Predict (afterok) → Filter (afterok) → Simulate (afterok) → Postprocess (afterok) → Retrain
```

- `afterok`: each stage runs only if the previous succeeded
- Simulate is an array job; tasks beyond the molecule count exit cleanly
- If a simulation fails, it logs the molecule and exits 0 (doesn't block the chain)
- If no molecules need simulation (all low uncertainty), the chain still runs
  but retrain detects this and skips retraining

---

## State Management

`tracker.json` tracks:
- `molecule_pool`: all candidate molecules
- `batch_order`: shuffled processing order
- `molecules`: per-molecule records (predictions, uncertainty, simulation results)
- `iterations`: per-iteration summaries
- `ensemble_models`: current list of 5 model directories

Each script reads and updates the tracker. The tracker persists across
iterations and can be inspected to understand pipeline state.

---

## Data Flow

```
ee_itp_100_new/          → candidate molecules (input, never modified)
data/ee_values_667.csv   → base training data (input, never modified)
data/ee_itp_667/         → base ITP files (input, never modified)

active_learning/
  tracker.json           → pipeline state
  updated_training.csv   → base 667 + accumulated new molecules
  combined_data/         → symlinks to all training molecule directories
  ensemble/              → trained ensemble models per iteration
  iterations/            → per-iteration outputs (predictions, comparisons, etc.)

../full-workflow/
  out_beads/             → staging area for GROMACS (only current batch)
  batches/               → archived simulation results from past batches
```
