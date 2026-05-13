"""Train (or retrain) ensemble GNN models.

Trains N models with different random seeds on the same dataset.
Used for both the initial bootstrap and per-iteration retraining.

If --skip-if-unchanged is set and no new molecules were simulated
this iteration, the current ensemble is kept as-is.

Usage:
    # Bootstrap: train initial ensemble from base data
    python active_learning/scripts/train_ensemble.py \
        --training-csv data/ee_values_667.csv \
        --data-dir data/ee_itp_667 \
        --iteration 0

    # Retrain after iteration
    python active_learning/scripts/train_ensemble.py \
        --training-csv active_learning/updated_training.csv \
        --data-dir active_learning/combined_data \
        --iteration 1
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_learning.config import PipelineConfig
from active_learning.tracker import PipelineTracker


def _clean_env() -> dict:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _train_member(
    member_idx: int,
    seed: int,
    training_csv: Path,
    data_dir: Path,
    epochs: int,
    output_dir: Path,
    project_root: Path,
) -> str:
    """Train a single ensemble member. Returns the model results directory."""
    member_dir = output_dir / f"member_{member_idx}"
    member_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "train",
        "--train-data", str(training_csv),
        "--data-dir", str(data_dir),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--results-dir", str(member_dir),
    ]

    print(f"\n  Training member {member_idx} (seed={seed}) ...")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(project_root), env=_clean_env(),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Training member {member_idx} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    model_dir = _parse_model_dir(result.stdout)

    # Convert absolute path to relative for portability
    try:
        model_dir_rel = str(Path(model_dir).relative_to(project_root))
    except ValueError:
        model_dir_rel = model_dir

    print(f"  Member {member_idx} saved to: {model_dir_rel}")
    return model_dir_rel


def _parse_model_dir(stdout: str) -> str:
    for line in stdout.strip().split("\n"):
        if "All results saved in:" in line:
            return line.split("All results saved in:")[-1].strip()
    raise RuntimeError("Could not find model output directory in training output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument(
        "--skip-if-unchanged", action="store_true",
        help="Skip retraining if no new molecules were simulated",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    tracker = PipelineTracker(config.resolve(config.tracker_file))

    # Check if retraining is needed
    if args.skip_if_unchanged and args.iteration > 0:
        iter_dir = (
            config.resolve(config.iteration_output_dir) / f"iter_{args.iteration}"
        )
        count_file = iter_dir / "simulate_count.txt"
        sim_count = 0
        if count_file.exists():
            sim_count = int(count_file.read_text().strip())

        if sim_count == 0:
            print("No new molecules simulated this iteration. "
                  "Keeping current ensemble.")
            return

    training_csv = Path(args.training_csv)
    data_dir = Path(args.data_dir)
    epochs = config.training_epochs

    output_dir = (
        config.resolve(config.al_dir) / "ensemble" / f"iter_{args.iteration}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training ensemble ({config.ensemble_size} members, "
          f"{epochs} epochs each)")
    print(f"  Data:   {training_csv}")
    print(f"  Output: {output_dir}")

    model_dirs = []
    for idx in range(config.ensemble_size):
        seed = config.ensemble_seeds[idx]
        model_dir = _train_member(
            member_idx=idx,
            seed=seed,
            training_csv=training_csv,
            data_dir=data_dir,
            epochs=epochs,
            output_dir=output_dir,
            project_root=config.project_root,
        )
        model_dirs.append(model_dir)

    tracker.ensemble_models = model_dirs
    tracker.save()

    print(f"\nEnsemble training complete ({len(model_dirs)} models):")
    for i, d in enumerate(model_dirs):
        print(f"  Member {i}: {d}")


if __name__ == "__main__":
    main()
