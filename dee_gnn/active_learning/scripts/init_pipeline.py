"""Initialize the active learning pipeline.

Discovers candidate molecules, shuffles them, creates the tracker.
Run once before the first iteration (called by bootstrap.slurm).

Usage:
    python active_learning/scripts/init_pipeline.py
"""

import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_learning.config import PipelineConfig
from active_learning.tracker import PipelineTracker


def main():
    config = PipelineConfig()

    new_mol_dir = config.resolve(config.new_molecules_dir)
    if not new_mol_dir.exists():
        print(f"Error: new molecules directory not found: {new_mol_dir}")
        sys.exit(1)

    molecule_pool = sorted(
        d.name
        for d in new_mol_dir.iterdir()
        if d.is_dir() and any(d.glob("*.itp"))
    )

    if not molecule_pool:
        print(f"Error: no molecule directories with .itp files in {new_mol_dir}")
        sys.exit(1)

    rng = random.Random(config.seed)
    batch_order = list(molecule_pool)
    rng.shuffle(batch_order)

    total_batches = (len(batch_order) + config.batch_size - 1) // config.batch_size

    tracker_path = config.resolve(config.tracker_file)
    tracker = PipelineTracker(tracker_path)
    tracker.initialize(
        molecule_pool=molecule_pool,
        batch_order=batch_order,
        base_csv=config.base_training_csv,
    )

    config.save()

    print("Pipeline initialized:")
    print(f"  Molecules:             {len(molecule_pool)}")
    print(f"  Batch size:            {config.batch_size}")
    print(f"  Total iterations:      {total_batches}")
    print(f"  Uncertainty threshold: {config.uncertainty_threshold}")
    print(f"  Ensemble size:         {config.ensemble_size}")
    print(f"  Tracker:               {tracker.path}")


if __name__ == "__main__":
    main()
