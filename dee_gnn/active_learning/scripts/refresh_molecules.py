"""Refresh the molecule pool by scanning for newly added molecules.

Re-scans ee_itp_100_new/ and appends any molecules not already in the
tracker's molecule_pool or batch_order. Preserves all existing state
(processed molecules, training data, ensemble models).

Run this before submit_iteration.sh whenever you've added new molecule
directories to ee_itp_100_new/.

Usage:
    python active_learning/scripts/refresh_molecules.py
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
    tracker_path = config.resolve(config.tracker_file)

    if not tracker_path.exists():
        print("Error: tracker.json not found. Run bootstrap first.")
        sys.exit(1)

    tracker = PipelineTracker(tracker_path)

    new_mol_dir = config.resolve(config.new_molecules_dir)
    if not new_mol_dir.exists():
        print(f"Error: {new_mol_dir} not found.")
        sys.exit(1)

    # Scan the directory for all molecules with .itp files
    on_disk = sorted(
        d.name
        for d in new_mol_dir.iterdir()
        if d.is_dir() and any(d.glob("*.itp"))
    )

    existing_pool = set(tracker.data["molecule_pool"])
    new_molecules = [m for m in on_disk if m not in existing_pool]

    if not new_molecules:
        print("No new molecules found. Nothing to add.")
        print(f"  Total in pool: {len(existing_pool)}")
        print(f"  On disk:       {len(on_disk)}")
        return

    # Shuffle the new molecules and append to batch order
    rng = random.Random(config.seed)
    shuffled_new = list(new_molecules)
    rng.shuffle(shuffled_new)

    tracker.data["molecule_pool"].extend(new_molecules)
    tracker.data["batch_order"].extend(shuffled_new)
    tracker.save()

    remaining = len(tracker.get_next_batch(len(tracker.data["batch_order"])))

    print(f"Added {len(new_molecules)} new molecules:")
    for m in new_molecules:
        print(f"  + {m}")
    print(f"\nPool: {len(existing_pool)} existing + {len(new_molecules)} new "
          f"= {len(tracker.data['molecule_pool'])} total")
    print(f"Unprocessed: {remaining}")
    print(f"\nRun: bash active_learning/submit_iteration.sh")


if __name__ == "__main__":
    main()

