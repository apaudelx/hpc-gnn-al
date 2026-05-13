"""Filter molecules by ensemble prediction uncertainty (budget-capped).

Reads predictions.json, ranks molecules by uncertainty, and selects
the top `max_simulations` most uncertain molecules (above threshold)
for GROMACS simulation. All other batch molecules are deferred — they
return to the unprocessed pool for re-evaluation in future iterations.

Also archives any previous simulation directories to preserve data.

When nothing is selected for simulation, leftover mol_* dirs under out_beads/
are archived to batches/batch_<iteration>/ so simulate.slurm (which only runs
compounds listed in simulate_list.txt) cannot pick up stale directories.

Usage:
    python active_learning/scripts/filter_uncertainty.py --iteration 1
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_learning.config import PipelineConfig


def _archive_previous_batch(iteration: int, config: PipelineConfig):
    """Move molecule directories from out_beads/ into batches/batch_N/."""
    out_beads = Path(config.sim_working_dir) / "out_beads"
    if not out_beads.exists():
        return

    mol_dirs = [d for d in out_beads.iterdir() if d.is_dir()]
    if not mol_dirs:
        return

    archive_dir = Path(config.sim_working_dir) / "batches" / f"batch_{iteration}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for d in mol_dirs:
        shutil.move(str(d), str(archive_dir / d.name))
        moved += 1

    for fname in ("lj_encapsulation.csv", "combined_lj.dat",
                  "encapsulation_mean_vs_logP_plape_res.dat"):
        src = out_beads / fname
        if src.exists():
            shutil.copy2(str(src), str(archive_dir / fname))

    print(f"  Archived {moved} molecule dirs → {archive_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    config = PipelineConfig()

    iter_dir = config.resolve(config.iteration_output_dir) / f"iter_{args.iteration}"
    with open(iter_dir / "predictions.json") as f:
        predictions = json.load(f)

    threshold = config.uncertainty_threshold
    max_sims = config.max_simulations

    above_threshold = [e for e in predictions if e["std"] > threshold]
    below_threshold = [e for e in predictions if e["std"] <= threshold]

    above_threshold.sort(key=lambda e: e["std"], reverse=True)
    simulate = [e["compound"] for e in above_threshold[:max_sims]]
    deferred_above = [e["compound"] for e in above_threshold[max_sims:]]
    deferred_below = [e["compound"] for e in below_threshold]
    deferred = deferred_above + deferred_below

    print(f"Uncertainty filter (threshold={threshold}, budget={max_sims}):")
    print(f"  Above threshold:    {len(above_threshold)}")
    print(f"  → Simulate (top {max_sims}): {len(simulate)}")
    print(f"  → Deferred:         {len(deferred)}  "
          f"({len(deferred_above)} above thr + {len(deferred_below)} below thr)")
    if simulate:
        top = above_threshold[:len(simulate)]
        print(f"  Std range of selected: "
              f"{top[-1]['std']:.6f} – {top[0]['std']:.6f}")

    (iter_dir / "simulate_list.txt").write_text("\n".join(simulate))
    (iter_dir / "deferred_list.txt").write_text("\n".join(deferred))
    (iter_dir / "simulate_count.txt").write_text(str(len(simulate)))

    if not simulate:
        print("  No molecules need simulation — nothing to stage.")
        # Move any leftover mol_* dirs out of out_beads/ so simulate.slurm
        # (which keys off simulate_list.txt only) does not leave stale dirs
        # for the next iteration, and disk matches batches/ layout.
        out_beads = Path(config.sim_working_dir) / "out_beads"
        if out_beads.exists() and any(d.is_dir() for d in out_beads.iterdir()):
            _archive_previous_batch(args.iteration, config)
        return

    # Archive molecule dirs from previous iteration
    prev = args.iteration - 1
    if prev >= 1:
        _archive_previous_batch(prev, config)
    else:
        out_beads = Path(config.sim_working_dir) / "out_beads"
        if out_beads.exists() and any(d.is_dir() for d in out_beads.iterdir()):
            _archive_previous_batch(0, config)

    # Copy selected molecules into out_beads/ for GROMACS
    out_beads = Path(config.sim_working_dir) / "out_beads"
    out_beads.mkdir(parents=True, exist_ok=True)

    new_mol_dir = config.resolve(config.new_molecules_dir)
    copied = 0
    for compound in simulate:
        src = new_mol_dir / compound
        dst = out_beads / compound
        if src.is_dir():
            shutil.copytree(src, dst)
            copied += 1
        else:
            print(f"  Warning: molecule directory not found: {src}")

    print(f"  Staged {copied}/{len(simulate)} molecules in {out_beads}")


if __name__ == "__main__":
    main()

