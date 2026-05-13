"""Collect simulation results and update the training set.

After GROMACS simulations complete:
1. Read lj_encapsulation.csv for ground truth values
2. Compare predictions vs simulation (for monitoring/logging)
3. Add simulated molecules to the training set and mark them processed
4. Deferred molecules are NOT recorded — they stay in the unprocessed pool
   for re-evaluation in future iterations (Option B: defer, not skip)

Usage:
    python active_learning/scripts/collect_results.py --iteration 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_learning.config import PipelineConfig
from active_learning.tracker import PipelineTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    config = PipelineConfig()
    tracker = PipelineTracker(config.resolve(config.tracker_file))

    iter_dir = config.resolve(config.iteration_output_dir) / f"iter_{args.iteration}"

    with open(iter_dir / "batch.json") as f:
        batch = json.load(f)

    with open(iter_dir / "predictions.json") as f:
        predictions = json.load(f)
    pred_map = {p["compound"]: p for p in predictions}

    simulate_list = []
    sim_path = iter_dir / "simulate_list.txt"
    if sim_path.exists():
        simulate_list = [
            l.strip() for l in sim_path.read_text().splitlines() if l.strip()
        ]

    deferred_list = []
    deferred_path = iter_dir / "deferred_list.txt"
    if deferred_path.exists():
        deferred_list = [
            l.strip() for l in deferred_path.read_text().splitlines() if l.strip()
        ]

    # ── Collect simulation ground truth ──────────────────────────────────────
    sim_results = {}
    if simulate_list:
        csv_path = Path(config.sim_working_dir) / config.sim_output_csv
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_sim = dict(zip(df["compound"], df["encapsulation_mean"]))
            sim_results = {c: all_sim[c] for c in simulate_list if c in all_sim}

            missing = set(simulate_list) - set(sim_results)
            if missing:
                print(f"  Warning: {len(missing)} simulated molecules missing "
                      f"from CSV: {sorted(missing)}")

            print(f"  Collected simulation results for {len(sim_results)} molecules")
        else:
            print(f"  Warning: simulation CSV not found at {csv_path}")

    # ── Record ONLY simulated molecules in tracker (processed=True) ──────────
    # Deferred molecules are intentionally NOT recorded so they remain
    # in the unprocessed pool for future iterations.
    for compound in simulate_list:
        if compound in sim_results:
            p = pred_map.get(compound, {})
            tracker.record_molecule(
                compound=compound,
                pred_mean=p.get("mean", 0.0),
                pred_std=p.get("std", 0.0),
                simulated=True,
                sim_value=sim_results[compound],
                added_to_training=True,
                iteration=args.iteration,
            )
    tracker.save()

    print(f"  Processed (simulated): {len(sim_results)}")
    print(f"  Deferred (back to pool): {len(deferred_list)}")

    # ── Build updated training CSV ───────────────────────────────────────────
    base_csv = config.resolve(config.base_training_csv)
    output_csv = config.resolve(config.updated_training_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(base_csv)

    additions = tracker.get_training_additions()
    sim_values = tracker.get_molecule_sim_values()

    new_rows = []
    for compound in additions:
        if compound in sim_values:
            new_rows.append({
                "compound": compound,
                "encapsulation_mean": sim_values[compound],
            })

    if new_rows:
        combined_df = pd.concat(
            [base_df, pd.DataFrame(new_rows)], ignore_index=True,
        )
        combined_df = combined_df.drop_duplicates(subset="compound", keep="last")
    else:
        combined_df = base_df

    combined_df.to_csv(output_csv, index=False)
    tracker.training_csv_path = str(output_csv)
    tracker.save()

    n_total = len(combined_df)
    print(f"  Training CSV: {len(base_df)} base + {len(new_rows)} new = {n_total} total")
    print(f"  Written to: {output_csv}")

    # ── Build combined data directory (symlinks) ─────────────────────────────
    combined_dir = config.resolve(config.combined_data_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)

    base_dir = config.resolve(config.base_data_dir)
    new_dir = config.resolve(config.new_molecules_dir)

    for d in base_dir.iterdir():
        if d.is_dir():
            target = combined_dir / d.name
            if not target.exists():
                os.symlink(d.resolve(), target)

    for compound in additions:
        src = new_dir / compound
        target = combined_dir / compound
        if src.is_dir() and not target.exists():
            os.symlink(src.resolve(), target)

    # ── Prediction vs simulation comparison (monitoring) ─────────────────────
    if sim_results:
        rows, errors = [], []
        for compound, sim_val in sim_results.items():
            p = pred_map.get(compound, {})
            pred_val = p.get("mean", 0.0)
            err = abs(pred_val - sim_val)
            errors.append(err)
            rows.append({
                "compound": compound,
                "predicted": pred_val,
                "simulated": sim_val,
                "abs_error": err,
                "pred_std": p.get("std", 0.0),
            })

        pd.DataFrame(rows).to_csv(iter_dir / "comparison.csv", index=False)
        mae = np.mean(errors)
        print(f"\n  Prediction vs simulation MAE: {mae:.4f}  "
              f"(max error: {np.max(errors):.4f})")

    # ── Save iteration summary ───────────────────────────────────────────────
    mean_unc = float(np.mean([p.get("std", 0) for p in predictions]))

    summary = {
        "iteration": args.iteration,
        "batch_size": len(batch),
        "simulated": len(sim_results),
        "deferred": len(deferred_list),
        "training_set_size": n_total,
        "mean_uncertainty": mean_unc,
        "sim_mae": float(np.mean(errors)) if sim_results else None,
    }
    with open(iter_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Record iteration in tracker ──────────────────────────────────────────
    tracker.record_iteration(
        iteration=args.iteration,
        batch=batch,
        simulated_count=len(sim_results),
        deferred_count=len(deferred_list),
        mean_uncertainty=mean_unc,
        ensemble_models=tracker.ensemble_models,
        training_set_size=n_total,
    )

    print(f"\nIteration {args.iteration} postprocessing complete.")
    print(f"  Simulated: {len(sim_results)}, Deferred: {len(deferred_list)}, "
          f"Training set: {n_total}")

    if not simulate_list:
        wrap_msg = (
            "======================================================================\n"
            "ACTIVE LEARNING — NO SIMULATIONS THIS ITERATION\n"
            "======================================================================\n"
            f"Iteration {args.iteration}: simulate_list.txt was empty (nothing sent "
            "to GROMACS). Typical causes: all ensemble uncertainties at or below "
            "`uncertainty_threshold`, or the above-threshold set was larger than "
            "`max_simulations` and only the top budget ran in earlier steps.\n\n"
            "The filter step archives any leftover molecule directories under "
            "out_beads/ into batches/batch_<iteration>/ when the simulate list is "
            "empty, so stale dirs are not re-run on later iterations.\n\n"
            "What to do next:\n"
            "  - If you still want MD on the pool, lower `uncertainty_threshold` "
            "in active_learning/pipeline_config.json (or raise max_simulations).\n"
            "  - If you consider the loop converged, stop submitting iterations.\n"
            "  - If you used submit_multi_iteration.sh, later SLURM jobs may still "
            "be queued; cancel with scancel if you do not want more predict/filter "
            "cycles.\n\n"
            "This message is also saved under this iteration directory as "
            "AL_WRAPUP_NO_SIMULATIONS.txt\n"
            "======================================================================\n"
        )
        print(wrap_msg)
        (iter_dir / "AL_WRAPUP_NO_SIMULATIONS.txt").write_text(wrap_msg)


if __name__ == "__main__":
    main()

