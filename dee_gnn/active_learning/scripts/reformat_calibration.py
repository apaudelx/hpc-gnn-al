"""Reformat merged calibration CSV into the requested summary table.

Usage examples:

python active_learning/scripts/reformat_calibration.py \
  --input active_learning/n3n4/n3n4_all_interations_merged.csv \
  --output active_learning/n3n4/reformatted_n3n4.csv

The script produces columns:
  iterations, compound, prediction_before, prediction_after, ground_truth,
  error_before, error_after, average_error_before, average_error_after,
    uncertainty, uncertainty_std, uncertainty_sem, average_pred_std

Behavior notes:
- "prediction_before" is taken from the iteration immediately before first simulation
    (fallback to iteration 0, then earliest available).
- "prediction_after" is taken from the earliest row where simulated_this_iteration is True.
- "average_error_before/after" are mean absolute errors for iterations before/after
  the first simulated iteration. If a compound was never simulated, "after" fields
  will be NaN and "average_error_before" will be the mean across all iterations.
- "uncertainty" is the pred_std at the same iteration used for prediction_before.
- "average_pred_std" is the mean pred_std across all iterations for the compound.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def safe_first(series):
    for v in series:
        if pd.notna(v):
            return v
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Merged calibration CSV")
    p.add_argument("--output", required=True, help="Output CSV path")
    args = p.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp)

    # Ensure types
    if "ensemble_iteration" in df.columns:
        df["ensemble_iteration"] = pd.to_numeric(df["ensemble_iteration"], errors="coerce")
    if "predicted" in df.columns:
        df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    if "pred_std" in df.columns:
        df["pred_std"] = pd.to_numeric(df["pred_std"], errors="coerce")
    if "ground_truth" in df.columns:
        df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce")
    if "abs_error" in df.columns:
        df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")

    groups = df.groupby(df["compound"].astype(str), sort=True)
    rows = []

    for compound, g in groups:
        g = g.sort_values("ensemble_iteration") if "ensemble_iteration" in g.columns else g

        iterations = safe_first(g.get("simulated_iterations", pd.Series([pd.NA])))

        # determine first simulated iteration (if any)
        first_sim_iter = None
        if "simulated_this_iteration" in g.columns:
            sim_rows = g[g["simulated_this_iteration"].astype(bool)]
            if not sim_rows.empty:
                sim_rows = sim_rows.sort_values("ensemble_iteration")
                first_sim_iter = int(sim_rows.iloc[0].get("ensemble_iteration"))

        # prediction before: prefer the prediction from the iteration immediately
        # before the first simulated iteration. Fallback order:
        # 1) prediction at max(iteration < first_sim_iter)
        # 2) prediction at ensemble_iteration == 0
        # 3) earliest available prediction
        pred_before = None
        pred_before_iter = None
        if "ensemble_iteration" in g.columns and "predicted" in g.columns:
            if first_sim_iter is not None:
                prior = g[g["ensemble_iteration"] < first_sim_iter]
                if not prior.empty:
                    # pick the row with the largest ensemble_iteration < first_sim_iter
                    prior_row = prior.loc[prior["ensemble_iteration"].idxmax()]
                    if pd.notna(prior_row.get("predicted")):
                        pred_before = float(prior_row["predicted"])
                        pred_before_iter = int(prior_row["ensemble_iteration"])
            if pred_before is None:
                # try iteration 0
                sel0 = g[g["ensemble_iteration"] == 0]
                if not sel0.empty and pd.notna(sel0.iloc[0].get("predicted")):
                    pred_before = float(sel0.iloc[0]["predicted"])
                    pred_before_iter = 0
            if pred_before is None:
                # fallback to earliest available predicted
                avail = g[g["predicted"].notna()].sort_values("ensemble_iteration")
                if not avail.empty:
                    pred_before = float(avail.iloc[0]["predicted"])
                    pred_before_iter = int(avail.iloc[0]["ensemble_iteration"])

        # prediction after: earliest row where simulated_this_iteration is True
        pred_after = None
        if "simulated_this_iteration" in g.columns:
            sim_rows = g[g["simulated_this_iteration"].astype(bool)]
            if not sim_rows.empty:
                sim_rows = sim_rows.sort_values("ensemble_iteration")
                row = sim_rows.iloc[0]
                pred_after = float(row.get("predicted", math.nan)) if pd.notna(row.get("predicted")) else None

        # ground truth (take first non-null)
        gt = safe_first(g.get("ground_truth", pd.Series([pd.NA])))
        gt = float(gt) if gt is not None and pd.notna(gt) else None

        # errors
        err_before = float(abs(pred_before - gt)) if (pred_before is not None and gt is not None) else None
        err_after = float(abs(pred_after - gt)) if (pred_after is not None and gt is not None) else None

        # average errors before/after: use abs_error column if present, otherwise compute
        avg_err_before = None
        avg_err_after = None
        if "abs_error" in g.columns and g["abs_error"].notna().any():
            if first_sim_iter is None:
                # no simulation: average over all
                avg_err_before = float(g["abs_error"].mean())
            else:
                before_mask = g["ensemble_iteration"] < first_sim_iter
                after_mask = g["ensemble_iteration"] >= first_sim_iter
                if before_mask.any():
                    avg_err_before = float(g.loc[before_mask, "abs_error"].mean())
                if after_mask.any():
                    avg_err_after = float(g.loc[after_mask, "abs_error"].mean())
        else:
            # compute from predicted & ground_truth if available
            if gt is not None and "predicted" in g.columns:
                if first_sim_iter is None:
                    avg_err_before = float(g["predicted"].sub(gt).abs().mean())
                else:
                    before = g[g["ensemble_iteration"] < first_sim_iter]
                    after = g[g["ensemble_iteration"] >= first_sim_iter]
                    if not before.empty:
                        avg_err_before = float(before["predicted"].sub(gt).abs().mean())
                    if not after.empty:
                        avg_err_after = float(after["predicted"].sub(gt).abs().mean())

        # uncertainty at pre-refinement iteration and average pred_std
        uncertainty = None
        uncertainty_std = None
        uncertainty_sem = None
        avg_pred_std = None
        if "pred_std" in g.columns:
            if "ensemble_iteration" in g.columns and pred_before_iter is not None:
                sel_before = g[g["ensemble_iteration"] == pred_before_iter]
                if not sel_before.empty and pd.notna(sel_before.iloc[0].get("pred_std")):
                    uncertainty = float(sel_before.iloc[0]["pred_std"])
            if uncertainty is None and "ensemble_iteration" in g.columns:
                sel0 = g[g["ensemble_iteration"] == 0]
                if not sel0.empty and pd.notna(sel0.iloc[0].get("pred_std")):
                    uncertainty = float(sel0.iloc[0]["pred_std"])
            if uncertainty is not None:
                uncertainty_std = float(uncertainty)
                # Ensemble size is 5 in this pipeline.
                uncertainty_sem = float(uncertainty_std / math.sqrt(5.0))
            if g["pred_std"].notna().any():
                avg_pred_std = float(g["pred_std"].mean())

        rows.append({
            "iterations": iterations if iterations is not None else "",
            "compound": compound,
            "prediction_before_refinement": pred_before,
            "prediction_after_refinement": pred_after,
            "ground_truth": gt,
            "error_before": err_before,
            "error_after": err_after,
            "average_error_before": avg_err_before,
            "average_error_after": avg_err_after,
            "uncertainty": uncertainty,
            "uncertainty_std": uncertainty_std,
            "uncertainty_sem": uncertainty_sem,
            "average_pred_std": avg_pred_std,
        })

    out_df = pd.DataFrame(rows)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outp, index=False)
    print(f"Wrote {outp}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()

