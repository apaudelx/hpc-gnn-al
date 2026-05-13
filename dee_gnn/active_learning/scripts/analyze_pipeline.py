"""Comprehensive analysis and visualization of the active learning pipeline.

Generates plots for:
  1. Ensemble diversity on training data (do the 5 models actually disagree?)
  2. Per-iteration uncertainty landscape (what's above/below threshold?)
  3. Cross-iteration tracking (how uncertainty shifts after retraining)
  4. Per-model prediction heatmap for high-uncertainty molecules
  5. Pipeline convergence curve

Usage:
    python active_learning/scripts/analyze_pipeline.py
    python active_learning/scripts/analyze_pipeline.py --output-dir analysis_plots
    python active_learning/scripts/analyze_pipeline.py --iteration 1   # single iteration
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from active_learning.config import PipelineConfig

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_tracker(config: PipelineConfig) -> dict:
    tracker_path = config.resolve(config.tracker_file)
    if not tracker_path.exists():
        print(f"Error: tracker not found at {tracker_path}")
        sys.exit(1)
    with open(tracker_path) as f:
        return json.load(f)


def load_iteration_predictions(config: PipelineConfig, iteration: int) -> list:
    path = config.resolve(config.iteration_output_dir) / f"iter_{iteration}" / "predictions.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_iteration_comparison(config: PipelineConfig, iteration: int) -> pd.DataFrame:
    path = config.resolve(config.iteration_output_dir) / f"iter_{iteration}" / "comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_ensemble_training_predictions(config: PipelineConfig, iteration: int) -> dict:
    """Load train_predictions.csv from each ensemble member for a given iteration.

    Returns dict: {member_idx: DataFrame with 'target' and 'predicted'}.
    """
    ensemble_dir = config.resolve(config.al_dir) / "ensemble" / f"iter_{iteration}"
    if not ensemble_dir.exists():
        return {}

    results = {}
    for member_dir in sorted(ensemble_dir.iterdir()):
        if not member_dir.is_dir() or not member_dir.name.startswith("member_"):
            continue
        idx = int(member_dir.name.split("_")[1])
        timestamp_dirs = sorted(member_dir.iterdir())
        if not timestamp_dirs:
            continue
        latest = timestamp_dirs[-1]
        csv_path = latest / "train_predictions.csv"
        if csv_path.exists():
            results[idx] = pd.read_csv(csv_path)
    return results


def get_available_iterations(config: PipelineConfig) -> list:
    iter_dir = config.resolve(config.iteration_output_dir)
    if not iter_dir.exists():
        return []
    iters = []
    for d in sorted(iter_dir.iterdir()):
        if d.is_dir() and d.name.startswith("iter_"):
            try:
                iters.append(int(d.name.split("_")[1]))
            except ValueError:
                pass
    return iters


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Ensemble Diversity on Training Data
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ensemble_training_diversity(config, output_dir, iteration=0):
    """Show how differently the 5 models predict on the same training molecules."""
    member_preds = load_ensemble_training_predictions(config, iteration)
    if not member_preds:
        print(f"  Skipping training diversity plot — no ensemble data for iter_{iteration}")
        return

    targets = member_preds[0]["target"].values
    pred_matrix = np.column_stack([member_preds[i]["predicted"].values for i in sorted(member_preds)])
    n_molecules, n_models = pred_matrix.shape

    per_mol_mean = pred_matrix.mean(axis=1)
    per_mol_std = pred_matrix.std(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Ensemble Diversity on Training Data (iter {iteration}, {n_molecules} molecules)",
                 fontsize=14, fontweight="bold")

    # (a) Histogram of per-molecule std
    ax = axes[0, 0]
    ax.hist(per_mol_std, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.axvline(config.uncertainty_threshold, color="red", ls="--", lw=1.5,
               label=f"Threshold ({config.uncertainty_threshold})")
    ax.set_xlabel("Per-Molecule Std (across 5 models)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Distribution of Ensemble Disagreement")
    ax.legend()
    above = (per_mol_std > config.uncertainty_threshold).sum()
    ax.text(0.97, 0.95, f"Above threshold: {above}/{n_molecules}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # (b) True vs Predicted Mean, colored by std
    ax = axes[0, 1]
    sc = ax.scatter(targets, per_mol_mean, c=per_mol_std, cmap="RdYlGn_r",
                    s=12, alpha=0.7, edgecolors="none")
    lims = [min(targets.min(), per_mol_mean.min()) - 0.05,
            max(targets.max(), per_mol_mean.max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Encapsulation")
    ax.set_ylabel("Ensemble Mean Prediction")
    ax.set_title("(b) True vs Predicted (colored by std)")
    plt.colorbar(sc, ax=ax, label="Std", shrink=0.8)

    # (c) Individual model predictions for top-10 most disagreed molecules
    ax = axes[1, 0]
    top_idx = np.argsort(per_mol_std)[-10:][::-1]
    x_labels = [f"Mol {i}" for i in top_idx]
    x_pos = np.arange(len(top_idx))
    for m in range(n_models):
        ax.scatter(x_pos + (m - 2) * 0.12, pred_matrix[top_idx, m],
                   s=30, label=f"Model {m}", zorder=3, alpha=0.8)
    ax.scatter(x_pos, targets[top_idx], marker="x", s=60, c="black",
               linewidths=1.5, label="True", zorder=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Prediction")
    ax.set_title("(c) Top-10 Most Disagreed Training Molecules")
    ax.legend(fontsize=7, ncol=3, loc="lower left")

    # (d) Model-to-model correlation
    ax = axes[1, 1]
    corr = np.corrcoef(pred_matrix.T)
    im = ax.imshow(corr, cmap="Blues", vmin=0.95, vmax=1.0)
    for i in range(n_models):
        for j in range(n_models):
            ax.text(j, i, f"{corr[i, j]:.4f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels([f"Model {i}" for i in range(n_models)])
    ax.set_yticklabels([f"Model {i}" for i in range(n_models)])
    ax.set_title("(d) Model Prediction Correlation")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = output_dir / f"ensemble_diversity_iter{iteration}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Per-Iteration Uncertainty Landscape
# ═══════════════════════════════════════════════════════════════════════════════

def plot_iteration_uncertainty(config, output_dir, iteration):
    """Scatter and distribution of uncertainty for a single iteration."""
    predictions = load_iteration_predictions(config, iteration)
    if not predictions:
        print(f"  Skipping iter {iteration} — no predictions found")
        return

    compounds = [p["compound"] for p in predictions]
    means = np.array([p["mean"] for p in predictions])
    stds = np.array([p["std"] for p in predictions])
    threshold = config.uncertainty_threshold

    above_mask = stds > threshold
    n_above = above_mask.sum()
    n_below = (~above_mask).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Iteration {iteration} — Uncertainty Landscape ({len(predictions)} molecules)",
                 fontsize=14, fontweight="bold")

    # (a) Mean vs Std scatter
    ax = axes[0]
    ax.scatter(means[~above_mask], stds[~above_mask], s=20, alpha=0.6,
               color="#4c72b0", label=f"Below threshold ({n_below})", zorder=2)
    ax.scatter(means[above_mask], stds[above_mask], s=30, alpha=0.8,
               color="#c44e52", label=f"Above threshold ({n_above})", zorder=3)
    ax.axhline(threshold, color="red", ls="--", lw=1, alpha=0.7)
    for i in np.where(above_mask)[0]:
        if stds[i] > threshold + 0.02:
            ax.annotate(compounds[i], (means[i], stds[i]), fontsize=6,
                        alpha=0.7, ha="center", va="bottom")
    ax.set_xlabel("Ensemble Mean Prediction")
    ax.set_ylabel("Ensemble Std")
    ax.set_title("(a) Prediction vs Uncertainty")
    ax.legend(fontsize=8)

    # (b) Ranked uncertainty bar chart (top 30)
    ax = axes[1]
    sorted_idx = np.argsort(stds)[::-1][:30]
    bar_colors = ["#c44e52" if stds[i] > threshold else "#4c72b0" for i in sorted_idx]
    bars = ax.barh(range(len(sorted_idx)), stds[sorted_idx], color=bar_colors, height=0.7)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([compounds[i] for i in sorted_idx], fontsize=7)
    ax.axvline(threshold, color="red", ls="--", lw=1, alpha=0.7, label=f"Threshold ({threshold})")
    ax.set_xlabel("Ensemble Std")
    ax.set_title(f"(b) Top 30 Most Uncertain Molecules")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # (c) Per-model predictions for top 5 most uncertain
    ax = axes[2]
    top5_idx = np.argsort(stds)[::-1][:5]
    x_pos = np.arange(5)
    has_per_model = "predictions" in predictions[0]

    if has_per_model:
        n_models = len(predictions[0]["predictions"])
        for m in range(n_models):
            model_preds = [predictions[i]["predictions"][m] for i in top5_idx]
            ax.scatter(x_pos + (m - 2) * 0.1, model_preds, s=40,
                       label=f"Model {m}", zorder=3, alpha=0.8)
        ax.errorbar(x_pos, means[top5_idx], yerr=stds[top5_idx],
                    fmt="none", ecolor="black", capsize=4, lw=1.5, zorder=2)
    else:
        ax.bar(x_pos, means[top5_idx], yerr=stds[top5_idx],
               capsize=5, color="#4c72b0", alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([compounds[i] for i in top5_idx], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Prediction")
    ax.set_title("(c) Top 5 — Individual Model Predictions")
    if has_per_model:
        ax.legend(fontsize=7, ncol=3)

    plt.tight_layout()
    path = output_dir / f"uncertainty_iter{iteration}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Prediction vs Simulation (for simulated molecules)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pred_vs_sim(config, output_dir, iteration):
    """Compare ensemble predictions with GROMACS simulation ground truth."""
    comparison = load_iteration_comparison(config, iteration)
    if comparison.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Iteration {iteration} — Prediction vs Simulation ({len(comparison)} molecules)",
                 fontsize=14, fontweight="bold")

    # (a) Scatter
    ax = axes[0]
    ax.scatter(comparison["simulated"], comparison["predicted"], s=50,
               c=comparison["pred_std"], cmap="RdYlGn_r", edgecolors="gray", lw=0.5, zorder=3)
    lims = [min(comparison["simulated"].min(), comparison["predicted"].min()) - 0.05,
            max(comparison["simulated"].max(), comparison["predicted"].max()) + 0.05]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    for _, row in comparison.iterrows():
        ax.annotate(row["compound"], (row["simulated"], row["predicted"]),
                    fontsize=6, alpha=0.7, ha="left", va="bottom")
    ax.set_xlabel("Simulated (Ground Truth)")
    ax.set_ylabel("Ensemble Prediction")
    ax.set_title("(a) Predicted vs Simulated")
    mae = comparison["abs_error"].mean()
    ax.text(0.03, 0.97, f"MAE = {mae:.4f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # (b) Error bar chart
    ax = axes[1]
    sorted_comp = comparison.sort_values("abs_error", ascending=False)
    colors = ["#c44e52" if e > 0.1 else "#4c72b0" for e in sorted_comp["abs_error"]]
    ax.barh(range(len(sorted_comp)), sorted_comp["abs_error"], color=colors, height=0.6)
    ax.set_yticks(range(len(sorted_comp)))
    ax.set_yticklabels(sorted_comp["compound"], fontsize=8)
    ax.set_xlabel("Absolute Error")
    ax.set_title("(b) Per-Molecule Prediction Error")
    ax.invert_yaxis()

    plt.tight_layout()
    path = output_dir / f"pred_vs_sim_iter{iteration}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Cross-Iteration Convergence
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence(config, output_dir, tracker):
    """Track pipeline progress across all iterations."""
    iterations = tracker.get("iterations", [])
    if len(iterations) < 1:
        print("  Skipping convergence plot — need at least 1 iteration")
        return

    iter_nums = [it["iteration"] for it in iterations]
    simulated = [it["simulated_count"] for it in iterations]
    deferred = [it.get("deferred_count", it.get("skipped_count", 0)) for it in iterations]
    train_sizes = [it["training_set_size"] for it in iterations]
    mean_uncs = [it["mean_uncertainty"] for it in iterations]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Active Learning Pipeline Convergence", fontsize=14, fontweight="bold")

    # (a) Simulated vs Deferred per iteration
    ax = axes[0, 0]
    w = 0.35
    x = np.arange(len(iter_nums))
    ax.bar(x - w / 2, simulated, w, label="Simulated", color="#c44e52")
    ax.bar(x + w / 2, deferred, w, label="Deferred", color="#4c72b0", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Iter {n}" for n in iter_nums])
    ax.set_ylabel("Molecule Count")
    ax.set_title("(a) Simulated vs Deferred per Iteration")
    ax.legend()
    for i, v in enumerate(simulated):
        ax.text(i - w / 2, v + 0.5, str(v), ha="center", fontsize=9)

    # (b) Mean uncertainty over iterations
    ax = axes[0, 1]
    ax.plot(iter_nums, mean_uncs, "o-", color="#c44e52", lw=2, markersize=8)
    ax.axhline(config.uncertainty_threshold, color="gray", ls="--", lw=1,
               label=f"Threshold ({config.uncertainty_threshold})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Ensemble Std")
    ax.set_title("(b) Mean Uncertainty Across Iterations")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # (c) Training set growth
    ax = axes[1, 0]
    ax.plot(iter_nums, train_sizes, "s-", color="#4c72b0", lw=2, markersize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Set Size")
    ax.set_title("(c) Training Set Growth")
    for i, v in enumerate(train_sizes):
        ax.text(iter_nums[i], v + 1, str(v), ha="center", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # (d) Cumulative simulations
    ax = axes[1, 1]
    cum_sim = np.cumsum(simulated)
    total_pool = len(tracker.get("molecule_pool", []))
    ax.plot(iter_nums, cum_sim, "D-", color="#dd8452", lw=2, markersize=8)
    if total_pool > 0:
        ax.axhline(total_pool, color="gray", ls=":", lw=1, label=f"Pool size ({total_pool})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Simulations")
    ax.set_title("(d) Cumulative Simulation Count")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = output_dir / "convergence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Cross-Iteration Uncertainty Shift (for deferred molecules)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_uncertainty_shift(config, output_dir, available_iters):
    """Show how molecule uncertainty changes across iterations (deferred molecules)."""
    if len(available_iters) < 2:
        print("  Skipping uncertainty shift — need at least 2 iterations")
        return

    all_preds = {}
    for it in available_iters:
        preds = load_iteration_predictions(config, it)
        for p in preds:
            all_preds.setdefault(p["compound"], {})[it] = p["std"]

    multi_iter = {c: stds for c, stds in all_preds.items() if len(stds) >= 2}
    if not multi_iter:
        print("  Skipping uncertainty shift — no molecules appear in multiple iterations")
        return

    max_std_range = {c: max(stds.values()) - min(stds.values()) for c, stds in multi_iter.items()}
    top_shifted = sorted(max_std_range, key=max_std_range.get, reverse=True)[:15]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Uncertainty Shift Across Iterations (Deferred Molecules)",
                 fontsize=14, fontweight="bold")

    for compound in top_shifted:
        stds = multi_iter[compound]
        iters = sorted(stds.keys())
        ax.plot(iters, [stds[i] for i in iters], "o-", label=compound, alpha=0.7, markersize=5)

    ax.axhline(config.uncertainty_threshold, color="red", ls="--", lw=1.5,
               alpha=0.7, label=f"Threshold ({config.uncertainty_threshold})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Ensemble Std")
    ax.set_title(f"Top 15 Molecules with Largest Uncertainty Shift")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = output_dir / "uncertainty_shift.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(tracker, config):
    """Print a clean text summary of all iterations."""
    iterations = tracker.get("iterations", [])
    total_pool = len(tracker.get("molecule_pool", []))
    processed = len([m for m, info in tracker.get("molecules", {}).items()
                     if info.get("processed")])

    print("\n" + "=" * 70)
    print("ACTIVE LEARNING PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Candidate pool:        {total_pool}")
    print(f"  Processed (simulated): {processed}")
    print(f"  Remaining:             {total_pool - processed}")
    print(f"  Threshold:             {config.uncertainty_threshold}")
    print(f"  Max sims/iteration:    {config.max_simulations}")
    print(f"  Ensemble size:         {config.ensemble_size}")
    print()

    if iterations:
        header = f"{'Iter':>4} {'Batch':>6} {'Simul':>6} {'Defer':>6} {'Train':>6} {'Mean Unc':>10}"
        print(header)
        print("-" * len(header))
        for it in iterations:
            deferred = it.get("deferred_count", it.get("skipped_count", 0))
            print(f"{it['iteration']:>4} {len(it['batch']):>6} {it['simulated_count']:>6} "
                  f"{deferred:>6} {it['training_set_size']:>6} "
                  f"{it['mean_uncertainty']:>10.6f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze active learning pipeline results")
    parser.add_argument("--output-dir", default="active_learning/analysis",
                        help="Directory to save plots (default: active_learning/analysis)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Analyze a single iteration (default: all)")
    parser.add_argument("--training-diversity-iter", type=int, default=None,
                        help="Which ensemble iteration to use for training diversity "
                             "(default: latest available)")
    args = parser.parse_args()

    config = PipelineConfig()
    tracker = load_tracker(config)

    output_dir = config.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary_table(tracker, config)

    available_iters = get_available_iterations(config)
    if not available_iters:
        print("No iteration data found. Run at least one iteration first.")
        return

    iters_to_analyze = [args.iteration] if args.iteration is not None else available_iters

    print(f"Output directory: {output_dir}")
    print(f"Iterations available: {available_iters}")
    print()

    # Ensemble diversity on training data
    div_iter = args.training_diversity_iter
    if div_iter is None:
        ensemble_dir = config.resolve(config.al_dir) / "ensemble"
        if ensemble_dir.exists():
            ens_iters = sorted([int(d.name.split("_")[1]) for d in ensemble_dir.iterdir()
                                if d.is_dir() and d.name.startswith("iter_")])
            div_iter = ens_iters[-1] if ens_iters else 0
        else:
            div_iter = 0

    print(f"── Ensemble Training Diversity (iter {div_iter}) ──")
    plot_ensemble_training_diversity(config, output_dir, div_iter)

    # Per-iteration plots
    for it in iters_to_analyze:
        print(f"\n── Iteration {it} ──")
        plot_iteration_uncertainty(config, output_dir, it)
        plot_pred_vs_sim(config, output_dir, it)

    # Cross-iteration analysis
    if len(available_iters) >= 2:
        print(f"\n── Cross-Iteration Analysis ──")
        plot_uncertainty_shift(config, output_dir, available_iters)

    print(f"\n── Convergence ──")
    plot_convergence(config, output_dir, tracker)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

