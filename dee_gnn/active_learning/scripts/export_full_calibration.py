"""Run the saved ensemble on every labeled compound and join with ground truth.

Uses existing trained models (no retraining). If you already have
``train_predictions.csv`` per member **with a ``compound`` column** (see ``src/train.py``),
prefer ``merge_ensemble_calibration_from_csvs.py`` — it only merges CSVs and does not
load PyTorch.

Writes one CSV:

  compound, ground_truth, predicted, pred_std, abs_error

Typical use — ensemble from iteration 2:

  python active_learning/scripts/export_full_calibration.py \\
      --ensemble-iteration 2

Or use paths currently stored in tracker.json (whatever iteration you last trained):

  python active_learning/scripts/export_full_calibration.py --use-tracker-ensemble

Compare every saved ensemble AL iteration (separate CSV + summary MAE per iter):

  python active_learning/scripts/export_full_calibration.py --all-ensemble-iterations

Each ``iter_N`` uses that folder's members only; ``pred_std`` / ``abs_error`` are
within that iteration's ensemble (not across iterations).

Ground truth comes from --training-csv (default: active_learning/updated_training.csv).
Graphs are built from --data-dir (default: active_learning/combined_data).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import joblib
from torch_geometric.loader import DataLoader

from parse_itp import parse_nbfix_table
from build_graphs import MolecularGraphBuilder
from gnn_model import EncapsulationGNN
from active_learning.config import PipelineConfig
from active_learning.tracker import PipelineTracker


def _load_model(model_dir: Path, device: torch.device) -> EncapsulationGNN:
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)["config"]
    model = EncapsulationGNN(
        node_dim=cfg["node_dim"],
        edge_dim=cfg["edge_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_bead_types=cfg["num_bead_types"],
        embedding_dim=cfg["embedding_dim"],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pth", map_location=device))
    model.eval()
    return model


def _predict(model, graphs, device):
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    preds, ids = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                batch.bead_type_id, batch.num_atoms, batch.num_bonds,
                batch.avg_degree, batch.max_degree, batch.graph_density,
                batch.total_charge, batch.charge_std, batch.unique_bead_types,
            )
            preds.extend(out.cpu().numpy().flatten().tolist())
            ids.extend(batch.compound_id)
    return dict(zip(ids, preds))


def _apply_scalers(graphs, model_dir: Path):
    node_scaler_path = model_dir / "node_scaler.pkl"
    edge_scaler_path = model_dir / "edge_scaler.pkl"
    graph_scaler_path = model_dir / "graph_scaler.pkl"

    if node_scaler_path.exists():
        scaler = joblib.load(node_scaler_path)
        for g in graphs:
            g.x = torch.tensor(scaler.transform(g.x.numpy()), dtype=torch.float32)
    if edge_scaler_path.exists():
        scaler = joblib.load(edge_scaler_path)
        for g in graphs:
            if g.edge_attr.shape[0] > 0:
                g.edge_attr = torch.tensor(
                    scaler.transform(g.edge_attr.numpy()), dtype=torch.float32
                )
    if graph_scaler_path.exists():
        scaler = joblib.load(graph_scaler_path)
        feats = np.array([[
            g.num_atoms.item(), g.num_bonds.item(), g.avg_degree.item(),
            g.max_degree.item(), g.graph_density.item(), g.total_charge.item(),
            g.charge_std.item(), g.unique_bead_types.item(),
        ] for g in graphs], dtype=np.float32)
        normed = scaler.transform(feats)
        for g, row in zip(graphs, normed):
            g.num_atoms = torch.tensor([row[0]], dtype=torch.float32)
            g.num_bonds = torch.tensor([row[1]], dtype=torch.float32)
            g.avg_degree = torch.tensor([row[2]], dtype=torch.float32)
            g.max_degree = torch.tensor([row[3]], dtype=torch.float32)
            g.graph_density = torch.tensor([row[4]], dtype=torch.float32)
            g.total_charge = torch.tensor([row[5]], dtype=torch.float32)
            g.charge_std = torch.tensor([row[6]], dtype=torch.float32)
            g.unique_bead_types = torch.tensor([row[7]], dtype=torch.float32)


def discover_member_run_dirs(ensemble_iter_dir: Path) -> list[Path]:
    """Latest timestamp subdirectory under each member_* containing model.pth."""
    member_roots = sorted(
        ensemble_iter_dir.glob("member_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    out: list[Path] = []
    for mdir in member_roots:
        candidates = sorted(
            d for d in mdir.iterdir()
            if d.is_dir() and (d / "model.pth").exists()
        )
        if not candidates:
            print(f"  Warning: no model.pth under {mdir}", file=sys.stderr)
            continue
        out.append(candidates[-1])
    return out


def resolve_ensemble_dirs(
    config: PipelineConfig,
    ensemble_iteration: int | None,
    use_tracker: bool,
) -> list[Path]:
    if use_tracker:
        tracker = PipelineTracker(config.resolve(config.tracker_file))
        paths = tracker.ensemble_models
        if not paths:
            sys.exit("tracker.json has empty ensemble_models — train an ensemble first.")
        resolved = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = config.project_root / pp
            if not (pp / "model.pth").exists():
                sys.exit(f"Missing model.pth at {pp}")
            resolved.append(pp)
        return resolved

    if ensemble_iteration is None:
        sys.exit("Provide --ensemble-iteration N or --use-tracker-ensemble")

    root = config.resolve(config.al_dir) / "ensemble" / f"iter_{ensemble_iteration}"
    if not root.is_dir():
        sys.exit(f"Ensemble directory not found: {root}")

    dirs = discover_member_run_dirs(root)
    if not dirs:
        sys.exit(f"No member models found under {root}")
    print(f"Using {len(dirs)} ensemble members from {root}")
    for i, d in enumerate(dirs):
        print(f"  member {i}: {d}")
    return dirs


def list_ensemble_iteration_dirs(al_ensemble_root: Path) -> list[tuple[int, Path]]:
    """Sorted (iteration_index, iter_dir) for ensemble/iter_*."""
    pairs: list[tuple[int, Path]] = []
    if not al_ensemble_root.is_dir():
        return pairs
    for d in al_ensemble_root.iterdir():
        if not d.is_dir() or not d.name.startswith("iter_"):
            continue
        try:
            n = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        pairs.append((n, d))
    pairs.sort(key=lambda x: x[0])
    return pairs


def run_single_ensemble_calibration(
    config: PipelineConfig,
    member_dirs: list[Path],
    training_csv: Path,
    data_dir: Path,
    nbfix: Path,
    device: torch.device,
    label: str = "",
    ensemble_iteration_col: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Build graphs with member_0's vocab/scalers, run all members, return (df, mae)."""
    ref_dir = member_dirs[0]
    prefix = f"[{label}] " if label else ""

    df = pd.read_csv(training_csv).dropna(subset=["encapsulation_mean"])
    print(f"{prefix}Loaded {len(df)} compounds with ground truth from {training_csv}")

    bead_type_to_id = None
    bead_map = ref_dir / "bead_type_to_id.json"
    if bead_map.exists():
        with open(bead_map) as f:
            bead_type_to_id = json.load(f)
        print(f"{prefix}Loaded bead_type_to_id mapping with {len(bead_type_to_id)} bead types")

    nbfix_map = parse_nbfix_table(str(nbfix))
    builder = MolecularGraphBuilder(
        nbfix_map,
        data_dir=str(data_dir),
        bead_type_to_id=bead_type_to_id,
    )

    graphs = builder.build_dataset(df)
    if not graphs:
        sys.exit(f"{prefix}No graphs built — check data-dir and CSV compound names.")

    valid_ids = [g.compound_id for g in graphs]
    gt_map = dict(zip(df["compound"], df["encapsulation_mean"]))

    _apply_scalers(graphs, ref_dir)

    print(f"{prefix}Device: {device}")

    all_preds: dict[str, list[float]] = {cid: [] for cid in valid_ids}

    for mi, mdir in enumerate(member_dirs):
        model = _load_model(mdir, device)
        preds = _predict(model, graphs, device)
        for cid in valid_ids:
            if cid in preds:
                all_preds[cid].append(preds[cid])
        print(f"{prefix}  Member {mi + 1}/{len(member_dirs)} OK")

    rows = []
    for cid in valid_ids:
        vals = all_preds[cid]
        gt = float(gt_map[cid])
        pred_mean = float(np.mean(vals))
        pred_std = float(np.std(vals))
        rows.append({
            "compound": cid,
            "ground_truth": gt,
            "predicted": pred_mean,
            "pred_std": pred_std,
            "abs_error": abs(pred_mean - gt),
        })

    out_df = pd.DataFrame(rows)
    if ensemble_iteration_col is not None:
        out_df.insert(0, "ensemble_iteration", ensemble_iteration_col)
    mae = float(out_df["abs_error"].mean())
    print(f"\n{prefix}Full-set MAE: {mae:.6f}  (n={len(out_df)})")
    return out_df, mae


def main():
    cfg0 = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv",
        default=cfg0.updated_training_csv,
        help="CSV with compound + encapsulation_mean (ground truth)",
    )
    parser.add_argument(
        "--data-dir",
        default=cfg0.combined_data_dir,
        help="Directory with compound subfolders + .itp files",
    )
    parser.add_argument("--nbfix", default=cfg0.nbfix_table)
    parser.add_argument(
        "--ensemble-iteration",
        type=int,
        default=None,
        help="Use active_learning/ensemble/iter_N/member_*/<latest>/",
    )
    parser.add_argument(
        "--use-tracker-ensemble",
        action="store_true",
        help="Use ensemble paths from active_learning/tracker.json instead",
    )
    parser.add_argument(
        "--all-ensemble-iterations",
        action="store_true",
        help="Run every active_learning/ensemble/iter_* (one CSV per AL ensemble iteration)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for single run (default: active_learning/calibration_full_iter_N.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for --all-ensemble-iterations (default: active_learning/)",
    )
    args = parser.parse_args()

    mode_ct = sum(
        1
        for x in (
            args.use_tracker_ensemble,
            args.ensemble_iteration is not None,
            args.all_ensemble_iterations,
        )
        if x
    )
    if mode_ct > 1:
        parser.error(
            "Use exactly one of: --ensemble-iteration N, --use-tracker-ensemble, "
            "--all-ensemble-iterations"
        )
    if mode_ct == 0:
        parser.error(
            "Provide --ensemble-iteration N, --use-tracker-ensemble, or "
            "--all-ensemble-iterations"
        )
    if args.all_ensemble_iterations and args.output is not None:
        parser.error("With --all-ensemble-iterations use --output-dir, not --output")

    config = cfg0
    training_csv = config.resolve(args.training_csv)
    data_dir = config.resolve(args.data_dir)
    nbfix = config.resolve(args.nbfix)

    if not training_csv.is_file():
        sys.exit(f"Training CSV not found: {training_csv}")
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.all_ensemble_iterations:
        ensemble_root = config.resolve(config.al_dir) / "ensemble"
        out_dir = (
            config.resolve(args.output_dir)
            if args.output_dir
            else config.resolve(config.al_dir)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        pairs = list_ensemble_iteration_dirs(ensemble_root)
        if not pairs:
            sys.exit(f"No ensemble/iter_* directories under {ensemble_root}")

        summary_rows = []
        for iter_n, iter_path in pairs:
            member_dirs = discover_member_run_dirs(iter_path)
            if not member_dirs:
                print(
                    f"Skipping iter_{iter_n}: no member model.pth under {iter_path}",
                    file=sys.stderr,
                )
                continue
            print(f"\n=== Ensemble AL iter_{iter_n} ({len(member_dirs)} members) ===")
            try:
                out_df, mae = run_single_ensemble_calibration(
                    config,
                    member_dirs,
                    training_csv,
                    data_dir,
                    nbfix,
                    device,
                    label=f"iter_{iter_n}",
                    ensemble_iteration_col=iter_n,
                )
            except (ValueError, RuntimeError) as e:
                print(
                    f"Skipping iter_{iter_n}: {e}",
                    file=sys.stderr,
                )
                continue
            out_path = out_dir / f"calibration_full_iter_{iter_n}.csv"
            out_df.to_csv(out_path, index=False)
            print(f"Wrote {out_path}")
            summary_rows.append({
                "ensemble_iteration": iter_n,
                "n_members": len(member_dirs),
                "n_compounds": len(out_df),
                "mae": mae,
            })

        if not summary_rows:
            sys.exit("No successful ensemble iterations — nothing written.")
        summary_path = out_dir / "ensemble_calibration_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary: {summary_path}")
        return

    member_dirs = resolve_ensemble_dirs(
        config, args.ensemble_iteration, args.use_tracker_ensemble
    )

    out_df, _mae = run_single_ensemble_calibration(
        config,
        member_dirs,
        training_csv,
        data_dir,
        nbfix,
        device,
        label="",
    )

    if args.output:
        out_path = Path(args.output)
    elif args.ensemble_iteration is not None:
        out_path = config.resolve(config.al_dir) / f"calibration_full_iter_{args.ensemble_iteration}.csv"
    else:
        out_path = config.resolve(config.al_dir) / "calibration_full_tracker_ensemble.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

