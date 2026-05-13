"""Run ensemble GNN prediction with uncertainty estimation.

Loads all ensemble models, runs inference on the current batch,
and computes the mean and standard deviation of predictions.

Output: predictions.json in the iteration directory.

Usage:
    python active_learning/scripts/predict_ensemble.py --iteration 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
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
    """Apply node/edge/graph scalers from a trained model to graphs."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    config = PipelineConfig()
    tracker = PipelineTracker(config.resolve(config.tracker_file))

    batch = tracker.get_next_batch(config.batch_size)
    if not batch:
        print("No unprocessed molecules remaining.")
        sys.exit(0)

    ensemble_models = tracker.ensemble_models
    if not ensemble_models:
        print("Error: no ensemble models found in tracker.")
        sys.exit(1)

    iter_dir = config.resolve(config.iteration_output_dir) / f"iter_{args.iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    with open(iter_dir / "batch.json", "w") as f:
        json.dump(batch, f, indent=2)

    print(f"Iteration {args.iteration}: {len(batch)} molecules, "
          f"{len(ensemble_models)} ensemble members")

    device = torch.device("cpu")
    new_mol_dir = config.resolve(config.new_molecules_dir)

    # Use first model's bead_type_to_id and scalers for graph building.
    # All ensemble members are trained on the same data, so these are identical.
    ref_dir = Path(ensemble_models[0])
    if not ref_dir.is_absolute():
        ref_dir = config.project_root / ref_dir

    bead_map_path = ref_dir / "bead_type_to_id.json"
    if not bead_map_path.is_file():
        print(
            f"Error: {bead_map_path} is required for inference compatible with trained "
            f"bead embeddings (same file saved during training next to model.pth).",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(bead_map_path) as f:
        bead_type_to_id = json.load(f)

    nbfix_map = parse_nbfix_table(str(config.resolve(config.nbfix_table)))
    builder = MolecularGraphBuilder(
        nbfix_map, data_dir=str(new_mol_dir), bead_type_to_id=bead_type_to_id,
    )

    graphs, valid_compounds = [], []
    for compound in batch:
        g = builder.build_graph(compound)
        if g:
            graphs.append(g)
            valid_compounds.append(compound)
        else:
            print(f"  Warning: could not build graph for {compound}")

    if not graphs:
        print("Error: no graphs built")
        sys.exit(1)

    _apply_scalers(graphs, ref_dir)

    # Run each ensemble member and collect predictions
    all_preds = {c: [] for c in valid_compounds}

    for i, model_path_str in enumerate(ensemble_models):
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            model_path = config.project_root / model_path

        model = _load_model(model_path, device)
        preds = _predict(model, graphs, device)

        for compound in valid_compounds:
            if compound in preds:
                all_preds[compound].append(preds[compound])

        print(f"  Member {i}: predictions complete")

    # Aggregate: mean and std across ensemble
    results = []
    for compound in valid_compounds:
        p = all_preds[compound]
        results.append({
            "compound": compound,
            "mean": round(float(np.mean(p)), 6),
            "std": round(float(np.std(p)), 6),
            "predictions": [round(v, 6) for v in p],
        })

    output_path = iter_dir / "predictions.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved to {output_path}")
    print(f"{'Compound':<30} {'Mean':>8} {'Std':>8}")
    print("-" * 48)
    for r in results:
        print(f"  {r['compound']:<28} {r['mean']:>8.4f} {r['std']:>8.4f}")


if __name__ == "__main__":
    main()

