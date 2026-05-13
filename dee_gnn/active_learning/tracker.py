"""JSON-backed state tracker for uncertainty-based active learning."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class PipelineTracker:
    """Tracks molecule status, ensemble models, and iteration history."""

    def __init__(self, tracker_path: Path):
        self.path = Path(tracker_path)
        self.data: dict = self._load_or_create()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_or_create(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "molecule_pool": [],
            "batch_order": [],
            "molecules": {},
            "iterations": [],
            "current_iteration": 0,
            "ensemble_models": [],
            "training_csv_path": "",
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    # ── Initialization ───────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return len(self.data["molecule_pool"]) > 0

    def initialize(
        self,
        molecule_pool: List[str],
        batch_order: List[str],
        base_csv: str,
    ):
        self.data["molecule_pool"] = molecule_pool
        self.data["batch_order"] = batch_order
        self.data["training_csv_path"] = base_csv
        self.save()

    # ── Ensemble models ──────────────────────────────────────────────────────

    @property
    def ensemble_models(self) -> List[str]:
        return self.data["ensemble_models"]

    @ensemble_models.setter
    def ensemble_models(self, value: List[str]):
        self.data["ensemble_models"] = list(value)
        self.save()

    # ── Batch management ─────────────────────────────────────────────────────

    def get_processed_compounds(self) -> set:
        return {
            m for m, info in self.data["molecules"].items()
            if info.get("processed")
        }

    def get_next_batch(self, batch_size: int) -> List[str]:
        processed = self.get_processed_compounds()
        unprocessed = [
            m for m in self.data["batch_order"] if m not in processed
        ]
        return unprocessed[:batch_size]

    def has_unprocessed(self) -> bool:
        return len(self.get_next_batch(1)) > 0

    # ── Molecule records ─────────────────────────────────────────────────────

    def record_molecule(
        self,
        compound: str,
        pred_mean: float,
        pred_std: float,
        simulated: bool,
        sim_value: Optional[float] = None,
        added_to_training: bool = False,
        iteration: int = 0,
    ):
        self.data["molecules"][compound] = {
            "processed": True,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "simulated": simulated,
            "sim_value": sim_value,
            "added_to_training": added_to_training,
            "iteration": iteration,
        }

    def get_training_additions(self) -> List[str]:
        return [
            m for m, info in self.data["molecules"].items()
            if info.get("added_to_training")
        ]

    def get_molecule_sim_values(self) -> Dict[str, float]:
        return {
            m: info["sim_value"]
            for m, info in self.data["molecules"].items()
            if info.get("sim_value") is not None
        }

    # ── Iteration records ────────────────────────────────────────────────────

    @property
    def current_iteration(self) -> int:
        return self.data["current_iteration"]

    @property
    def training_csv_path(self) -> str:
        return self.data["training_csv_path"]

    @training_csv_path.setter
    def training_csv_path(self, value: str):
        self.data["training_csv_path"] = str(value)

    def record_iteration(
        self,
        iteration: int,
        batch: List[str],
        simulated_count: int,
        deferred_count: int,
        mean_uncertainty: float,
        ensemble_models: List[str],
        training_set_size: int,
        test_metrics: Optional[Dict] = None,
    ):
        self.data["iterations"].append({
            "iteration": iteration,
            "batch": batch,
            "simulated_count": simulated_count,
            "deferred_count": deferred_count,
            "mean_uncertainty": round(mean_uncertainty, 6),
            "ensemble_models": ensemble_models,
            "training_set_size": training_set_size,
            "test_metrics": test_metrics,
        })
        self.data["current_iteration"] = iteration
        self.data["ensemble_models"] = ensemble_models
        self.save()

