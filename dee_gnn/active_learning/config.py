"""Pipeline configuration for uncertainty-based active learning."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields, asdict
from pathlib import Path
import json
import os
from typing import List


def _default_project_root():
    return Path(__file__).resolve().parent.parent


def _default_sim_working_dir():
    env_value = os.environ.get("SIM_WORKING_DIR")
    if env_value:
        return env_value
    return str(_default_project_root().parent / "full-workflow")


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=_default_project_root)
    load_saved: InitVar[bool] = True

    # ── Data inputs ──────────────────────────────────────────────────────────
    new_molecules_dir: str = "pool_1"
    base_training_csv: str = "data/ee_values_739.csv"
    base_data_dir: str = "data/ee_itp_739"
    nbfix_table: str = "data/NBFIX_table"
    model_config: str = "config/config.json"
    results_dir: str = "results"

    # ── Test set (optional) ──────────────────────────────────────────────────
    test_set_dir: str = ""
    test_set_csv: str = ""

    # ── Active learning work paths ───────────────────────────────────────────
    al_dir: str = "active_learning"
    tracker_file: str = "active_learning/tracker.json"
    combined_data_dir: str = "active_learning/combined_data"
    iteration_output_dir: str = "active_learning/iterations"
    updated_training_csv: str = "active_learning/updated_training.csv"

    # ── Simulation (HPC) ────────────────────────────────────────────────────
    sim_working_dir: str = field(default_factory=_default_sim_working_dir)
    sim_output_csv: str = "out_beads/lj_encapsulation.csv"

    # ── Ensemble ─────────────────────────────────────────────────────────────
    ensemble_size: int = 5
    ensemble_seeds: List[int] = field(
        default_factory=lambda: [121, 122, 123, 124, 125]
    )

    # ── Pipeline parameters ──────────────────────────────────────────────────
    batch_size: int = 1000
    max_simulations: int = 20
    uncertainty_threshold: float = 0.05
    training_epochs: int = 200
    seed: int = 121

    # ── SLURM ────────────────────────────────────────────────────────────────
    slurm_account: str = "loni_zwitterion"
    slurm_partition: str = "checkpt"
    slurm_time_sim: str = "72:00:00"
    slurm_cpus_sim: int = 64
    slurm_time_ml: str = "04:00:00"
    slurm_cpus_ml: int = 4

    # ── Helpers ──────────────────────────────────────────────────────────────

    def __post_init__(self, load_saved: bool):
        if load_saved:
            self._load_saved_config()

    def _load_saved_config(self):
        config_path = self.resolve(self.al_dir) / "pipeline_config.json"
        if not config_path.exists():
            return

        with open(config_path) as f:
            data = json.load(f)

        valid = {f.name for f in fields(self)}
        if "project_root" in data:
            root = Path(data["project_root"])
            self.project_root = root if root.is_absolute() else (config_path.parent / root).resolve()

        for key, value in data.items():
            if key not in valid or key == "project_root":
                continue
            if key == "sim_working_dir":
                sim_dir = Path(value)
                value = str(sim_dir if sim_dir.is_absolute() else (self.project_root / sim_dir).resolve())
            setattr(self, key, value)

    def resolve(self, rel_path: str) -> Path:
        p = Path(rel_path)
        return p if p.is_absolute() else self.project_root / p

    def save(self, path: str | Path | None = None):
        dest = (
            Path(path) if path
            else self.resolve(self.al_dir) / "pipeline_config.json"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["project_root"] = str(data["project_root"])
        with open(dest, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        if "project_root" in filtered:
            filtered["project_root"] = Path(filtered["project_root"])
        return cls(**filtered, load_saved=False)
