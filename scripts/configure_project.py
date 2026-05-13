#!/usr/bin/env python3
"""Configure local/HPC paths for the GNN active-learning workflow.

This script updates the small set of project files that carry deployment
specific paths or Slurm settings. It is intentionally conservative: it edits
known shell, Slurm, and JSON config files instead of walking through molecule
data or generated model artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEE_GNN_DIR = REPO_ROOT / "dee_gnn"
FULL_WORKFLOW_DIR = REPO_ROOT / "full-workflow"
DEFAULT_WORK_ROOT = Path("/work") / os.environ.get("USER", "USER")
DEFAULT_CONDA_PREFIX = DEFAULT_WORK_ROOT / "conda"

LEGACY_PROJECT_ROOTS = (
    "/ddnB/work/apaude4/dee_gnn",
)
LEGACY_SIM_DIRS = (
    "/work/apaude4/full-workflow",
)
LEGACY_CONDA_PREFIXES = (
    "/work/apaude4/anaconda3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update project paths and common Slurm settings."
    )
    parser.add_argument(
        "--dee-gnn-root",
        default=str(DEE_GNN_DIR),
        help="Absolute path to the dee_gnn directory on the system that will run jobs.",
    )
    parser.add_argument(
        "--full-workflow-root",
        default=str(FULL_WORKFLOW_DIR),
        help="Absolute path to the full-workflow directory on the system that will run jobs.",
    )
    parser.add_argument(
        "--conda-prefix",
        default=os.environ.get("HPC_CONDA_PREFIX", str(DEFAULT_CONDA_PREFIX)),
        help="Conda installation prefix on work storage, containing etc/profile.d/conda.sh.",
    )
    parser.add_argument(
        "--conda-env",
        default="hpc-gnn-al",
        help="Conda environment name activated by GPU training Slurm jobs.",
    )
    parser.add_argument(
        "--slurm-account",
        default="loni_zwitterion",
        help="Slurm account used in #SBATCH -A directives.",
    )
    parser.add_argument(
        "--cpu-partition",
        default="checkpt",
        help="Slurm partition for CPU and simulation jobs.",
    )
    parser.add_argument(
        "--gpu-partition",
        default="gpu2",
        help="Slurm partition for GPU training jobs.",
    )
    parser.add_argument(
        "--gromacs-module",
        default="gromacs/2021.7/intel-2021.5.0-intel-mpi-2021.5.1",
        help="Environment module loaded before GROMACS simulation stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would change without writing them.",
    )
    return parser.parse_args()


def normalize_abs(path_value: str) -> str:
    return str(Path(path_value).expanduser().resolve())


def update_text_file(path: Path, transform: Callable[[str], str], dry_run: bool) -> bool:
    original = path.read_text()
    updated = transform(original)
    if updated == original:
        return False
    if not dry_run:
        path.write_text(updated)
    return True


def replace_legacy_paths(text: str, project_root: str, sim_dir: str, conda_prefix: str) -> str:
    for old in LEGACY_PROJECT_ROOTS:
        text = text.replace(old, project_root)
    for old in LEGACY_SIM_DIRS:
        text = text.replace(old, sim_dir)
    for old in LEGACY_CONDA_PREFIXES:
        text = text.replace(old, conda_prefix)
    return text


def update_slurm_settings(
    text: str,
    *,
    account: str,
    partition: str,
    conda_prefix: str,
    conda_env: str,
    gromacs_module: str,
) -> str:
    text = re.sub(r"(?m)^#SBATCH -A .*$", f"#SBATCH -A {account}", text)
    text = re.sub(r"(?m)^#SBATCH --partition=.*$", f"#SBATCH --partition={partition}", text)
    text = re.sub(r"(?m)^#SBATCH -p .*$", f"#SBATCH -p {partition}", text)
    text = re.sub(r'(?m)^: "\$\{HPC_CONDA_PREFIX:=.*\}"$', f': "${{HPC_CONDA_PREFIX:={conda_prefix}}}"', text)
    text = re.sub(r'(?m)^: "\$\{HPC_CONDA_ENV:=.*\}"$', f': "${{HPC_CONDA_ENV:={conda_env}}}"', text)
    text = re.sub(r"(?m)^module load gromacs/.*$", f"module load {gromacs_module}", text)
    return text


def update_pipeline_json(path: Path, args: argparse.Namespace, dry_run: bool) -> bool:
    data = json.loads(path.read_text())
    data["project_root"] = args.dee_gnn_root
    data["sim_working_dir"] = args.full_workflow_root
    data["slurm_account"] = args.slurm_account
    data["slurm_partition"] = args.cpu_partition
    rendered = json.dumps(data, indent=2) + "\n"
    if rendered == path.read_text():
        return False
    if not dry_run:
        path.write_text(rendered)
    return True


def update_tracker_json(path: Path, project_root: str, dry_run: bool) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    target = str(Path(project_root) / "active_learning" / "updated_training.csv")
    if data.get("training_csv_path") == target:
        return False
    data["training_csv_path"] = target
    rendered = json.dumps(data, indent=2) + "\n"
    if not dry_run:
        path.write_text(rendered)
    return True


def main() -> None:
    args = parse_args()
    args.dee_gnn_root = normalize_abs(args.dee_gnn_root)
    args.full_workflow_root = normalize_abs(args.full_workflow_root)
    args.conda_prefix = normalize_abs(args.conda_prefix)

    changed: list[Path] = []

    pipeline_json = DEE_GNN_DIR / "active_learning" / "pipeline_config.json"
    if pipeline_json.exists() and update_pipeline_json(pipeline_json, args, args.dry_run):
        changed.append(pipeline_json)

    tracker_json = DEE_GNN_DIR / "active_learning" / "tracker.json"
    if update_tracker_json(tracker_json, args.dee_gnn_root, args.dry_run):
        changed.append(tracker_json)

    gpu_jobs = {"bootstrap.slurm", "retrain.slurm"}
    text_files = [
        FULL_WORKFLOW_DIR / "submit_array.sh",
        DEE_GNN_DIR / "reset_pipeline.sh",
        DEE_GNN_DIR / "active_learning" / "submit_iteration.sh",
        DEE_GNN_DIR / "active_learning" / "submit_multi_iteration.sh",
        DEE_GNN_DIR / "active_learning" / "config.py",
        DEE_GNN_DIR / "active_learning" / "slurm" / "simulate.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "bootstrap.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "retrain.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "bootstrap_cpu.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "retrain_cpu.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "filter.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "postprocess.slurm",
        DEE_GNN_DIR / "active_learning" / "slurm" / "predict.slurm",
    ]

    for path in text_files:
        if not path.exists():
            continue

        def transform(text: str, path: Path = path) -> str:
            text = replace_legacy_paths(text, args.dee_gnn_root, args.full_workflow_root, args.conda_prefix)
            partition = args.gpu_partition if path.name in gpu_jobs else args.cpu_partition
            return update_slurm_settings(
                text,
                account=args.slurm_account,
                partition=partition,
                conda_prefix=args.conda_prefix,
                conda_env=args.conda_env,
                gromacs_module=args.gromacs_module,
            )

        if update_text_file(path, transform, args.dry_run):
            changed.append(path)

    mode = "Would update" if args.dry_run else "Updated"
    if changed:
        print(f"{mode} {len(changed)} file(s):")
        for path in changed:
            print(f"  {path.relative_to(REPO_ROOT)}")
    else:
        print("No changes needed.")

    print("\nEffective settings:")
    print(f"  dee_gnn root:       {args.dee_gnn_root}")
    print(f"  full-workflow root: {args.full_workflow_root}")
    print(f"  conda prefix:       {args.conda_prefix}")
    print(f"  conda env:          {args.conda_env}")
    print(f"  Slurm account:      {args.slurm_account}")
    print(f"  CPU partition:      {args.cpu_partition}")
    print(f"  GPU partition:      {args.gpu_partition}")


if __name__ == "__main__":
    main()
