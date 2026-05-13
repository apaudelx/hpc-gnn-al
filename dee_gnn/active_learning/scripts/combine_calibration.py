"""Combine calibration_full_iter_*.csv files for compounds starting with bigm_.

Outputs:
1) A merged file containing all matching rows from all iterations.
2) A unique-per-compound file (latest iteration per compound by default).

Example:
    python active_learning/scripts/combine_bigm_calibration.py \
        --input-dir active_learning/calibration_by_iter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _iter_index_from_name(path: Path) -> int:
    stem = path.stem  # calibration_full_iter_6
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def _load_simulated_by_iteration(iterations_dir: Path) -> dict[int, set[str]]:
    """Read iter_*/simulate_list.txt files into {iteration: {compound,...}}."""
    out: dict[int, set[str]] = {}
    if not iterations_dir.is_dir():
        return out

    for sim_file in sorted(iterations_dir.glob("iter_*/simulate_list.txt")):
        iter_name = sim_file.parent.name
        try:
            iter_idx = int(iter_name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        compounds: set[str] = set()
        with open(sim_file, "r", encoding="utf-8") as f:
            for line in f:
                c = line.strip()
                if c:
                    compounds.add(c)
        out[iter_idx] = compounds

    return out


def _fmt_iter_list(iters: list[int]) -> str:
    return ";".join(str(i) for i in sorted(iters)) if iters else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="active_learning/calibration_by_iter",
        help="Directory containing calibration_full_iter_*.csv files",
    )
    parser.add_argument(
        "--prefix",
        default="mol_",
        help="Compound prefix to keep (default: mol_)",
    )
    parser.add_argument(
        "--merged-out",
        default="mol_all_iterations_merged.csv",
        help="Output CSV filename for all matching rows",
    )
    parser.add_argument(
        "--iterations-dir",
        default="active_learning/iterations",
        help="Directory containing iter_*/simulate_list.txt files",
    )
    parser.add_argument(
        "--unique-out",
        default="mol_unique_latest.csv",
        help="Output CSV filename for one row per compound",
    )
    parser.add_argument(
        "--write-unique",
        action="store_true",
        help="Also write one-row-per-compound output (disabled by default)",
    )
    parser.add_argument(
        "--unique-policy",
        choices=["latest", "earliest", "min_abs_error"],
        default="latest",
        help="How to select one row when a compound appears in multiple iterations",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    iterations_dir = Path(args.iterations_dir).expanduser().resolve()

    files = sorted(
        input_dir.glob("calibration_full_iter_*.csv"),
        key=_iter_index_from_name,
    )
    if not files:
        sys.exit(f"No calibration_full_iter_*.csv files found in: {input_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "compound" not in df.columns:
            print(f"Skipping {f.name}: missing 'compound' column", file=sys.stderr)
            continue

        if "ensemble_iteration" not in df.columns:
            df["ensemble_iteration"] = _iter_index_from_name(f)

        m = df["compound"].astype(str).str.startswith(args.prefix)
        df = df.loc[m].copy()
        if df.empty:
            continue

        df["source_file"] = f.name
        frames.append(df)

    if not frames:
        sys.exit(f"No rows found with prefix '{args.prefix}'")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["compound", "ensemble_iteration"]).reset_index(drop=True)

    simulated_by_iter = _load_simulated_by_iteration(iterations_dir)
    simulated_any = set().union(*simulated_by_iter.values()) if simulated_by_iter else set()
    compound_to_iters: dict[str, list[int]] = {}
    for iter_idx, names in simulated_by_iter.items():
        for name in names:
            compound_to_iters.setdefault(name, []).append(iter_idx)

    def _simulated_this_row(row) -> bool:
        iter_idx = int(row["ensemble_iteration"])
        comp = str(row["compound"])
        return comp in simulated_by_iter.get(iter_idx, set())

    merged["simulated_this_iteration"] = merged.apply(_simulated_this_row, axis=1)
    merged["simulated_in_any_iteration"] = merged["compound"].astype(str).isin(simulated_any)
    merged["first_simulated_iteration"] = merged["compound"].astype(str).map(
        lambda c: min(compound_to_iters.get(c, [])) if compound_to_iters.get(c, []) else pd.NA
    )
    merged["simulated_iterations"] = merged["compound"].astype(str).map(
        lambda c: _fmt_iter_list(compound_to_iters.get(c, []))
    )

    merged_out = input_dir / args.merged_out
    merged.to_csv(merged_out, index=False)

    unique_df = None
    unique_out = input_dir / args.unique_out
    if args.write_unique:
        if args.unique_policy == "latest":
            unique_df = (
                merged.sort_values(["compound", "ensemble_iteration"]) 
                .drop_duplicates(subset=["compound"], keep="last")
                .reset_index(drop=True)
            )
        elif args.unique_policy == "earliest":
            unique_df = (
                merged.sort_values(["compound", "ensemble_iteration"]) 
                .drop_duplicates(subset=["compound"], keep="first")
                .reset_index(drop=True)
            )
        else:
            if "abs_error" not in merged.columns:
                sys.exit("Cannot use --unique-policy min_abs_error because 'abs_error' column is missing")
            unique_df = (
                merged.sort_values(["compound", "abs_error", "ensemble_iteration"]) 
                .drop_duplicates(subset=["compound"], keep="first")
                .reset_index(drop=True)
            )

        unique_df.to_csv(unique_out, index=False)

    print("Done")
    print(f"  Input files used: {len(files)}")
    print(f"  Total matching rows: {len(merged)}")
    print(f"  Unique compounds: {merged['compound'].nunique()}")
    print(f"  Iterations dir used: {iterations_dir}")
    print(f"  Merged output: {merged_out}")
    if args.write_unique:
        print(f"  Unique output: {unique_out}")


if __name__ == "__main__":
    main()

