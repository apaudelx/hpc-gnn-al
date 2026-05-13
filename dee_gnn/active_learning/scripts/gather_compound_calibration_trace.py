"""Collect one compound's calibration rows from all ``calibration_full_iter_*.csv`` files.

Use after ``export_full_calibration.py --all-ensemble-iterations``.

Example::

    python active_learning/scripts/gather_compound_calibration_trace.py bigm_047

    python active_learning/scripts/gather_compound_calibration_trace.py HEaSC00031 \\
        --input-dir active_learning --output analysis/trace.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from active_learning.config import PipelineConfig

_ITER_RE = re.compile(r"calibration_full_iter_(\d+)\.csv$", re.IGNORECASE)


def _iteration_from_path(path: Path) -> int | None:
    m = _ITER_RE.search(path.name)
    return int(m.group(1)) if m else None


def main():
    cfg0 = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "compound",
        help="Compound id as in the calibration CSV ``compound`` column (e.g. bigm_047)",
    )
    parser.add_argument(
        "--input-dir",
        default=cfg0.al_dir,
        help="Directory containing calibration_full_iter_*.csv (default: active_learning)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input-dir>/<compound>.csv)",
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Match compound case-insensitively",
    )
    args = parser.parse_args()

    config = cfg0
    input_dir = config.resolve(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("calibration_full_iter_*.csv"))
    if not files:
        sys.exit(f"No calibration_full_iter_*.csv files under {input_dir}")

    compound_needle = args.compound.strip()
    if not compound_needle:
        sys.exit("Empty compound id.")

    rows = []
    for path in sorted(files, key=lambda p: (_iteration_from_path(p) or -1, p.name)):
        it = _iteration_from_path(path)
        if it is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: skip {path}: {e}", file=sys.stderr)
            continue
        if "compound" not in df.columns:
            print(f"Warning: skip {path}: no 'compound' column", file=sys.stderr)
            continue

        if args.case_insensitive:
            sel = df["compound"].astype(str).str.lower() == compound_needle.lower()
        else:
            sel = df["compound"].astype(str) == compound_needle

        hit = df.loc[sel]
        if hit.empty:
            continue
        if len(hit) > 1:
            print(
                f"Warning: {path.name} has {len(hit)} rows for this compound; using the first.",
                file=sys.stderr,
            )
            hit = hit.iloc[:1]

        row = hit.iloc[0].to_dict()
        if "ensemble_iteration" not in row or pd.isna(row.get("ensemble_iteration")):
            row["ensemble_iteration"] = it
        rows.append(row)

    if not rows:
        print(
            f"No rows found for compound {compound_needle!r} in any of "
            f"{len(files)} calibration_full_iter_*.csv file(s) under:\n  {input_dir}\n"
            f"Check spelling and that export included this compound.",
            file=sys.stderr,
        )
        sys.exit(1)

    out = pd.DataFrame(rows)
    preferred = ["ensemble_iteration", "compound", "ground_truth", "predicted", "pred_std", "abs_error"]
    cols = [c for c in preferred if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = config.project_root / out_path
    else:
        safe = re.sub(r"[^\w.\-]+", "_", compound_needle)
        out_path = input_dir / f"{safe}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out)} iteration(s))")


if __name__ == "__main__":
    main()
