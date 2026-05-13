#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

shopt -s nullglob
mapfile -t MOL_DIRS < <(printf '%s\n' out_beads/*/ | sed 's:/$::')

if [ ${#MOL_DIRS[@]} -eq 0 ]; then
  echo "Error: No molecule directories found in out_beads/"
  exit 1
fi

ARRAY_MAX=$((${#MOL_DIRS[@]} - 1))

echo "Submitting array job: 0-$ARRAY_MAX (${#MOL_DIRS[@]} molecules)"
sbatch --array=0-$ARRAY_MAX submit_array.sh

