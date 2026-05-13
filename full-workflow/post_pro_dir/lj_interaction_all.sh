#!/bin/bash
for d in */; do
  (cd "$d" && echo "12 14 18" | srun gmx_mpi energy -f energy.edr -o lj_interactions.xvg > lj_interactions_avg.dat)
done

