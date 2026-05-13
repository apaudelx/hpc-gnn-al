#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque
from matplotlib import cm

# ---------------------------------------------------
# Read combined_lj.dat (CAS → logP value from 3rd column: parts[2])
# ---------------------------------------------------
logP_map = {}
with open("combined_lj.dat") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) >= 3:
            cas = parts[0].strip().rstrip("/")  # <-- FIX: remove trailing slash
            try:
                logP_val = float(parts[2])      # use logP column
                logP_map[cas] = logP_val
            except ValueError:
                continue

# ---------------------------------------------------
# Collect data from each CAS folder
# ---------------------------------------------------
depth_data = []
logP_values = []
cas_list = []

for cas in sorted(logP_map.keys()):
    data_file = os.path.join(cas, f"res_core_inside_outside_count_{cas}.dat")
    if not os.path.isfile(data_file):
        print(f"Missing data file for {cas}: {data_file}")
        continue

    values = []
    with open(data_file) as f:
        lines = deque(f, 250)  # last 250 lines

    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue

    if values:
        depth_data.append(values)
        logP_values.append(logP_map[cas])
        cas_list.append(cas)

# ---------------------------------------------------
# Stop early if nothing found
# ---------------------------------------------------
if not depth_data:
    raise SystemExit(
        "No valid data collected.\n"
        "Expected files like: <CAS>/res_core_inside_outside_count_<CAS>.dat\n"
        "Also confirm CAS names in combined_lj.dat match folder names."
    )

# ---------------------------------------------------
# Sort by logP
# ---------------------------------------------------
sorted_indices = np.argsort(logP_values)
depth_data = [depth_data[i] for i in sorted_indices]
logP_values = [logP_values[i] for i in sorted_indices]
cas_list = [cas_list[i] for i in sorted_indices]

# ---------------------------------------------------
# Print summary
# ---------------------------------------------------
print("Compounds sorted by logP (3rd column):")
for cas, logP in zip(cas_list, logP_values):
    print(f"{cas}: logP = {logP:.3f}")

# ---------------------------------------------------
# Plot violin
# ---------------------------------------------------
plt.figure(figsize=(18, 10))
parts = plt.violinplot(depth_data, showmedians=True, showmeans=False)

plt.xticks(
    ticks=np.arange(1, len(logP_values) + 1),
    labels=[f"{v:.2f}" for v in logP_values],
    rotation=45,
    ha="right"
)

plt.xlabel("logP", fontsize=16)  # <-- FIX (was "plape-res")
plt.ylabel("Number of particles inside the PLA core", fontsize=16)
plt.title("Variation of Encapsulation vs logP", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Colors
cmap = cm.get_cmap("tab20", len(logP_values))
colors = [cmap(i) for i in range(len(logP_values))]
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor("black")
    pc.set_alpha(0.8)

# Outline styling (LineCollections in modern matplotlib)
for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
    parts[partname].set_color("black")

# Add mean as red circle + mean trend line
means = [np.mean(v) for v in depth_data]
for i, mean_val in enumerate(means, start=1):
    plt.scatter(i, mean_val, facecolors="red", edgecolors="black",
                linewidths=0.8, marker="o", zorder=3)

plt.plot(range(1, len(depth_data) + 1), means,
         color="blue", linestyle="-", linewidth=3, zorder=2)

# Final styling
plt.tick_params(direction="in", width=1.5)
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig("violin_encapsulation_logP_plape_res.png", dpi=300)
plt.show()

# ---------------------------------------------------
# Save mean vs logP data to file
# ---------------------------------------------------
with open("encapsulation_mean_vs_logP_plape_res.dat", "w") as f:
    f.write("# logP\tmean_inside_PLA_core\n")
    for lp, mean in zip(logP_values, means):
        f.write(f"{lp:.4f}\t{mean:.6f}\n")

