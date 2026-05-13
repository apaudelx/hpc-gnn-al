#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import apply_PBC

u = mda.Universe("plape_res.gro", "whole.xtc")

# selections
target = u.select_atoms("resname res")
target.masses = np.ones(len(target))
residues = target.residues
plape_c1 = u.select_atoms("resname plape and name C1")

# box lengths (nm)
box_lengths = np.array([12.29, 12.29, 12.29, 90.0, 90.0, 90.0])

# MIC-based COM function
def mic_center_of_mass(group, box):
    ref_pos = group.positions[0]
    shifted = apply_PBC(group.positions - ref_pos, box=box) + ref_pos
    #masses = group.masses
    #return np.average(shifted, axis=0, weights=masses)
    return shifted.mean(axis=0)

with open("res_com_core.dat", "w") as f, open("res_core_dist.dat", "w") as f2:
    # headers
    f.write("# time(ps) " +
            " ".join([f"res{res.resid}_x res{res.resid}_y res{res.resid}_z" for res in residues]) +
            " coreC1_x coreC1_y coreC1_z\n")

    f2.write("# time(ps) " + " ".join([f"dist_res{res.resid}_core" for res in residues]) + "\n")

    for ts in u.trajectory:
        line = [f"{ts.time:.2f}"]
        line2 = [f"{ts.time:.2f}"]

        # MIC COMs for each residue
        res_coms = []
        for res in residues:
            com = mic_center_of_mass(res.atoms, box_lengths)
            res_coms.append(com)
            line += [f"{x:.5f}" for x in com]

        # MIC COM for plape C1
        core_com = mic_center_of_mass(plape_c1, box_lengths)
        line += [f"{x:.5f}" for x in core_com]

        # distances resid-core
        for com in res_coms:
            diff = com - core_com
            dist = np.linalg.norm(diff)
            line2.append(f"{dist:.5f}")

        f.write(" ".join(line) + "\n")
        f2.write(" ".join(line2) + "\n")
        
        
# --- NEW PART: farthest C1 atom from plape C1 COM ---
with open("plape_c1_farthest.dat", "w") as f3:
    f3.write("# time(ps) farthest_C1_dist\n")

    for ts in u.trajectory:
        core_com = mic_center_of_mass(plape_c1, box_lengths)

        # compute distances of all C1 atoms from COM
        dists = []
        for atom in plape_c1:
            diff = atom.position - core_com
            dists.append(np.linalg.norm(diff))

        farthest = max(dists)
        f3.write(f"{ts.time:.2f} {farthest:.5f}\n")
        
        
# --- UPDATED PART: stats of res-core distances (median, Q1, Q3, mean, std, outliers) ---
with open("res_core_dist_stats.dat", "w") as f4:
    f4.write("# time(ps) median Q1 Q3 mean std outliers\n")

    for ts in u.trajectory:
        # MIC COM of core
        core_com = mic_center_of_mass(plape_c1, box_lengths)

        # MIC COMs of residues
        res_coms = [mic_center_of_mass(res.atoms, box_lengths) for res in residues]

        # distances to core
        dists = [np.linalg.norm(com - core_com) for com in res_coms]

        if len(dists) == 0:
            continue

        # quartiles
        q1 = np.percentile(dists, 25)
        median = np.median(dists)
        q3 = np.percentile(dists, 75)

        # mean and std
        mean_val = np.mean(dists)
        std_val = np.std(dists)

        # IQR method for outliers
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [f"{d:.5f}" for d in dists if d < lower_bound or d > upper_bound]

        f4.write(f"{ts.time:.2f} {median:.5f} {q1:.5f} {q3:.5f} {mean_val:.5f} {std_val:.5f} " + ",".join(outliers) + "\n")


# --- NEW PART: count how many residues are inside half of the farthest C1 radius + report max_core_dist ---
with open("res_core_halfcorecount.dat", "w") as f8:
    f8.write("# time(ps) max_core_dist N_inside_half_core\n")

    for ts in u.trajectory:
        # core COM
        core_com = mic_center_of_mass(plape_c1, box_lengths)

        # get max distance from core COM to any C1 atom (i.e., farthest core radius)
        max_core_dist = max(np.linalg.norm(atom.position - core_com) for atom in plape_c1)
        half_radius = 0.5 * max_core_dist

        # count how many residues lie within this half-radius
        count_inside = 0
        for res in residues:
            res_com = mic_center_of_mass(res.atoms, box_lengths)
            dist = np.linalg.norm(res_com - core_com)
            if dist <= half_radius:
                count_inside += 1

        f8.write(f"{ts.time:.2f} {max_core_dist:.5f} {count_inside}\n")


# --- NEW PART: boxplot of res-core distances every 20 frames (no whiskers, x-axis in ps) ---
import matplotlib.pyplot as plt
import numpy as np

# collect all distances per frame
all_times = []
all_dists = []

for ts in u.trajectory:
    core_com = mic_center_of_mass(plape_c1, box_lengths)
    res_coms = [mic_center_of_mass(res.atoms, box_lengths) for res in residues]
    dists = [np.linalg.norm(com - core_com) for com in res_coms]
    all_times.append(ts.time)
    all_dists.append(dists)

# group into bins of 20 frames
bin_size = 20
binned = []
bin_times = []

for i in range(0, len(all_dists), bin_size):
    binned.append(np.concatenate(all_dists[i:i+bin_size]))
    # midpoint time of the bin
    bin_times.append(np.mean(all_times[i:i+bin_size]))



# --- NEW PART: boxplot of res-core distances every 20 frames (time axis, no outliers) ---
import matplotlib.pyplot as plt

# collect all distances per frame
all_dists = []

for ts in u.trajectory:
    core_com = mic_center_of_mass(plape_c1, box_lengths)
    res_coms = [mic_center_of_mass(res.atoms, box_lengths) for res in residues]
    dists = [np.linalg.norm(com - core_com) for com in res_coms]
    all_dists.append(dists)

# group into bins of 20 frames
bin_size = 20
binned = [np.concatenate(all_dists[i:i+bin_size]) 
          for i in range(0, len(all_dists), bin_size)]

# make boxplot
plt.figure(figsize=(10, 5))
plt.boxplot(binned, positions=range(1, len(binned)+1), showfliers=False)

# x-axis ticks in ps, without shrinking boxes
xticks = range(1, len(binned)+1)
xtick_labels = [(i-1)*30000 for i in xticks]  # time in ps

plt.xticks(xticks, xtick_labels, rotation=45)
# y-axis fixed range
plt.ylim(0, 200)
plt.xlabel("Time (ps)")
plt.ylabel("Distance (Angs)")
#plt.title("Tricyclazole (1.42)–PLA Distances")
plt.tight_layout()
plt.savefig("res_core_boxplot.png", dpi=300)
#plt.show()

# --- NEW PART: metric summary file ---
with open("res_core_dispersion_metrics.dat", "w") as f5:
    f5.write("# time(ps) metric_custom  std_div_by_median  iqr_by_median\n")

    for ts in u.trajectory:
        # MIC COM of core
        core_com = mic_center_of_mass(plape_c1, box_lengths)

        # MIC COMs of residues
        res_coms = [mic_center_of_mass(res.atoms, box_lengths) for res in residues]

        # distances to core
        dists = [np.linalg.norm(com - core_com) for com in res_coms]

        if len(dists) == 0:
            continue

        # skip bad frames
        median = np.median(dists)
        if median < 1e-6:
            continue

        # core farthest distance
        dists_c1 = [np.linalg.norm(atom.position - core_com) for atom in plape_c1]
        farthest_c1 = max(dists_c1)

        # dispersion metrics
        std_div_median = np.std(dists) / median
        q1 = np.percentile(dists, 25)
        q3 = np.percentile(dists, 75)
        iqr_by_median = (q3 - q1) / median

        # your custom metric
        metric_custom = 1 - abs(farthest_c1 / median)

        f5.write(f"{ts.time:.2f} {metric_custom:.5f} {std_div_median:.5f} {iqr_by_median:.5f}\n")


# --- NEW PART: drug depth relative to cluster COM ---
with open("res_core_depth.dat", "w") as f6:
    f6.write("# time(ps) avg_depth " + " ".join([f"res_{r.resid}" for r in residues]) + "\n")

    for ts in u.trajectory:
        # define cluster (core + all drugs)
        cluster = plape_c1 #+ target
        com_cluster = cluster.center_of_mass()

        # max distance from PLA-C1 atoms to COM → defines cluster radius
        cluster_radius = np.max(np.linalg.norm(plape_c1.positions - com_cluster, axis=1))

        depths = []
        for res in residues:
            com_res = res.atoms.center_of_mass()
            dist = np.linalg.norm(com_res - com_cluster)
            depth = abs(1 - (dist / cluster_radius))
            depths.append(depth)

        avg_depth = np.mean(depths)
        f6.write(f"{ts.time:.2f} {avg_depth:.3f} " + " ".join(f"{d:.3f}" for d in depths) + "\n")


# --- NEW PART: count how many residues are inside or outside the cluster ---
with open("res_core_inside_outside_count.dat", "w") as f7:
    f7.write("# time(ps) N_inside N_outside\n")

    for ts in u.trajectory:
        cluster = plape_c1 + target
        com_cluster = cluster.center_of_mass()
        cluster_radius = np.max(np.linalg.norm(plape_c1.positions - com_cluster, axis=1))

        N_inside = 0
        N_outside = 0

        for res in residues:
            com_res = res.atoms.center_of_mass()
            dist = np.linalg.norm(com_res - com_cluster)
            if dist <= cluster_radius:
                N_inside += 1
            else:
                N_outside += 1
            ratio = N_inside / (N_inside + N_outside)
        f7.write(f"{ts.time:.3f} {ratio} {N_inside} {N_outside}\n")


# --- NEW PART: compute radius of gyration (Rg) of the core (plape C1 atoms) ---
with open("core_cluster_rg.dat", "w") as f9:
    f9.write("# time(ps) Rg_core(nm)\n")

    for ts in u.trajectory:
        core_com = mic_center_of_mass(plape_c1, box_lengths)
        positions = plape_c1.positions
        displacements = positions - core_com
        squared_distances = np.sum(displacements**2, axis=1)
        rg_core = np.sqrt(np.mean(squared_distances))  # in nm

        f9.write(f"{ts.time:.2f} {rg_core:.5f}\n")
        
# --- UPDATED PART: drug depth relative to cluster COM, normalized by Rg ---
with open("res_core_depth_rg.dat", "w") as f10:
    f10.write("# time(ps) avg_depth_Rg " + " ".join([f"res_{r.resid}" for r in residues]) + "\n")

    for ts in u.trajectory:
        # core COM
        core_com = mic_center_of_mass(plape_c1, box_lengths)

        # compute radius of gyration (Rg) of the core
        positions = plape_c1.positions
        displacements = positions - core_com
        squared_distances = np.sum(displacements**2, axis=1)
        rg_core = np.sqrt(np.mean(squared_distances))  # in nm

        # per-residue depth (1 = deeply buried, 0 = on Rg shell)
        depths = []
        for res in residues:
            res_com = mic_center_of_mass(res.atoms, box_lengths)
            dist = np.linalg.norm(res_com - core_com)
            depth = 1 - (dist / rg_core)
            depths.append(depth)

        avg_depth = np.mean(depths)
        f10.write(f"{ts.time:.2f} {avg_depth:.3f} " + " ".join(f"{d:.3f}" for d in depths) + "\n")

# --- NEW PART: count how many residues are inside half of the core Rg ---
with open("res_core_half_rg_count.dat", "w") as f11:
    f11.write("# time(ps) Rg_core N_inside_halfRg\n")

    for ts in u.trajectory:
        core_com = mic_center_of_mass(plape_c1, box_lengths)
        positions = plape_c1.positions
        displacements = positions - core_com
        rg_core = np.sqrt(np.mean(np.sum(displacements**2, axis=1)))

        half_rg = 0.5 * rg_core
        count_inside = 0

        for res in residues:
            res_com = mic_center_of_mass(res.atoms, box_lengths)
            dist = np.linalg.norm(res_com - core_com)
            if dist <= half_rg:
                count_inside += 1

        f11.write(f"{ts.time:.2f} {rg_core:.5f} {count_inside}\n")

# --- NEW PART: count residues inside/outside Rg shell ---
with open("res_core_inside_outside_count_rg.dat", "w") as f12:
    f12.write("# time(ps) Rg_core ratio_in_out N_inside N_outside\n")

    for ts in u.trajectory:
        core_com = mic_center_of_mass(plape_c1, box_lengths)
        positions = plape_c1.positions
        displacements = positions - core_com
        rg_core = np.sqrt(np.mean(np.sum(displacements**2, axis=1)))

        N_inside = 0
        N_outside = 0

        for res in residues:
            res_com = mic_center_of_mass(res.atoms, box_lengths)
            dist = np.linalg.norm(res_com - core_com)
            if dist <= 1.5*rg_core:
                N_inside += 1
            else:
                N_outside += 1

        ratio = N_inside / (N_inside + N_outside)
        f12.write(f"{ts.time:.2f} {rg_core:.5f} {ratio:.5f} {N_inside} {N_outside}\n")

# --- NEW PART: dispersion metric using Rg instead of farthest C1 ---
with open("res_core_dispersion_metrics_rg.dat", "w") as f13:
    f13.write("# time(ps) metric_custom_rg  std_div_by_median  iqr_by_median\n")

    for ts in u.trajectory:
        core_com = mic_center_of_mass(plape_c1, box_lengths)
        res_coms = [mic_center_of_mass(res.atoms, box_lengths) for res in residues]
        dists = [np.linalg.norm(com - core_com) for com in res_coms]

        if len(dists) == 0:
            continue

        median = np.median(dists)
        if median < 1e-6:
            continue

        # Rg of core
        positions = plape_c1.positions
        displacements = positions - core_com
        rg_core = np.sqrt(np.mean(np.sum(displacements**2, axis=1)))

        std_div_median = np.std(dists) / median
        q1 = np.percentile(dists, 25)
        q3 = np.percentile(dists, 75)
        iqr_by_median = (q3 - q1) / median

        metric_custom = 1 - (rg_core / median)
        f13.write(f"{ts.time:.2f} {metric_custom:.5f} {std_div_median:.5f} {iqr_by_median:.5f}\n")


# --- NEW PART: compute Rg of entire PLA (not just C1 atoms) ---
plape_all = u.select_atoms("resname plape")

with open("core_cluster_rg_fullplape.dat", "w") as f14:
    f14.write("# time(ps) Rg_plape_full(nm)\n")

    for ts in u.trajectory:
        core_com = mic_center_of_mass(plape_all, box_lengths)
        positions = plape_all.positions
        displacements = positions - core_com
        squared_distances = np.sum(displacements**2, axis=1)
        rg_full = np.sqrt(np.mean(squared_distances))  # in nm

        f14.write(f"{ts.time:.2f} {rg_full:.5f}\n")


        # --- NEW PART: compute mean of pesticide cluster from entire PLA (C1 atoms) ---
import numpy as np

# Select residues (drug)
residues = u.select_atoms("resname res").residues

# Collect distances per frame (in Å)
all_dists = []

for ts in u.trajectory:
    core_com = mic_center_of_mass(plape_c1, box_lengths)
    res_coms = [mic_center_of_mass(res.atoms, box_lengths) for res in residues]
    dists = [np.linalg.norm(com - core_com) for com in res_coms] 
    all_dists.append(dists)

# Bin every 20 frames
bin_size = 1
num_bins = len(all_dists) // bin_size

# Output: mean and std per bin
with open("res_core_dist_stats_mean.dat", "w") as f15:
    f15.write("# time(ps) mean_dist_Ang std_dist_Ang\n")
    for i in range(num_bins):
        bin_frames = all_dists[i * bin_size : (i + 1) * bin_size]
        bin_data = np.concatenate(bin_frames)
        mean_dist = np.mean(bin_data)
        std_dist = np.std(bin_data)
        time_ps = i * bin_size * u.trajectory.dt
        f15.write(f"{time_ps:.1f} {mean_dist:.3f} {std_dist:.3f}\n")

#print("Done. Stats written to res_core_stats.dat")
import numpy as np
import matplotlib.pyplot as plt

# Load mean and std data
data = np.loadtxt("res_core_dist_stats_mean.dat", comments="#")
time = data[:, 0] / 100
mean = data[:, 1]
std = data[:, 2]

# Plot mean ± std
plt.figure(figsize=(4, 3))
plt.plot(time, mean, color="black", linewidth=2)
plt.fill_between(time, mean - std, mean + std, color="gray", alpha=0.3)#, label="±1 Std Dev")

plt.ylim(0, 150)
plt.xlim(0, 750)
plt.xlabel("Time (ns)")
plt.ylabel("Distance (Å)")
#plt.title("Drug–Core Mean Distance ± Std Dev")
#xticks = np.arange(time[0], time[-1] + 1, 150)
#plt.xticks(xticks, rotation=45)
xticks = np.arange(0, 751, 150)
plt.xticks(xticks)
plt.legend()
plt.tight_layout()
plt.savefig("res_core_mean_std.png", dpi=300)
#plt.show()

