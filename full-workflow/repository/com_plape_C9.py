#!/usr/bin/env python3
import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import apply_PBC

u = mda.Universe("plape_res.gro", "whole.xtc")

# selections
target = u.select_atoms("resname res")
residues = target.residues
plape_c1 = u.select_atoms("resname plape and name C1")

# box lengths (nm)
box_lengths = np.array([12.29, 12.29, 12.29, 90.0, 90.0, 90.0])

# MIC-based COM function
def mic_center_of_mass(group, box):
    ref_pos = group.positions[0]
    shifted = apply_PBC(group.positions - ref_pos, box=box) + ref_pos
    masses = group.masses
    return np.average(shifted, axis=0, weights=masses)

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
plt.title("Tricyclazole (1.42)–PLA Distances")
plt.tight_layout()
plt.savefig("res_core_boxplot.png", dpi=300)
plt.show()

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
plt.show()


# --- NEW PART: MSD of drug residues (inside vs outside polymer core) ---
with open("res_com_core.dat") as f:
    header = f.readline().strip().split()
data = np.loadtxt("res_com_core.dat")

time_ps = data[:, 0]
n_frames = len(time_ps)

# number of drug residues
n_drugs = (data.shape[1] - 1 - 3) // 3
drug_com = data[:, 1:1 + 3*n_drugs].reshape(n_frames, n_drugs, 3)

# compute per-frame cartesian displacements
dx = drug_com[1:, :, 0] - drug_com[:-1, :, 0]  # (n_frames-1, n_drugs)
dy = drug_com[1:, :, 1] - drug_com[:-1, :, 1]
dz = drug_com[1:, :, 2] - drug_com[:-1, :, 2]

# write to file
with open("res_frame_displacement.dat", "w") as f16:
    # header
    hdr = ["# time(ps)"]
    for j in range(n_drugs):
        hdr += [f"dx_res{j+1}", f"dy_res{j+1}", f"dz_res{j+1}"]
    f16.write(" ".join(hdr) + "\n")

    for i in range(n_frames - 1):
        line = [f"{time_ps[i+1]:.2f}"]
        for j in range(n_drugs):
            line += [f"{dx[i, j]:.4f}", f"{dy[i, j]:.4f}", f"{dz[i, j]:.4f}"]
        f16.write(" ".join(line) + "\n")


# box length in Å (adjust if different)
L = 122.9

# apply MIC on dx, dy, dz
dx -= L * np.round(dx / L)
dy -= L * np.round(dy / L)
dz -= L * np.round(dz / L)

# write corrected displacements
with open("res_frame_displacement_unfolded.dat", "w") as f17:
    hdr = ["# time(ps)"]
    for j in range(n_drugs):
        hdr += [f"dx_res{j+1}", f"dy_res{j+1}", f"dz_res{j+1}"]
    f17.write(" ".join(hdr) + "\n")

    for i in range(n_frames - 1):
        line = [f"{time_ps[i+1]:.2f}"]
        for j in range(n_drugs):
            line += [f"{dx[i, j]:.4f}", f"{dy[i, j]:.4f}", f"{dz[i, j]:.4f}"]
        f17.write(" ".join(line) + "\n")

# per-residue displacement magnitude (Å)
disp_mag = np.sqrt(dx**2 + dy**2 + dz**2)  # (n_frames-1, n_drugs)

# write to file
with open("res_frame_displacement_magnitude.dat", "w") as f18:
    hdr = ["# time(ps)"]
    for j in range(n_drugs):
        hdr.append(f"disp_res{j+1}")
    f18.write(" ".join(hdr) + "\n")

    for i in range(n_frames - 1):
        line = [f"{time_ps[i+1]:.2f}"]
        for j in range(n_drugs):
            line.append(f"{disp_mag[i, j]:.4f}")
        f18.write(" ".join(line) + "\n")


# reconstruct unwrapped positions by cumulative sum (Å)
x_unwrap = np.cumsum(np.vstack([np.zeros((1, n_drugs)), dx]), axis=0)
y_unwrap = np.cumsum(np.vstack([np.zeros((1, n_drugs)), dy]), axis=0)
z_unwrap = np.cumsum(np.vstack([np.zeros((1, n_drugs)), dz]), axis=0)

pos_unwrap = np.stack([x_unwrap, y_unwrap, z_unwrap], axis=2)  # (n_frames, n_drugs, 3)


n_frames = pos_unwrap.shape[0]
max_lag = n_frames // 3
msd = np.zeros(max_lag)
counts = np.zeros(max_lag, dtype=int)

for lag in range(1, max_lag):
    disp = pos_unwrap[lag:] - pos_unwrap[:-lag]       # (n_frames-lag, n_drugs, 3)
    sq_disp = np.sum(disp**2, axis=2)                 # squared distances
    msd[lag] = np.mean(sq_disp)                       # average over all
    counts[lag] = sq_disp.size

dt_ps = np.mean(np.diff(time_ps))

with open("res_core_msd.dat", "w") as f19:
    f19.write("# lag(ps) MSD(nm^2) N_pairs\n")
    for lag in range(1, max_lag):
        lag_time = lag * dt_ps
        f19.write(f"{lag_time:.2f} {msd[lag]:.6f} {counts[lag]}\n")


# --- load res_com_core.dat ---
with open("res_com_core.dat") as f:
    header = f.readline().strip().split()
data = np.loadtxt("res_com_core.dat")
time_ps = data[:, 0]
n_frames = len(time_ps)

# number of residues
n_drugs = (data.shape[1] - 1 - 3) // 3
drug_com = data[:, 1:1 + 3*n_drugs].reshape(n_frames, n_drugs, 3)
core_com = data[:, -3:]

# --- load farthest distances ---
farthest = np.loadtxt("plape_c1_farthest.dat", comments="#")
core_radius = farthest[:, 1]  # Å

# distances drug-core
dist = np.linalg.norm(drug_com - core_com[:, None, :], axis=2)  # (n_frames, n_drugs)
inside = dist <= core_radius[:, None]

# --- compute displacements ---
dx = drug_com[1:, :, 0] - drug_com[:-1, :, 0]
dy = drug_com[1:, :, 1] - drug_com[:-1, :, 1]
dz = drug_com[1:, :, 2] - drug_com[:-1, :, 2]

# --- write raw displacements (inside only) ---
with open("res_frame_displacement_inside.dat", "w") as f21:
    hdr = ["# time(ps)"]
    for j in range(n_drugs):
        hdr += [f"dx_res{j+1}", f"dy_res{j+1}", f"dz_res{j+1}"]
    f21.write(" ".join(hdr) + "\n")

    for i in range(n_frames - 1):
        line = [f"{time_ps[i+1]:.2f}"]
        for j in range(n_drugs):
            if inside[i+1, j] and inside[i, j]:
                line += [f"{dx[i, j]:.4f}", f"{dy[i, j]:.4f}", f"{dz[i, j]:.4f}"]
            else:
                line += ["0.0", "0.0", "0.0"]
        f21.write(" ".join(line) + "\n")

# --- MIC correction ---
L = 122.9  
dx_mic = dx - L * np.round(dx / L)
dy_mic = dy - L * np.round(dy / L)
dz_mic = dz - L * np.round(dz / L)

# --- write MIC displacements (inside only) ---
with open("res_frame_displacement_inside_unfolded.dat", "w") as f22:
    hdr = ["# time(ps)"]
    for j in range(n_drugs):
        hdr += [f"dx_res{j+1}", f"dy_res{j+1}", f"dz_res{j+1}"]
    f22.write(" ".join(hdr) + "\n")

    for i in range(n_frames - 1):
        line = [f"{time_ps[i+1]:.2f}"]
        for j in range(n_drugs):
            if inside[i+1, j] and inside[i, j]:
                line += [f"{dx_mic[i, j]:.4f}", f"{dy_mic[i, j]:.4f}", f"{dz_mic[i, j]:.4f}"]
            else:
                line += ["0.0", "0.0", "0.0"]
        f22.write(" ".join(line) + "\n")

################ MSD inside core (with MIC unwrapping)

# --- load res_com_core.dat (COMs of drugs + core COM) ---
with open("res_com_core.dat") as f:
    header = f.readline().strip().split()
data = np.loadtxt("res_com_core.dat")

time_ps = data[:, 0]
n_frames = len(time_ps)

# number of drug residues
n_drugs = (data.shape[1] - 1 - 3) // 3
drug_com = data[:, 1:1 + 3*n_drugs].reshape(n_frames, n_drugs, 3)
core_com = data[:, -3:]

# --- load plape_c1_farthest.dat (farthest C1 distance per frame) ---
farthest = np.loadtxt("plape_c1_farthest.dat", comments="#")
core_radius = farthest[:, 1]  # 

# --- build inside mask ---
dist = np.linalg.norm(drug_com - core_com[:, None, :], axis=2)  # (n_frames, n_drugs)
inside = dist <= core_radius[:, None]

# --- unwrap positions with MIC ---
L = 122.9 
dx = drug_com[1:, :, 0] - drug_com[:-1, :, 0]
dy = drug_com[1:, :, 1] - drug_com[:-1, :, 1]
dz = drug_com[1:, :, 2] - drug_com[:-1, :, 2]

dx -= L * np.round(dx / L)
dy -= L * np.round(dy / L)
dz -= L * np.round(dz / L)

# cumulative sum to reconstruct continuous trajectories
x_unwrap = np.cumsum(np.vstack([np.zeros((1, n_drugs)), dx]), axis=0)
y_unwrap = np.cumsum(np.vstack([np.zeros((1, n_drugs)), dy]), axis=0)
z_unwrap = np.cumsum(np.vstack([np.zeros((1, n_drugs)), dz]), axis=0)

pos_unwrap = np.stack([x_unwrap, y_unwrap, z_unwrap], axis=2)  # (n_frames, n_drugs, 3)

# --- parameters ---
dt_ps = np.mean(np.diff(time_ps))
max_lag = n_frames // 3

# storage
msd_accum = np.zeros(max_lag)
count_accum = np.zeros(max_lag, dtype=int)

# --- loop over residues and build inside segments ---
for j in range(n_drugs):
    inside_idx = np.where(inside[:, j])[0]
    if inside_idx.size == 0:
        continue

    # split into contiguous segments
    breaks = np.where(np.diff(inside_idx) > 1)[0] + 1
    segments = np.split(inside_idx, breaks)

    for seg in segments:
        if len(seg) < 2:
            continue

        pos_seg = pos_unwrap[seg, j, :]
        seg_len = len(seg)
        seg_maxlag = min(max_lag, seg_len)

        for lag in range(1, seg_maxlag):
            disp = pos_seg[lag:] - pos_seg[:-lag]
            sq_disp = np.sum(disp**2, axis=1)

            msd_accum[lag] += np.sum(sq_disp)
            count_accum[lag] += len(sq_disp)

# --- average and convert to 
msd_inside = np.zeros(max_lag)
valid = count_accum > 0
msd_inside[valid] = (msd_accum[valid] / count_accum[valid]) 

# --- write output ---
with open("res_core_msd_inside.dat", "w") as f20:
    f20.write("# lag(ps) MSD_inside(nm^2) N_pairs\n")
    for lag in range(1, max_lag):
        lag_time = lag * dt_ps
        f20.write(f"{lag_time:.2f} {msd_inside[lag]:.6f} {count_accum[lag]}\n")


## Obtain diffusion coefficient

# load MSD data
data = np.loadtxt("res_core_msd_inside.dat", comments="#")
time = data[:,0]   # ps
msd  = data[:,1]   # nm^2

# select last 1000 points
last_n = 1000
time_fit = time[-last_n:]
msd_fit  = msd[-last_n:]

# linear fit
coeffs = np.polyfit(time_fit, msd_fit, 1)
slope, intercept = coeffs

# predicted MSD values
msd_pred = slope * time_fit + intercept

# R^2 of the fit
ss_res = np.sum((msd_fit - msd_pred)**2)
ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
r2 = 1 - (ss_res / ss_tot)

# diffusivity
D = slope / 6.0  # Angs^2/ps

# write results to file
with open("diffusivity.dat", "w") as f23:
    f23.write("# slope(nm^2/ps) intercept(nm^2) D(nm^2/ps) R2\n")
    f23.write(f"{slope:.6e} {intercept:.6e} {D:.6e} {r2:.6f}\n")
