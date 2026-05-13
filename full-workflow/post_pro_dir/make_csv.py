#!/usr/bin/env python3
import csv

combined_file = "combined_lj.dat"
encap_file = "encapsulation_mean_vs_logP_plape_res.dat"
out_csv = "lj_encapsulation.csv"

# ---------
# Read encapsulation_mean_vs_logP.dat
# Keyed by logP (1st col), value = encapsulation mean (2nd col)
# We'll use a rounded key to avoid float-string mismatch
# ---------
encap_by_logp = {}
with open(encap_file, "r") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        logp = float(parts[0])
        enc_mean = float(parts[1])
        key = round(logp, 4)  # encap file has 4 decimals
        encap_by_logp[key] = enc_mean

# ---------
# Read combined_lj.dat and join on plape-res (2nd numeric column)
# combined format:
# compound  plape-plape  plape-res  res-res
# ---------
rows = []
missing = []

with open(combined_file, "r") as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue

        compound_raw = parts[0]
        compound = compound_raw.strip().rstrip("/")   # remove trailing slash

        lj_plape_plape = float(parts[1])
        lj_plape_res   = float(parts[2])  # <-- join key
        lj_res_res     = float(parts[3])

        key = round(lj_plape_res, 4)
        if key not in encap_by_logp:
            missing.append((compound, lj_plape_res))
            continue

        enc_mean = encap_by_logp[key]

        rows.append([compound, lj_plape_plape, lj_plape_res, lj_res_res, enc_mean])

# ---------
# Write CSV
# ---------
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["compound", "LJ_plape_plape", "LJ_plape_res", "LJ_res_res", "encapsulation_mean"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_csv}")

if missing:
    print("\nNo encapsulation match for these (compound, plape-res):")
    for comp, val in missing:
        print(f"  {comp}\t{val}")

