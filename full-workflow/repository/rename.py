#!/usr/bin/env python3
import os

# get current working directory
cwd = os.getcwd()

# take the last part of the path as the tag
tag = os.path.basename(cwd)

print(f"Using tag: {tag}")

# loop through files in the current directory
for fname in os.listdir("."):
    if not os.path.isfile(fname):
        continue
    
    base, ext = os.path.splitext(fname)
    
    # only process .dat files
    if ext == ".dat" and not base.endswith(tag):
        new_name = f"{base}_{tag}{ext}"
        os.rename(fname, new_name)
        print(f"Renamed: {fname} -> {new_name}")

