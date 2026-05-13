def shift_gro_coordinates(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    header = lines[0]        # first line = title
    natoms = int(lines[1])   # second line = number of atoms
    atom_lines = lines[2:2+natoms]
    box_line = lines[2+natoms]  # last line = box dimensions

    # Extract all coordinates
    coords = []
    for line in atom_lines:
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        coords.append((x, y, z))

    # Find global minimum
    min_coord = min(min(x, y, z) for x, y, z in coords)
    shift = abs(min_coord) + 1.0

    # Rewrite coordinates with shift applied
    new_atom_lines = []
    for i, line in enumerate(atom_lines):
        res = line[:20]
        x, y, z = coords[i]
        new_line = f"{res}{x+shift:8.3f}{y+shift:8.3f}{z+shift:8.3f}\n"
        new_atom_lines.append(new_line)

    # Write new file
    with open(output_file, "w") as f:
        f.write(header)
        f.write(str(natoms).rjust(5) + "\n")
        f.writelines(new_atom_lines)
        f.write(box_line)  # keep box dimensions unchanged

    print(f"Shifted .gro file written to {output_file}, shift applied = {shift:.3f}")


shift_gro_coordinates("res.gro", "res_shifted.gro")
