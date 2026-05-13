
# Load your .gro file
mol new npt.gro type gro


set sel [atomselect top "(resname plape and name C1) or resname res"]

$sel writegro plape_res.gro

quit

