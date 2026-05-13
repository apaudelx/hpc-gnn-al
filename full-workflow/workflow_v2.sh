#!/bin/bash
# Usage: ./workflow.sh <MOL_DIR>
# Example: ./workflow.sh 108-70-3

#------- Date - 09/22/2025 ---------#

if [ -z "$1" ]; then
    echo "Usage: $0 <MOL_DIR>"
    exit 1
fi

MOL_DIR=$1
ROOT=$(dirname "$MOL_DIR")
REPO_DIR="./repository"
POST_PRO_DIR="./post_pro_dir"

if [ ! -d "$MOL_DIR" ]; then
    echo "Error: Molecule directory $MOL_DIR does not exist!"
    exit 1
fi

echo "=== Running workflow for $MOL_DIR ==="

# --- Copy everything from repository into molecule folder ---
cp $REPO_DIR/* $MOL_DIR/

# --- Copy everything from post processing  into out_beads folder ---
if [ ! -f "$ROOT/some_expected_file" ]; then
    cp $POST_PRO_DIR/* "$ROOT/"
fi

cd $MOL_DIR || exit
MOL_ID=$(basename "$PWD")

# --- Rename molecule files for consistency ---
cp ${MOL_ID}.gro res.gro
cp ${MOL_ID}.itp res.itp

# --- Shift molecule ---
python3 shift.py   # Output: res_shifted.gro

# --- Minimization of shifted molecule ---
srun gmx_mpi grompp -f min.mdp -p topol1.top -c res_shifted.gro -o res_min.tpr -maxwarn 11
srun gmx_mpi mdrun -deffnm res_min -v
echo ">>>>><<<<<>>ASHIM><<<>>><<>>><<>>"
srun gmx_mpi editconf -f res_min.gro -o res_min.pdb

if [ ! -f "res_min.pdb" ]; then
    echo "Error: srun gmx_mpi editconf failed to create res_min.pdb"
    exit 1
fi


echo ">>>>><<<<<>>ASHIM><<<>>><<>>><<>>"
# --- Packmol solvation ---
packmol < pack_water.inp

# --- Add solvent with srun gmx_mpi solvate ---
srun gmx_mpi solvate -cp peg_pla_pes_solvated.pdb -cs solvent.gro -o solvated.gro -maxsol 14500 -box 15 15 15

# --- Minimization ---
srun gmx_mpi grompp -f min.mdp -p topol.top -c solvated.gro -o min.tpr -maxwarn 11
srun gmx_mpi mdrun -deffnm min -v

# --- NVT equilibration ---
srun gmx_mpi grompp -f nvt.mdp -p topol.top -c min.gro -o nvt.tpr -maxwarn 11
srun gmx_mpi mdrun -deffnm nvt -v

# --- NPT production ---
srun gmx_mpi grompp -f run2.mdp -p topol.top -c nvt.gro -o npt.tpr -maxwarn 11
srun gmx_mpi mdrun -deffnm npt -v

# --- Extracting only drug and polymer complex ---
module load vmd
vmd -dispdev text -e plape_res.tcl

# --- Index file creation for drug and polymer complex ---
srun gmx_mpi make_ndx -f npt.gro -o index.ndx << EOF
r PLAPE & a C1
r RES
name 5 PLAPE_C1
name 6 RES_ALL
5 | 6
name 7 PLAPE_C1_PLUS_RES
q
EOF

# --- Trajectory for drug and polymer complex ---
echo 7 | srun gmx_mpi trjconv -s npt.tpr -f npt.xtc -o plape_res.xtc -n index.ndx

# please see https://github.com/BartBruininks/mdvwhole to install mdvwhole
mdvwhole -f plape_res.gro -x plape_res.xtc -o whole.xtc

# --- Analysis ---
python3 com_plape_C8.py
python3 rename.py

#------ Energy Analysis ----- 
srun gmx_mpi make_ndx -f npt.gro -o index2.ndx << EOF
r plape
r res
q
EOF

srun gmx_mpi grompp -f run2_energy.mdp -c npt.gro -p topol.top -n index2.ndx -o energy.tpr
srun gmx_mpi mdrun -s energy.tpr -rerun npt.xtc -e energy.edr

echo "12 14 18" | srun gmx_mpi energy -f energy.edr -o lj_interactions.xvg

rm -rf \#*	

echo "=== Workflow till C8 Complete ==="

echo "=== RUNNING POST PROCESSING ==="
# --- Generate .dat and CSV files ---
cd ../.
bash lj_interaction_all.sh
bash combine_lj.sh
python3 plot_plape_res_fix.py
python3 make_csv.py

# Skip diffusion
#------- Date - 09/26/2025 ---------#
#------ Compute Diffusivity ----#

#mkdir -p diffusion
#mv diffusion.sh diffusion.mdp com_plape_C9.py diffusion/
#cp *.tcl topol.top *.itp npt.gro diffusion/
#
#cd diffusion/
#
#
## ------------- run diffusion simulation starting from npt --------------
#
#srun gmx_mpi grompp -f diffusion.mdp -p topol.top -c npt.gro -o diffusion.tpr -maxwarn 10
#srun gmx_mpi mdrun -deffnm diffusion -v
#
#
#module load vmd
#
## --- Extracting only drug and polymer complex ---
#vmd -dispdev text -e plape_res.tcl
#
## --- Index file creation for drug and polymer complex ---
#srun gmx_mpi make_ndx -f diffusion.gro -o index.ndx << EOF
#r plape & a C1
#r res
#name 5 PLAPE
#name 6 RES_ALL
#5 | 6
#name 7 PLAPE_PLUS_RES
#q
#EOF
#
## --- Trajectory for drug and polymer complex ---
#echo 7 | srun gmx_mpi trjconv -s diffusion.tpr -f diffusion.xtc -o plape_res.xtc -n index.ndx
#
## please see https://github.com/BartBruininks/mdvwhole to install mdvwhole
#mdvwhole -f plape_res.gro -x plape_res.xtc -o whole.xtc
#
## --- Analysis ---
#python3 com_plape_C9.py
#
echo "=== Workflow completed for $MOL_ID ==="
