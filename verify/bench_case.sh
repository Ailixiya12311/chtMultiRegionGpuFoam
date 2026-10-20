#!/bin/bash
# Benchmark case: the V5 conjugate cavity made 3-D and refined (K times the V5 cell counts in x and y, NZ cells in z),
# scaled 30x like M4 (3 m cavity, 3 m deep), RNG k-epsilon with wall functions, from rest with hot/cold side walls
# (make_turb_v5.py "active"); the front/back faces are adiabatic no-slip walls.
#   bench_case.sh OUT K NZ [NSTEPS NOUTER DT]  ->  OUT/cht_frame, (76 K)^2 NZ cells, (60 K)^2 NZ of them air
#   K 2, NZ 60: 1.39 M cells (0.86 M air);  K 3, NZ 90: 4.68 M (2.92 M);  K 4, NZ 120: 11.1 M (6.91 M)
# Linear-solver tolerances as the V7 building case (p_rgh 1e-7, relTol 0.05 / Final 0.01; U h k epsilon 1e-9,
# relTol 0.1 / Final 0.01). deltaT 1 s is unstable on these meshes (OpenFOAM too: Co 354 at the third step);
# the default 0.05 s keeps Co of order 1.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -eu
OUT=$1; K=$2; NZ=$3; NSTEPS=${4:-5}; NOUTER=${5:-3}; DT=${6:-0.05}
END=$(python3 -c "print($NSTEPS*$DT)")
mkdir -p $OUT
C=$(python3 - "$OUT" "$K" "$NZ" <<'PY'
import re, sys
sys.path.insert(0, "/py/verify")
import cht_case
out, k, nz = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cht_case.NS = [8*k, 60*k, 8*k]
case = cht_case.mesh(out)
p = case + "/system/blockMeshDict"
s = open(p).read()
s = s.replace(" 1) simpleGrading", f" {nz}) simpleGrading")
s = s.replace("frontAndBack { type empty;", "frontAndBack { type wall;")
s = re.sub(r" 0\.01\)", " 0.1)", s)          # depth 0.1 m (3 m after the 30x scaling)
open(p, "w").write(s)
print(case)
PY
)
cd $C
blockMesh > log.blockMesh 2>&1
splitMeshRegions -cellZones -overwrite > log.split 2>&1
python3 /py/verify/cht_case.py regions $C
for r in air solidL solidR solidB solidT; do
  sed -i 's/frontAndBack { type empty; }/frontAndBack { type zeroGradient; }/' 0/$r/T
  sed -i 's/frontAndBack { type empty; }/frontAndBack { type calculated; value uniform 1e5; }/' 0/$r/p
done
sed -i 's/frontAndBack { type empty; }/frontAndBack { type noSlip; }/' 0/air/U
sed -i 's/frontAndBack { type empty; }/frontAndBack { type fixedFluxPressure; value uniform 1e5; }/' 0/air/p_rgh
for r in air solidL solidR solidB solidT; do transformPoints -region $r -scale 30 > /dev/null; done
python3 /py/verify/make_turb_v5.py $C $END $END $NOUTER active
sed -i 's/frontAndBack { type empty; }/frontAndBack { type kqRWallFunction; value uniform 1e-4; }/' 0/air/k
sed -i 's/frontAndBack { type empty; }/frontAndBack { type epsilonWallFunction; value uniform 1e-5; }/' 0/air/epsilon
sed -i 's/frontAndBack { type empty; }/frontAndBack { type nutkWallFunction; value uniform 0; }/' 0/air/nut
sed -i 's/frontAndBack { type empty; }/frontAndBack { type compressible::alphatJayatillekeWallFunction; Prt 0.85; value uniform 0; }/' 0/air/alphat
! grep -l "type empty" 0/*/* || { echo "bench_case: empty patches left"; exit 1; }
sed -i "s/deltaT 1;/deltaT $DT;/" system/controlDict
sed -i -e 's/tolerance 1e-10; relTol 0.01;/tolerance 1e-7; relTol 0.05;/' -e 's/tolerance 1e-10; relTol 0;/tolerance 1e-7; relTol 0.01;/' \
       -e 's/tolerance 1e-12; relTol 0.1;/tolerance 1e-9; relTol 0.1;/' -e 's/tolerance 1e-12; relTol 0;/tolerance 1e-9; relTol 0.01;/' \
       system/*/fvSolution
grep -q "deltaT $DT;" system/controlDict || { echo "bench_case: deltaT not set"; exit 1; }
echo $C
