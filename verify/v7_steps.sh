#!/bin/bash
# V7 cut-out (room 2S5, 359,854 cells, 7 regions, FCU supply / outdoor-air / return patches with fixed flow rates),
# OpenFOAM-runnable variant v7_std: OpenFOAM chtMultiRegionFoam and chtMultiRegionGpuFoam (BACKEND build) for NSTEPS
# steps of 0.02 s with NOUTER outer correctors, compared every step; then the cells next to the outdoor-air inlet.
#   v7_steps.sh [NSTEPS NOUTER]
# OPEN=1: the return grille hvac_HVAC_2S5_0_ra becomes a fixed-pressure opening written like the whole building's
# tower_louver_opening (prghTotalPressure / pressureInletOutletVelocity / inletOutlet). The cut-out is otherwise a
# closed box with fixed mass flows, which a Boussinesq (mass-conserving, rho(T)) solver cannot satisfy once cold
# supply air changes the mean density: the pressure equation has no solution and both solvers diverge.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-cuda}
NS=${1:-3}; NO=${2:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}
W=/w/cases/v7_steps_n${NO}$([ "${OPEN:-0}" = 1 ] && echo _open)
END=$(python3 -c "print(round(43200 + $NS*0.02, 6))")
for X in of gpu; do
  C=$W/$X
  rm -rf $C; mkdir -p $C
  cp -r /py/v7/v7_std/constant /py/v7/v7_std/system /py/v7/v7_std/0 $C/
  mv $C/0 $C/43200
  sed -i "s/endTime 43202;/endTime $END;/; s/writeInterval 3600;/writeInterval 0.02;/" $C/system/controlDict
  sed -i "s/nOuterCorrectors 20;/nOuterCorrectors $NO;/" $C/system/fvSolution
  if [ "${OPEN:-0}" = 1 ]; then
    python3 - $C/43200/air <<'PY'
import re, sys
d = sys.argv[1]
def internal(f):
    m = re.search(r"internalField\s+uniform\s+([^;]+);", open(f"{d}/{f}").read())
    return m.group(1).strip()
new = {
    "U": "type pressureInletOutletVelocity; value uniform (0 0 0);",
    "p_rgh": "type prghTotalPressure; p0 uniform 100090; value uniform 100090;",
    "T": f"type inletOutlet; inletValue uniform {internal('T')}; value uniform {internal('T')};",
    "k": "type inletOutlet; inletValue uniform 0.0001; value uniform 0.0001;",
    "epsilon": "type inletOutlet; inletValue uniform 1e-05; value uniform 1e-05;",
    "waterVapour": f"type inletOutlet; inletValue uniform {internal('waterVapour')}; value uniform {internal('waterVapour')};",
    "dryAir": f"type inletOutlet; inletValue uniform {internal('dryAir')}; value uniform {internal('dryAir')};",
}
for f, e in new.items():
    p = f"{d}/{f}"
    s = open(p).read()
    s2, n = re.subn(r"(\n\s+hvac_HVAC_2S5_0_ra )\{[^}\n]*\}", lambda m: m.group(1) + "{ " + e + " }", s)
    if n != 1:
        sys.exit(f"v7_steps OPEN: {f}: {n} matches for the return grille")
    open(p, "w").write(s2)
print("return grille -> fixed-pressure opening")
PY
  fi
done
(cd $W/of && { time chtMultiRegionFoam > log.run 2>&1; } 2> log.time) &
(cd $W/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time) &
wait
for X in of gpu; do
  echo "[$X] $(grep -c '^Time =' $W/$X/log.run) steps, wall $(grep real $W/$X/log.time)"
  grep -E "FOAM FATAL|gpucht FATAL|Floating point" -A4 $W/$X/log.run | head -8
  grep -E "Min/max T.*\(air\)" $W/$X/log.run | awk 'NR % 1 == 0' | tail -$NS
done
for i in $(seq 1 $NS); do
  t=$(python3 -c "print(round(43200 + $i*0.02, 6))")
  echo "== t=$t gpu vs of (V7 std, nOuter $NO)"
  python3 /w/verify/compare_of.py $W/gpu $t $W/of $t /tmp/v7s.json | grep -E "air/"
done
cd $W && python3 /w/verify/diag_spike.py gpu of $END air 4 2>&1 | head -12
python3 /w/verify/diag_patch.py gpu of $END air hvac_HVAC_2S5_0_oa 2>&1 | tail -3
echo V7_STEPS_DONE
