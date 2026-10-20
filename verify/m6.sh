#!/bin/bash
# M6: Building physics, HVAC closed-loop controller, and adaptive time stepping:
#   Part 1 (adj): V5 conjugate cavity with adjustTimeStep (maxCo, maxDi, maxDeltaT) vs OpenFOAM v2606 chtMultiRegionFoam
#   Part 2 (v7f): V7 full building physics cut-out (359,854 cells, 7 regions, gpuCht::exteriorSurface, weather,
#                 solar ray-traced ShadowMap, interiorRadiation, and Python FCU/FAU HVAC closed-loop controller)
#                 comparing chtMultiRegionGpuFoam vs gpuChtFoam.py over 2 steps (startup + active return flow)
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-omp}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}

echo "=== Part 1: V5 conjugate cavity with adjustTimeStep (maxCo 120, maxDi 5.0, maxDeltaT 2.5, 15 outer) ==="
W1=/w/cases/m6_adj
REF=/py/cases/cht_frame
mkdir -p $W1
for X in of gpu; do
  C=$W1/$X
  if [ "$X" = of ] && [ -d "$C/10" ]; then continue; fi
  rm -rf $C; mkdir -p $C
  cp -r $REF/0 $REF/constant $REF/system $C/
  python3 /py/verify/make_transient_v5.py $C 10 5 15 > /dev/null
  rm -rf $C/0; cp -r $REF/4000 $C/
  python3 /py/verify/make_step_v5.py $C > /dev/null
  sed -i -E \
    -e 's/writeControl[[:space:]]+[^;]+;/writeControl adjustableRunTime;/g' \
    -e 's/writeInterval[[:space:]]+[^;]+;/writeInterval 5;/g' \
    -e 's/adjustTimeStep[[:space:]]+[^;]+;/adjustTimeStep yes;/g' \
    -e 's/maxCo[[:space:]]+[^;]+;/maxCo 120.0;/g' \
    -e 's/maxDi[[:space:]]+[^;]+;/maxDi 5.0; maxDeltaT 2.5;/g' \
    $C/system/controlDict
done
if [ ! -d "$W1/of/10" ]; then
  (cd $W1/of && { time chtMultiRegionFoam > log.run 2>&1; } 2> log.time)
fi
(cd $W1/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time)
echo "of  adj steps=$(grep -c '^Time =' $W1/of/log.run), $(grep -E '^ExecutionTime' $W1/of/log.run | tail -1)"
echo "gpu adj steps=$(grep -c '^Time =' $W1/gpu/log.run), $(grep -E '^ExecutionTime' $W1/gpu/log.run | tail -1), wall $(grep real $W1/gpu/log.time)"
echo "-- deltaT sequence (of vs gpu):"
paste <(grep '^deltaT =' $W1/of/log.run) <(grep '^deltaT =' $W1/gpu/log.run) | head -15
for t in 5 10; do
  echo "== t=$t gpu vs of (M6 V5 adjustTimeStep)"
  python3 /w/verify/compare_of.py $W1/gpu $t $W1/of $t /tmp/m6_adj_${t}.json
done

echo "=== Part 2: V7 full building physics + HVAC closed-loop (359,854 cells, 7 regions, 2 steps) ==="
W2=/w/cases/m6_v7full
mkdir -p $W2
for X in py gpu; do
  C=$W2/$X
  if [ "$X" = py ] && [ -d "$C/43200.04" ]; then continue; fi
  rm -rf $C; mkdir -p $C
  cp -r /py/v7/v7_full/constant /py/v7/v7_full/system /py/v7/v7_full/0 $C/
  cp $C/constant/building/hvac_design_runtime.py $C/constant/
  mv $C/0 $C/43200
  sed -i 's|/w/building|/py/building|g' $C/constant/gpuChtProperties
  sed -i 's/endTime 43800;/endTime 43200.04;/; s/deltaT 60;/deltaT 0.02;/; s/writeInterval 3600;/writeInterval 0.02;/' $C/system/controlDict
  sed -i 's/nOuterCorrectors 20;/nOuterCorrectors 1;/' $C/system/fvSolution
done
if [ ! -d "$W2/py/43200.04" ]; then
  (cd $W2/py && { time env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CHT_BACKEND=numpy python3 - <<'PY' > log.run 2>&1
import sys
sys.path.insert(0, "/py")
import cht.building.hvac as H
_orig_setup = H.FanCoils._setup
def _setup(self):
    zn = self.rr.mesh.zone_names or []
    zc = self.rr.mesh.zone
    for k, n in enumerate(zn):
        if n.startswith("room_") and int((zc == k).sum()) == 0:
            zn[k] = "_" + n
    units_orig = self.config["units"]
    if self.mapping.get("component_only", False):
        fids = {u["fcu_id"] for u in self.mapping["fcu_ports"]}
        self.config["units"] = [u for u in units_orig if u["fcu_id"] in fids]
    try:
        _orig_setup(self)
    finally:
        self.config["units"] = units_orig
        self.room_of = {u["fcu_id"]: u["room_id"] for u in units_orig}
H.FanCoils._setup = _setup
from cht.foam.app import main
sys.argv = ["/py/gpuChtFoam.py", "-case", "/w/cases/m6_v7full/py"]
sys.exit(main())
PY
  } 2> log.time)
fi
(cd $W2/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time)
echo "py  v7full steps=$(grep -c '^Time =' $W2/py/log.run), wall $(grep real $W2/py/log.time 2>/dev/null || true)"
echo "gpu v7full steps=$(grep -c '^Time =' $W2/gpu/log.run), $(grep -E '^ExecutionTime' $W2/gpu/log.run | tail -1), wall $(grep real $W2/gpu/log.time)"
grep -E "^building:|^solar:|^hvac:" $W2/py/log.run
echo "---"
grep -E "^building:|^solar:|^hvac:" $W2/gpu/log.run
for t in 43200.02 43200.04; do
  echo "== t=$t gpu vs py (M6 V7 full building physics + HVAC)"
  python3 /w/verify/compare_of.py $W2/gpu $t $W2/py $t /tmp/m6_v7full_${t}.json
done
python3 - <<'PY'
import json
a = json.load(open("/w/cases/m6_v7full/py/postProcessing/hvac/hvac.json"))
b = json.load(open("/w/cases/m6_v7full/gpu/postProcessing/hvac/hvac.json"))
for i, (ra, rb) in enumerate(zip(a, b)):
    ds = max(abs(ra["sensors"][k] - rb["sensors"][k]) for k in ra["sensors"])
    dm = abs(ra["supply_moist_kg_s"] - rb["supply_moist_kg_s"])
    dp = abs(ra["plant_W"] - rb["plant_W"])
    print(f"hvac.json step {i} (t={rb['t']}): |d_supply|={dm:.3e} kg/s, |d_plant|={dp:.3e} W, max|d_sensor|={ds:.3e} C")
PY
echo M6_DONE
