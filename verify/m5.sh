#!/bin/bash
# M5: Building boundary conditions and fvOptions:
#   Part 1 (ftc): V5 step-change conjugate cavity with fixedTemperatureConstraint (cellZone lab held at 300.15 K)
#   Part 2 (v7) : V7 cut-out of Room 2S5 from whole-building r3d mesh (359,854 cells, 7 regions, FCU flowRateInlet/
#                 OutletVelocity, externalWallHeatFluxTemperature, scalarSemiImplicitSource on h and waterVapour)
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-omp}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}

echo "=== Part 1: V5 fixedTemperatureConstraint (60 outer correctors) ==="
W1=/w/cases/m5_ftc60
mkdir -p $W1
for X in of gpu; do
  C=$W1/$X
  if [ "$X" = of ] && [ -d "/py/compat/v5ftc60_of/60" ]; then
    rm -rf $C; cp -r /py/compat/v5ftc60_of $C
    continue
  fi
  rm -rf $C; mkdir -p $C
  cp -r /py/compat/v5ftc60_of/0 /py/compat/v5ftc60_of/constant /py/compat/v5ftc60_of/system $C/
done
if [ ! -d "$W1/of/60" ]; then
  (cd $W1/of && chtMultiRegionFoam > log.run 2>&1)
fi
(cd $W1/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time)
echo "gpu ftc60 steps=$(grep -c '^Time =' $W1/gpu/log.run), $(grep -E '^ExecutionTime' $W1/gpu/log.run | tail -1), wall $(grep real $W1/gpu/log.time)"
for t in 10 60; do
  echo "== t=$t gpu vs of (M5 V5 fixedTemperatureConstraint, 60 outer)"
  python3 /w/verify/compare_of.py $W1/gpu $t $W1/of $t /tmp/m5_ftc60_${t}.json
done
python3 - <<'PY'
import os, sys, numpy as np
sys.path.insert(0, "/py")
from cht.mesh import read_cellzones
from cht.foam.dictio import read_dict
from cht.foam import bcs as B
for X in ("of", "gpu"):
    c = "/w/cases/m5_ftc60/" + X
    z = read_cellzones(os.path.join(c, "constant", "air", "polyMesh", "cellZones"))["lab"]
    d = read_dict(os.path.join(c, "60", "air", "T"))
    T = np.asarray(B.field_value(d.tokens("internalField"), 3600, 1, "T"))
    print("%s: %d zone cells, zone T at 60 s %.9f..%.9f, outside %.4f..%.4f" % (
        X, len(z), T[z].min(), T[z].max(), np.delete(T, z).min(), np.delete(T, z).max()))
PY

echo "=== Part 2: V7 cut-out single step (359,854 cells, 7 regions) ==="
W2=/w/cases/m5_v7
mkdir -p $W2
for X in of gpu; do
  C=$W2/$X
  rm -rf $C; mkdir -p $C
  cp -r /py/v7/v7_std/constant /py/v7/v7_std/system /py/v7/v7_std/0 $C/
  mv $C/0 $C/43200
  sed -i 's/endTime 43202;/endTime 43200.02;/; s/writeInterval 3600;/writeInterval 0.02;/' $C/system/controlDict
  sed -i 's/nOuterCorrectors 20;/nOuterCorrectors 1;/' $C/system/fvSolution
done
(cd $W2/of && { time chtMultiRegionFoam > log.run 2>&1; } 2> log.time)
(cd $W2/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time)
echo "of  v7 steps=$(grep -c '^Time =' $W2/of/log.run), $(grep -E '^ExecutionTime' $W2/of/log.run | tail -1), wall $(grep real $W2/of/log.time)"
echo "gpu v7 steps=$(grep -c '^Time =' $W2/gpu/log.run), $(grep -E '^ExecutionTime' $W2/gpu/log.run | tail -1), wall $(grep real $W2/gpu/log.time)"
echo "== t=43200.02 gpu vs of (M5 V7 359,854-cell 7-region cut-out)"
python3 /w/verify/compare_of.py $W2/gpu 43200.02 $W2/of 43200.02 /tmp/m5_v7.json
echo M5_DONE
