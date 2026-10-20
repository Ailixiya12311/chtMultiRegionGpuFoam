#!/bin/bash
# M4: Turbulent conjugate heat transfer (RNG k-epsilon + kqRWallFunction + epsilonWallFunction + nutkWallFunction
# + compressible::alphatJayatillekeWallFunction on fluid-solid interfaces) on the 30x-scaled V5 active cavity
# (3 m cavity, hRef = 0.61368608, deltaT = 0.05 s, 20 outer correctors, endTime = 2 s, 40 time steps).
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-omp}
NO=${NOUTER:-20}
END=${ENDT:-2}
DT=${DELTAT:-0.05}
W=/w/cases/m4_n${NO}_t${TIGHT:-0}
REF=/py/cases/cht_frame
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export LD_LIBRARY_PATH=/w/build/$B:${LD_LIBRARY_PATH:-}
[ "${CLEAN:-0}" = 1 ] && rm -rf $W
mkdir -p $W

HREF='FoamFile { version 2.0; format ascii; class uniformDimensionedScalarField; location "constant/air"; object hRef; }
dimensions [0 1 0 0 0 0 0];
value 0.61368608;'

for X in of gpu; do
  C=$W/$X
  if [ "$X" = of ] && [ -d "$C/$END" ]; then continue; fi
  if [ "$X" = of ] && [ "${TIGHT:-0}" = 0 ] && [ "$NO" = 20 ] && [ "$END" = 2 ] && [ "$DT" = 0.05 ] && [ -d "/py/compat/v5kr_of/2" ]; then
    rm -rf $C; cp -r /py/compat/v5kr_of $C
    continue
  fi
  rm -rf $C; mkdir -p $C
  if [ -d "/py/compat/v5kp_of/0" ]; then
    cp -r /py/compat/v5kp_of/0 /py/compat/v5kp_of/constant /py/compat/v5kp_of/system $C/
    sed -i "s/endTime [0-9.]*;/endTime $END;/; s/deltaT [0-9.]*;/deltaT $DT;/; s/writeInterval [0-9.]*;/writeInterval 1;/" $C/system/controlDict
    sed -i "s/nOuterCorrectors [0-9]*;/nOuterCorrectors $NO;/" $C/system/fvSolution
  else
    cp -r $REF/0 $REF/constant $REF/system $C/
    (cd $C && for r in air solidL solidR solidB solidT; do transformPoints -region $r -scale 30 > /dev/null || exit 1; done)
    python3 /py/verify/make_turb_v5.py $C $END 1 $NO active > /dev/null
    echo "$HREF" > $C/constant/air/hRef
    sed -i "s/deltaT [0-9.]*;/deltaT $DT;/" $C/system/controlDict
  fi
  if [ "${TIGHT:-0}" = 1 ]; then
    sed -i -e 's/tolerance 1e-10; relTol 0.01;/tolerance 1e-11; relTol 0;/' -e 's/tolerance 1e-12; relTol 0.1;/tolerance 1e-13; relTol 0;/' \
           -e 's/tolerance 1e-12; relTol 0;/tolerance 1e-13; relTol 0;/' -e 's/tolerance 1e-10; relTol 0;/tolerance 1e-11; relTol 0;/' \
           $C/system/*/fvSolution
  fi
done

if [ ! -d "$W/of/$END" ]; then
  (cd $W/of && chtMultiRegionFoam > log.run 2>&1) &
fi
(cd $W/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time) &
wait

echo "of  rc=$(grep -c '^Time =' $W/of/log.run) steps, $(grep -E '^ExecutionTime' $W/of/log.run | tail -1)"
echo "gpu rc=$(grep -c '^Time =' $W/gpu/log.run) steps, $(grep -E '^ExecutionTime' $W/gpu/log.run | tail -1), wall $(grep real $W/gpu/log.time)"
grep -E "FOAM FATAL|gpucht FATAL" -A10 $W/of/log.run $W/gpu/log.run $W/gpu/log.time | head -25
for t in 1 $END; do
  echo "== t=$t gpu vs of (M4 turbulent V5)"
  python3 /w/verify/compare_of.py $W/gpu $t $W/of $t /tmp/m4_${t}.json
done
if [ -f "/py/verify/cmp_patchvals.py" ]; then
  echo "== patch values at t=$END (air/alphat)"
  python3 /py/verify/cmp_patchvals.py $W/gpu $END $W/of $END air alphat
fi
echo M4_DONE
