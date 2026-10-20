#!/bin/bash
# M3: two-component humid air (multiComponentMixture, dryAir + waterVapour, Boussinesq) on the V5 step-change
# conjugate cavity (60 s, deltaT 2 s, 15 outer correctors, div(phi,Yi_h) Gauss limitedLinear 1) in three variants:
#   pas  : waterVapour Cp 1006, Href 0       (passive scalar: T, U match pure-air M2)
#   cp   : waterVapour Cp 1860, Href 0       (variable mixture Cp)
#   full : waterVapour Cp 1860, Href 2501000 (variable mixture Cp + reference enthalpy)
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-omp}
NO=${NOUTER:-15}
END=${ENDT:-60}
TREF=${TREF:-298.15}
W=/w/cases/m3_n${NO}_t${TIGHT:-0}
REF=/py/cases/cht_frame
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}
[ "${CLEAN:-0}" = 1 ] && rm -rf $W
mkdir -p $W

run_var() {
  V=$1; CP=$2; HR=$3
  for X in of gpu; do
    C=$W/${V}_$X
    if [ "$X" = of ] && [ -d "$C/$END" ]; then continue; fi
    rm -rf $C; mkdir -p $C; cp -r $REF/0 $REF/constant $REF/system $C/
    python3 /py/verify/make_transient_v5.py $C $END 10 $NO > /dev/null
    rm -rf $C/0; cp -r $REF/4000 $C/
    (cd /py/verify && python3 make_step_v5.py $C > /dev/null && python3 make_species_v5.py $C $CP $HR $TREF > /dev/null)
    if [ "${TIGHT:-0}" = 1 ]; then
      sed -i -e 's/tolerance 1e-10; relTol 0.01;/tolerance 1e-11; relTol 0;/' -e 's/tolerance 1e-12; relTol 0.1;/tolerance 1e-13; relTol 0;/' \
             -e 's/tolerance 1e-12; relTol 0;/tolerance 1e-13; relTol 0;/' -e 's/tolerance 1e-10; relTol 0;/tolerance 1e-11; relTol 0;/' \
             $C/system/*/fvSolution
    fi
  done
  if [ ! -d "$W/${V}_of/$END" ]; then
    (cd $W/${V}_of && chtMultiRegionFoam > log.run 2>&1) &
  fi
  (cd $W/${V}_gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time) &
  wait
  {
    echo "######## variant $V (Cp_v $CP, Href_v $HR, Tref $TREF)"
    echo "of  rc=$(grep -c '^Time =' $W/${V}_of/log.run) steps, $(grep -E '^ExecutionTime' $W/${V}_of/log.run | tail -1)"
    echo "gpu rc=$(grep -c '^Time =' $W/${V}_gpu/log.run) steps, $(grep -E '^ExecutionTime' $W/${V}_gpu/log.run | tail -1), wall $(grep real $W/${V}_gpu/log.time)"
    grep -E "FOAM FATAL|gpucht FATAL" -A10 $W/${V}_of/log.run $W/${V}_gpu/log.run $W/${V}_gpu/log.time | head -25
    for t in 10 30 $END; do
      echo "== t=$t gpu vs of ($V)"
      python3 /w/verify/compare_of.py $W/${V}_gpu $t $W/${V}_of $t /tmp/m3_${V}_${t}.json
    done
  } > $W/report_${V}.txt 2>&1
}

run_var pas 1006 0 &
run_var cp 1860 0 &
run_var full 1860 2501000 &
wait
cat $W/report_pas.txt $W/report_cp.txt $W/report_full.txt
echo M3_DONE
