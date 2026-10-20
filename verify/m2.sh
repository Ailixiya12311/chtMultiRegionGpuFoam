#!/bin/bash
# M2: laminar fluid region + four solid walls (Py version's V5 conjugate cavity, Boussinesq air), transient step
# from the converged steady state: hot outer face 303.15 -> 308.15 K, deltaT 2 s, 60 s, nOuterCorrectors NOUTER,
# nCorrectors 2, U/h relaxation 0.7/0.9, p_rgh 0.7 (Final 1), linearUpwind U and h, limited corrected 0.5.
# OpenFOAM chtMultiRegionFoam and chtMultiRegionGpuFoam (BACKEND build) on copies of the same case; T, U, p_rgh,
# p, rho compared at 10 / 30 / 60 s. Also OpenFOAM with 60 outer correctors as the outer-iteration reference.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-omp}
NO=${NOUTER:-15}
END=${ENDT:-60}
W=/w/cases/m2_n${NO}_t${TIGHT:-0}
REF=/py/cases/cht_frame
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
rm -rf $W; mkdir -p $W
mk() {
  C=$W/$1; mkdir -p $C; cp -r $REF/0 $REF/constant $REF/system $C/
  python3 /py/verify/make_transient_v5.py $C $END 10 $2 > /dev/null
  rm -rf $C/0; cp -r $REF/4000 $C/; python3 /py/verify/make_step_v5.py $C > /dev/null
  # TIGHT=1: every linear solve to relTol 0 (isolates the algorithm from inner-solver convergence within relTol)
  if [ "${TIGHT:-0}" = 1 ]; then
    sed -i -e 's/tolerance 1e-10; relTol 0.01;/tolerance 1e-11; relTol 0;/' -e 's/tolerance 1e-12; relTol 0.1;/tolerance 1e-13; relTol 0;/' \
           -e 's/tolerance 1e-12; relTol 0;/tolerance 1e-13; relTol 0;/' -e 's/tolerance 1e-10; relTol 0;/tolerance 1e-11; relTol 0;/' \
           $C/system/*/fvSolution
  fi
}
mk of $NO; mk gpu $NO; mk of60 60
(cd $W/of && chtMultiRegionFoam > log.run 2>&1; echo "OpenFOAM rc=$? $(grep -c '^Time =' log.run) steps") &
(cd $W/of60 && chtMultiRegionFoam > log.run 2>&1; echo "OpenFOAM(60) rc=$? $(grep -c '^Time =' log.run) steps") &
(cd $W/gpu && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time; echo "gpu rc=$? $(grep -c '^Time =' log.run) steps, wall $(grep real log.time)") &
wait
grep -E "FOAM FATAL|gpucht FATAL" -A12 $W/of/log.run $W/gpu/log.run $W/gpu/log.time | head -40
grep -E "^ExecutionTime" $W/of/log.run | tail -1; grep -E "^ExecutionTime" $W/gpu/log.run | tail -1
echo "--- gpu log, last step"; awk '/^Time = /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $W/gpu/log.run | grep -vE "^\s*$" | tail -60
for t in 10 30 $END; do
  echo "== t=$t gpu vs of"; python3 /w/verify/compare_of.py $W/gpu $t $W/of $t /tmp/m2.json
  echo "== t=$t of vs of60"; python3 /w/verify/compare_of.py $W/of $t $W/of60 $t /tmp/m2b.json | grep -E "air/(T|U) |solidL/T "
done
echo M2_DONE
