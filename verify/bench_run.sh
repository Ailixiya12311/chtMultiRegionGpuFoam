#!/bin/bash
# Time chtMultiRegionGpuFoam (BACKEND build) against OpenFOAM chtMultiRegionFoam (NP MPI ranks, scotch) on copies of
# one case (made by bench_case.sh), then compare the fields at the end time.
#   bench_run.sh CASE [NP]      (NP 0: skip OpenFOAM)
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-cuda}
S=$1; NP=${2:-8}
W=$(dirname $S)
G=$W/gpu_$B; O=$W/of_np$NP
rm -rf $G; mkdir -p $G; cp -r $S/0 $S/constant $S/system $G/
(cd $G && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time)
echo "[$B] rc=$? $(grep -c '^Time =' $G/log.run) steps, wall $(grep real $G/log.time)"
grep -E "FATAL" -A8 $G/log.run | head -20
grep -E "device memory" $G/log.run
grep -A12 "wall time by phase" $G/log.run
summ() {
  grep -E "Solving for" $1 | sed -E 's/.*:  Solving for ([A-Za-z_]+),.*No Iterations ([0-9]+)/\1 \2/' |
    awk '{s[$1]+=$2; n[$1]++; if($2>m[$1])m[$1]=$2} END {for(k in s) printf "    %-12s %4d solves, mean %7.1f, max %5d iterations\n", k, n[k], s[k]/n[k], m[k]}' | sort
}
echo "  linear solves ($B):"; summ $G/log.run
if [ "$NP" -gt 0 ]; then
  rm -rf $O; mkdir -p $O; cp -r $S/0 $S/constant $S/system $O/
  printf "FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }\nnumberOfSubdomains $NP;\nmethod scotch;\n" > $O/system/decomposeParDict
  for r in $(ls $O/constant | grep -vE "^(polyMesh|cellToRegion|g|regionProperties)$"); do
    [ -d $O/system/$r ] && cp $O/system/decomposeParDict $O/system/$r/
  done
  (cd $O && decomposePar -allRegions > log.decompose 2>&1 &&
   { time mpirun --allow-run-as-root --oversubscribe -np $NP chtMultiRegionFoam -parallel > log.run 2>&1; } 2> log.time &&
   reconstructPar -allRegions -latestTime > log.reconstruct 2>&1)
  echo "[OpenFOAM np $NP] $(grep -c '^Time =' $O/log.run) steps, wall $(grep real $O/log.time), $(grep -E '^ExecutionTime' $O/log.run | tail -1)"
  grep -E "FATAL|Floating point" $O/log.run | head -3
  echo "  linear solves (OpenFOAM):"; summ $O/log.run
  t=$(ls -d $G/[0-9]* | xargs -n1 basename | sort -g | tail -1)
  [ -d $O/$t ] && { echo "== t=$t $B vs OpenFOAM"; python3 /w/verify/compare_of.py $G $t $O $t /tmp/bench_cmp.json; }
fi
echo BENCH_DONE
