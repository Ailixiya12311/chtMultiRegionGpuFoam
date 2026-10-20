#!/bin/bash
# A/B regression: run the same verification cases with two builds of chtMultiRegionGpuFoam and compare all fields at
# the end time (compare_of.py). A change that keeps the arithmetic must give zero differences.
#   ab.sh [A [B [CASE ...]]]      A, B: build directories under /w/build (default cuda_base, cuda)
#   CASE: <case dir under /w/cases>:<start time>, default the M1-M6 cases and the bench_k2 frame
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
A=${1:-cuda_base}; B=${2:-cuda}
[ $# -ge 2 ] && shift 2 || shift $#
CASES=${*:-"m1_n30/gpu:0 m3_n15_t0/full_gpu:0 m4_n20_t0/gpu:0 m5_ftc60/gpu:0 m6_adj/gpu:0 m6_v7full/gpu:43200 bench_k2/cht_frame:0"}
export LD_LIBRARY_PATH=/py/ofapps/lib:${LD_LIBRARY_PATH:-}
for c in $CASES; do
  S=/w/cases/${c%%:*}; T0=${c##*:}
  W=/w/cases/ab/$(echo ${c%%:*} | tr / _)
  for X in $A $B; do
    R=$W/$X
    rm -rf $R; mkdir -p $R
    cp -r $S/constant $S/system $S/$T0 $R/
    (cd $R && /w/build/$X/chtMultiRegionGpuFoam > log.run 2>&1)
  done
  t=$(ls -d $W/$A/[0-9]* | xargs -n1 basename | sort -g | tail -1)
  wa=$(grep -oE "wall time by phase \([0-9]+ steps, [0-9.e+-]+ s" $W/$A/log.run | grep -oE "[0-9.e+-]+ s$")
  wb=$(grep -oE "wall time by phase \([0-9]+ steps, [0-9.e+-]+ s" $W/$B/log.run | grep -oE "[0-9.e+-]+ s$")
  echo "== ${c%%:*}  t=$t  time loop $A ${wa:-FAILED}  $B ${wb:-FAILED}"
  grep -m2 -E "FATAL|sigFpe" $W/$A/log.run $W/$B/log.run
  [ -d $W/$B/$t ] && python3 /w/verify/compare_of.py $W/$B $t $W/$A $t /tmp/ab.json 2>&1 | awk '{print "   " $0}' | head -24
done
echo AB_DONE
