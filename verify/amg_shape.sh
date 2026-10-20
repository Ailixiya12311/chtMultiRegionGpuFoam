#!/bin/bash
# V-cycle shape scan on the bench_k4 frame (verify/bench_case.sh): GPUCHT_AMG_PRE / POST / MR variants, the time
# loop, the pressure phase and the p_rgh iteration counts of each.
#   amg_shape.sh "PRE POST MR" ...        e.g. amg_shape.sh "1 2 1" "1 1 1" "1 2 0" "1 1 0"
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true
F=${FRAME:-/w/cases/bench_k4/cht_frame}
for v in "$@"; do
  set -- $v
  G=/w/cases/amg_shape/p$1_q$2_m$3; rm -rf $G; mkdir -p $G; cp -r $F/0 $F/constant $F/system $G/
  (cd $G && GPUCHT_AMG_PRE=$1 GPUCHT_AMG_POST=$2 GPUCHT_AMG_MR=$3 /w/build/${BACKEND:-cuda}/chtMultiRegionGpuFoam > log.run 2>&1)
  it=$(grep -E "Solving for p_rgh" $G/log.run | sed -E "s/.*No Iterations ([0-9]+)/\1/" | awk '{s+=$1} END {print s}')
  echo "pre $1 post $2 mr $3: $(grep -oE "wall time by phase \([0-9]+ steps, [0-9.]+ s" $G/log.run), pressure $(grep -A9 "wall time by phase" $G/log.run | grep pressure | awk '{print $2}') s, p_rgh iterations $it"
  grep -m2 -E "FATAL|sigFpe" $G/log.run
done
