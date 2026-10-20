#!/bin/bash
# chtMultiRegionGpuFoam (BACKEND build) on a V7 variant that v7_exp.sh already ran with OpenFOAM (same case files,
# OpenFOAM's results written every 0.02 s), compared with OpenFOAM at every written time.
#   v7_cmp.sh VARIANT [EVERY]      EVERY: compare every n-th written time (default 1)
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-cuda}
V=$1; EV=${2:-1}
export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}
O=/w/cases/v7_exp/$V
G=/w/cases/v7_exp/${V}_gpu
[ -d $O/43200 ] || { echo "v7_cmp: run v7_exp.sh for $V first"; exit 1; }
rm -rf $G; mkdir -p $G
cp -r $O/constant $O/system $O/43200 $G/
(cd $G && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time)
echo "[$B] $(grep -c '^Time =' $G/log.run) steps, wall $(grep real $G/log.time), $(grep -m1 -E 'gpucht FATAL|FOAM FATAL|^End$' $G/log.run)"
grep -A12 "wall time by phase" $G/log.run | head -12
echo "[OpenFOAM] $(grep -c '^Time =' $O/log.run) steps, $(grep -E '^ExecutionTime' $O/log.run | tail -1)"
i=0
for t in $(ls -d $O/432* | xargs -n1 basename | sort -g | tail -n +2); do
  i=$((i + 1))
  [ $((i % EV)) = 0 ] || [ $t = $(ls -d $O/432* | xargs -n1 basename | sort -g | tail -1) ] || continue
  [ -d $G/$t ] || { echo "== t=$t: no gpu result"; continue; }
  echo "== t=$t $B vs OpenFOAM ($V)"
  python3 /w/verify/compare_of.py $G $t $O $t /tmp/v7cmp.json | grep -E "air/(T|U|p_rgh|k|nut|waterVapour) |glazing|wall_substrate"
done
echo V7_CMP_DONE
