#!/bin/bash
# Time-step scan on a V7 variant made by v7_exp.sh (its case files, e.g. side_tp), chtMultiRegionGpuFoam only:
# NSTEPS steps of DT seconds each, per step the air Courant number, T range, largest |U| and the outer correctors used.
#   v7_dt.sh VARIANT DT NSTEPS [TAG] [EDIT...]
# EDIT (sed expressions applied in order to system/air/fvSolution, or to system/air/fvSchemes when prefixed S:), e.g.
#   's/".\*Final" 1;/".*Final" 0.7;/'      relax the final outer corrector too
#   'S:s/Gauss/bounded Gauss/'             bounded convection
# TAG names the run directory (default dt<DT>). APP=of runs OpenFOAM's chtMultiRegionFoam instead (directory suffix _of).
# NOUTER=n overrides nOuterCorrectors. HYDRO=rho writes a hydrostatic initial p with that density (verify/hydro_p.py).
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
B=${BACKEND:-cuda}
V=$1; DT=$2; NS=$3; TAG=${4:-dt$DT}
shift 3; [ $# -gt 0 ] && shift
export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}
O=/w/cases/v7_exp/$V
APP=${APP:-gpu}
G=/w/cases/v7_dt/${V}_$TAG$([ $APP = of ] && echo _of)
EXE=$([ $APP = of ] && echo chtMultiRegionFoam || echo /w/build/$B/chtMultiRegionGpuFoam)
[ -d $O/43200 ] || { echo "v7_dt: run v7_exp.sh for $V first"; exit 1; }
rm -rf $G; mkdir -p $G
cp -r $O/constant $O/system $O/43200 $G/
END=$(python3 -c "print(round(43200 + $NS*$DT, 6))")
WI=$DT
sed -i -E "s/endTime [0-9.e+-]+;/endTime $END;/; s/deltaT [0-9.e+-]+;/deltaT $DT;/; s/writeInterval [0-9.e+-]+;/writeInterval $WI;/" \
  $G/system/controlDict
[ -n "${HYDRO:-}" ] && python3 /w/verify/hydro_p.py $G/43200 air 100090 $HYDRO
[ -n "${NOUTER:-}" ] && sed -i "s/nOuterCorrectors [0-9]*;/nOuterCorrectors $NOUTER;/" $G/system/fvSolution
for e in "$@"; do
  case $e in S:*) sed -i "${e#S:}" $G/system/air/fvSchemes ;; *) sed -i "$e" $G/system/air/fvSolution ;; esac
done
grep -E "relaxationFactors|PIMPLE" $G/system/air/fvSolution | cut -c1-200
grep -E "div\(phi" $G/system/air/fvSchemes | tr -s ' ' | tr '\n' ' '; echo
(cd $G && { time $EXE > log.run 2>&1; } 2> log.time)
echo "[$([ $APP = of ] && echo OpenFOAM || echo $B) $V $TAG] $(grep -c '^Time =' $G/log.run) of $NS steps, wall $(grep real $G/log.time | awk '{print $2}'), \
$(grep -m1 -oE 'gpucht FATAL.*|FOAM FATAL.*|^End$' $G/log.run)"
# per step: max Co, air T range of the last outer corrector, continuity, number of outer correctors
awk '/^Time = /{if(t!="")printf "%-12s Co %-10.4g %-44s outer %3d  %s\n", t, co, T, no, c; t=$3; no=0; s=0}
     /^Region: air Courant/{co=$NF}
     /^Solving for fluid region air/{no++; s=1}
     /^Min\/max T/{if(s){T=$0; s=0}}
     /continuity errors \(air\)/{c=$0}
     END{printf "%-12s Co %-10.4g %-44s outer %3d  %s\n", t, co, T, no, c}' $G/log.run |
  sed -E 's/time step continuity errors \(air\): //; s/Min\/max T://' | cut -c1-170
grep -A12 "wall time by phase" $G/log.run | head -11
echo V7_DT_DONE
