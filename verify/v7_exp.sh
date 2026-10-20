#!/bin/bash
# V7 start-up experiments with OpenFOAM chtMultiRegionFoam only (the case question, not the solver): v7_std with
# one opening added, 20 outer correctors, one variant per run, all in parallel.
#   v7_exp.sh NSTEPS VARIANT...
# VARIANT = OPENING[+lam][+upw][+dt5]
#   OPENING  closed     as v7_std: every port has a fixed mass flow, no pressure boundary (Boussinesq rho(T) in a
#                       closed box: the pressure equation has no solution once the mean density changes)
#            ra_total   return grille -> prghTotalPressure / pressureInletOutletVelocity (like tower_louver_opening)
#            ra_static  return grille -> prghPressure (static) / pressureInletOutletVelocity
#            side       return grille keeps its fixed flow; domain_side (the face where the cut-out was clipped from
#                       the building, a wall in v7_std) -> prghPressure / pressureInletOutletVelocity
#                       (uniform static p on a 4 m tall face: not hydrostatic, drives a spurious rho g dz flow)
#            side_tp    as side, but p_rgh totalPressure p0 uniform: the outside is hydrostatic (p_rgh uniform)
#   +lam laminar  +upw div(phi,U), div(phi,h) upwind  +dt5 deltaT 0.005 s (4x the steps)
#   +nodpdt  dpdt no; in the air thermophysicalProperties (no pressure work in the enthalpy equation)
# Prints per step the air T range and continuity error of the last outer corrector, and how far each run got.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -u
NS=$1; shift
export LD_LIBRARY_PATH=/py/ofapps/lib:${LD_LIBRARY_PATH:-}
W=/w/cases/v7_exp
mkdir -p $W
run() {
  V=$1; C=$W/$V; DT=0.02; N=$NS
  case $V in *+dt5*) DT=0.005; N=$((NS*4)) ;; esac
  rm -rf $C; mkdir -p $C
  cp -r /py/v7/v7_std/constant /py/v7/v7_std/system /py/v7/v7_std/0 $C/
  mv $C/0 $C/43200
  END=$(python3 -c "print(round(43200 + $N*$DT, 6))")
  sed -i "s/endTime 43202;/endTime $END;/; s/deltaT 0.02;/deltaT $DT;/; s/writeInterval 3600;/writeInterval 0.02;/" $C/system/controlDict
  python3 - $C/43200/air ${V%%+*} <<'PY'
import re, sys
d, opening = sys.argv[1], sys.argv[2]
def internal(f):
    return re.search(r"internalField\s+uniform\s+([^;]+);", open(f"{d}/{f}").read()).group(1).strip()
p_rgh = {"ra_total": "type prghTotalPressure; p0 uniform 100090; value uniform 100090;",
         "ra_static": "type prghPressure; p uniform 100090; value uniform 100090;",
         "side": "type prghPressure; p uniform 100090; value uniform 100090;",
         "side_tp": "type totalPressure; p0 uniform 100090; value uniform 100090;"}
patch = {"ra_total": "hvac_HVAC_2S5_0_ra", "ra_static": "hvac_HVAC_2S5_0_ra", "side": "domain_side",
         "side_tp": "domain_side"}
if opening == "closed":
    sys.exit(0)
new = {
    "U": "type pressureInletOutletVelocity; value uniform (0 0 0);",
    "p_rgh": p_rgh[opening],
    "T": f"type inletOutlet; inletValue uniform {internal('T')}; value uniform {internal('T')};",
    "k": "type inletOutlet; inletValue uniform 0.0001; value uniform 0.0001;",
    "epsilon": "type inletOutlet; inletValue uniform 1e-05; value uniform 1e-05;",
    "waterVapour": f"type inletOutlet; inletValue uniform {internal('waterVapour')}; value uniform {internal('waterVapour')};",
    "dryAir": f"type inletOutlet; inletValue uniform {internal('dryAir')}; value uniform {internal('dryAir')};",
}
pn = patch[opening]
if pn == "domain_side":
    # an opening, not a wall: patch type patch, no wall functions (OpenFOAM would otherwise keep nut/alphat wall
    # functions there with an inletOutlet epsilon; chtMultiRegionGpuFoam rejects that mix)
    new["nut"] = "type calculated; value uniform 0;"
    new["alphat"] = "type calculated; value uniform 0;"
    b = d.replace("/43200/air", "/constant/air/polyMesh/boundary")
    s, n = re.subn(r"(\n\s+domain_side\s*\{\s*type\s+)wall;\s*inGroups\s+1\(wall\);", r"\1patch;", open(b).read())
    if n != 1:
        sys.exit(f"boundary: {n} matches for domain_side")
    open(b, "w").write(s)
for f, e in new.items():
    p = f"{d}/{f}"
    s, n = re.subn(r"(\n\s+" + pn + r" )\{[^}\n]*\}", lambda m: m.group(1) + "{ " + e + " }", open(p).read())
    if n != 1:
        sys.exit(f"{f}: {n} matches for {pn}")
    open(p, "w").write(s)
PY
  [ $? = 0 ] || { echo "== $V: case edit failed" > $C/summary.txt; return; }
  case $V in *+lam*) sed -i 's/simulationType RAS;/simulationType laminar;/' $C/constant/air/turbulenceProperties ;; esac
  case $V in *+nodpdt*) sed -i 's/^thermoType/dpdt no;\nthermoType/' $C/constant/air/thermophysicalProperties ;; esac
  case $V in *+upw*) sed -i 's/div(phi,U) Gauss linearUpwind grad(U);/div(phi,U) Gauss upwind;/; s/div(phi,h) Gauss linearUpwind grad(h);/div(phi,h) Gauss upwind;/' \
                       $C/system/air/fvSchemes ;; esac
  (cd $C && chtMultiRegionFoam > log.run 2>&1)
  {
    echo "== $V (deltaT $DT): $(grep -c '^Time =' $C/log.run) of $N steps started, $(grep -v trapFpe $C/log.run | grep -m1 -oE 'FOAM FATAL ERROR|Floating point exception|^End$')"
    awk '/^Time = /{if(t!="")print t"  "T"  "c; t=$3} /^Min\/max T/{if(!s){T=$0; s=1}} /^Solving for fluid region/{s=0} /continuity errors/{c=$0} END{print t"  "T"  "c}' $C/log.run |
      sed -E 's/time step continuity errors \(air\): //' | cut -c1-160
  } > $C/summary.txt
}
for V in "$@"; do run $V & done
wait
for V in "$@"; do cat $W/$V/summary.txt; done
echo V7_EXP_DONE
