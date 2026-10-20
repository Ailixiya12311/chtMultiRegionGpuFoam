#!/bin/bash
# M1: two solid regions on a skewed block (non-orthogonal faces, jagged interface), transient conduction.
#   left: uniformFixedValue table 300 -> 350 K; right: externalWallHeatFluxTemperature (h 10, Ta 280, emissivity 0.9)
#   grad(h) cellLimited leastSquares 1, laplacian limited corrected 0.5, h relaxation 0.8, nOuterCorrectors NOUTER
# OpenFOAM chtMultiRegionFoam and chtMultiRegionGpuFoam (BACKEND build) on copies of the same case; T compared.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u (bashrc uses unset variables)
set -u
B=${BACKEND:-omp}
NO=${NOUTER:-30}
W=/w/cases/m1_n$NO
rm -rf $W; mkdir -p $W/base/system $W/base/constant
cd $W/base
hdr() { printf 'FoamFile { version 2.0; format ascii; class %s; object %s; }\n' $1 $2; }
{ hdr dictionary controlDict; cat <<'E'; } > system/controlDict
application chtMultiRegionFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 50; deltaT 5;
writeControl timeStep; writeInterval 2; purgeWrite 0;
writeFormat ascii; writePrecision 14; writeCompression off; timeFormat general; timePrecision 6;
runTimeModifiable false;
E
{ hdr dictionary blockMeshDict; cat <<'E'; } > system/blockMeshDict
vertices ( (0 0 0) (1 0 0) (1.2 0.5 0) (0.2 0.5 0) (0 0 0.2) (1 0 0.2) (1.2 0.5 0.2) (0.2 0.5 0.2) );
blocks ( hex (0 1 2 3 4 5 6 7) (30 16 3) simpleGrading (1 1 1) );
boundary
(
    left { type wall; faces ((0 4 7 3)); }
    right { type wall; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }
);
E
{ hdr dictionary topoSetDict; cat <<'E'; } > system/topoSetDict
actions
(
    { name solidA; type cellZoneSet; action new; source boxToCell; box (-1 -1 -1) (0.62 1 1); }
    { name solidB; type cellZoneSet; action new; source boxToCell; box (-1 -1 -1) (0.62 1 1); }
    { name solidB; type cellZoneSet; action invert; }
);
E
{ hdr dictionary fvSchemes; echo 'ddtSchemes { default Euler; } gradSchemes { default leastSquares; } divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }'; } > system/fvSchemes
{ hdr dictionary fvSolution; echo "PIMPLE { nOuterCorrectors $NO; }"; } > system/fvSolution
{ hdr dictionary regionProperties; echo 'regions ( fluid () solid (solidA solidB) );'; } > constant/regionProperties
{ hdr uniformDimensionedVectorField g; echo 'dimensions [0 1 -2 0 0 0 0]; value (0 0 -9.81);'; } > constant/g
blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 && splitMeshRegions -cellZones -overwrite > log.split 2>&1 || { echo "mesh FAILED"; tail -20 log.*; exit 1; }
for r in solidA solidB; do
  o=$([ $r = solidA ] && echo solidB || echo solidA)
  mkdir -p system/$r 0/$r
  { hdr dictionary fvSchemes; echo 'ddtSchemes { default Euler; } gradSchemes { default leastSquares; grad(h) cellLimited leastSquares 1; } divSchemes { default none; } laplacianSchemes { default Gauss linear limited corrected 0.5; } interpolationSchemes { default linear; } snGradSchemes { default limited corrected 0.5; }'; } > system/$r/fvSchemes
  { hdr dictionary fvSolution; echo 'solvers { h { solver PCG; preconditioner DIC; tolerance 1e-13; relTol 0; } hFinal { $h; } } relaxationFactors { equations { h 0.8; } } PIMPLE { nNonOrthogonalCorrectors 0; }'; } > system/$r/fvSolution
  if [ $r = solidA ]; then RHO=2000; CP=900; K=1.4; else RHO=800; CP=1500; K=0.2; fi
  { hdr dictionary thermophysicalProperties; cat <<E; } > constant/$r/thermophysicalProperties
thermoType { type heSolidThermo; mixture pureMixture; transport constIso; thermo hConst; equationOfState rhoConst; specie specie; energy sensibleEnthalpy; }
mixture { specie { molWeight 50; } transport { kappa $K; } thermodynamics { Hf 0; Cp $CP; } equationOfState { rho $RHO; } }
E
  { hdr volScalarField T; cat <<E; } > 0/$r/T
dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    left { type uniformFixedValue; uniformValue table ((0 300) (50 350)); value uniform 300; }
    right { type externalWallHeatFluxTemperature; kappaMethod solidThermo; mode coefficient; h uniform 10; Ta constant 280; emissivity 0.9; value uniform 300; }
    walls { type zeroGradient; }
    ${r}_to_${o} { type compressible::turbulentTemperatureCoupledBaffleMixed; Tnbr T; kappaMethod solidThermo; value uniform 300; }
}
E
  { hdr volScalarField p; cat <<E; } > 0/$r/p
dimensions [1 -1 -2 0 0 0 0];
internalField uniform 1e5;
boundaryField { ".*" { type calculated; value uniform 1e5; } }
E
done
rm -rf 0/cellToRegion constant/polyMesh 2>/dev/null
cp -r $W/base $W/of; cp -r $W/base $W/gpu
(cd $W/of && chtMultiRegionFoam > log.run 2>&1); echo "OpenFOAM rc=$? $(grep -c 'Time =' $W/of/log.run) steps"
(cd $W/gpu && /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1); echo "gpu rc=$? $(grep -c '^Time =' $W/gpu/log.run) steps"
grep -E "FOAM FATAL|gpucht FATAL" -A8 $W/of/log.run $W/gpu/log.run | head -30
tail -12 $W/gpu/log.run
for t in 10 30 50; do echo "== t=$t"; python3 /w/verify/compare_of.py $W/gpu $t $W/of $t /tmp/m1.json | grep -E "/T"; done
echo M1_DONE
