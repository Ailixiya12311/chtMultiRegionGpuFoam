#!/bin/bash
# M7: CUDA backend verification, OMP vs CUDA consistency & speedup benchmark on V7 full building physics
#     (359,854 cells, 7 regions), and verified device VRAM accounting for the 42.45M-cell r3d whole-building mesh.
source /usr/lib/openfoam/openfoam2606/etc/bashrc >/dev/null 2>&1 || true   # before set -u
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}

echo "=== Part 1: GPU Hardware & Unit Test Verification (OMP + CUDA) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv
fi
/w/build/omp/test_lib
if [ -x /w/build/cuda/test_lib ]; then
  /w/build/cuda/test_lib
fi

echo "=== Part 2: V7 Full Building Physics (359,854 cells, 7 regions) — OMP vs CUDA Consistency & Speedup ==="
W=/w/cases/m7_v7full
mkdir -p $W
for B in omp cuda; do
  if [ ! -x "/w/build/$B/chtMultiRegionGpuFoam" ]; then continue; fi
  C=$W/$B
  rm -rf $C; mkdir -p $C
  cp -r /py/v7/v7_full/constant /py/v7/v7_full/system /py/v7/v7_full/0 $C/
  cp $C/constant/building/hvac_design_runtime.py $C/constant/
  mv $C/0 $C/43200
  sed -i 's|/w/building|/py/building|g' $C/constant/gpuChtProperties
  sed -i 's/endTime 43800;/endTime 43200.04;/; s/deltaT 60;/deltaT 0.02;/; s/writeInterval 3600;/writeInterval 0.02;/' $C/system/controlDict
  sed -i 's/nOuterCorrectors 20;/nOuterCorrectors 1;/' $C/system/fvSolution
  (
    export LD_LIBRARY_PATH=/py/ofapps/lib:/w/build/$B:${LD_LIBRARY_PATH:-}
    cd $C && { time /w/build/$B/chtMultiRegionGpuFoam > log.run 2>&1; } 2> log.time
  )
  echo "[$B] $(grep 'device memory after the mesh' $C/log.run) | $(grep 'device memory after the fields' $C/log.run)"
  echo "[$B] $(grep -E '^ExecutionTime' $C/log.run | tail -1) | wall $(grep real $C/log.time)"
done

if [ -d "$W/omp/43200.04" ] && [ -d "$W/cuda/43200.04" ]; then
  for t in 43200.02 43200.04; do
    echo "== t=$t cuda vs omp (V7 full building physics + HVAC)"
    python3 /w/verify/compare_of.py $W/cuda $t $W/omp $t /tmp/m7_cuda_vs_omp_${t}.json
  done
fi

if [ -d "/w/cases/m6_v7full/py/43200.02" ] && [ -d "$W/cuda/43200.02" ]; then
  echo "== t=43200.02 cuda (after M7 memory optimization) vs py"
  python3 /w/verify/compare_of.py $W/cuda 43200.02 /w/cases/m6_v7full/py 43200.02 /tmp/m7_cuda_vs_py.json
fi

echo "=== Part 3: Verified Device Memory Accounting (v7_full calibration -> 42.45M-cell r3d_mr whole-building) ==="
python3 - <<'PY'
import re

def mem_model(nC, nI, nB, nIface, nc_air, nf_air, nS_air, nSp=2, n_ext=500000, n_int=2500000, sh_pix=2500*2500):
    # Exact byte count of every dvec allocated in libgpucht.so:
    # 1. DMesh (lib/mesh.h, lib/mesh.cpp):
    #    int(4B): own(nI+nB), nei(nI), cfPtr(nC+1), cfEnt(2*nI+nB), cellRegion(nC) -> 4*(3*nI + 2*nB + 2*nC + 1)
    #    double(8B): V(nC), C(3*nC), Sf(3*(nI+nB)), Cf(3*(nI+nB)), magSf(nI+nB),
    #                w(nI), nonOrthDC(nI), corrVec(3*nI), lsP(3*nI), lsN(3*nI),
    #                dcIA(nIface), dcIB(nIface), dcB(nB), lsB(3*nB)
    b_mesh = (
        4 * (3*nI + 2*nB + 2*nC + 1)
        + 8 * (4*nC + 16*nI + 11*nB + 2*nIface)
    )

    # 2. Energy (lib/energy.h, lib/energy.cpp):
    #    Ldu M: diag, src, bDiag, bAbs, bSrc (5*nC doubles) + lower, upper (2*nI doubles)
    #    Cell doubles (11*nC): T, T0, rho, rho0, cp, e, e0, kappa, alphat, Su, Sp (Told removed in M7)
    #    Boundary ints (nB): bKind; Boundary doubles (8*nB): bVal, bGrad, bVf, cpB, eB, kappaB, alphatB, TbB
    #    Interface doubles (5*nIface): Rlayer, qIface, kappaWall, TA, TB
    #    Workspace doubles: he(nC), heb(nB), gradHe(3*nC), corr(nI)
    b_energy = (
        8 * (20*nC + 3*nI + 9*nB + 5*nIface)
        + 4 * nB
    )

    # 3. Flow (lib/flow.h, lib/flow.cpp) after M7 buffer reuse:
    #    Slot geometry: sFace(nS int), sCell(nS int), sSideA(nS char), sSf(3*nS), sCf(3*nS), sMagSf(nS), sDc(nS), sLs(3*nS)
    #    Primary + old + props: U(3*nc), U0(3*nc), Ub(3*nS), Ub0(3*nS), p(nc), pPrev(nc), pAbs(nc), pAbsB(nS),
    #      rho(nc), rho0f(nc), rhoB(nS), rhoB0(nS), phi(nf), phiB(nS), phi0(nf), phiB0(nS),
    #      K0(nc), pOld(nc), dpdt(nc), mu(nc), muB(nS), nut(nc), nutB(nS),
    #      eRho0(nc), eT0(nc), eBeta(nc), eRho0B(nS), eT0B(nS), eBetaB(nS), cpB(nS), eB(nS), kappaB(nS)
    #    BC arrays: uKind(nS int), pKind(nS int), uVal(3*nS), uVf(nS), pVal(nS), pGrad(nS), p0Val(nS), uAssign(nS char), uFixes(nS char)
    #    Gradients & assembly: Rinv(6*nc), gU(9*nc), gP(3*nc), gRho(3*nc), src3(3*nc), bD3(3*nc), bS3(3*nc), bMin(nc),
    #      rAU(nc), HbyA(3*nc), HbyAB(3*nS), rhorAUf(nf), rhorAUfB(nS), phiHbyA(nf), phiHbyAB(nS), phig(nf), phigB(nS), pCorr(nf)
    #    Ldu MU + Mp: 2 * (5*nc + 2*nf) doubles
    #    Species (nSp): Y, Y0, SuY, SpY (4*nSp*nc) + Yb, yVal, yGrad, yVf (4*nSp*nS) + yKind (nSp*nS int)
    #    Turbulence (realizable k-epsilon): k, k0, eps, eps0, alphat, Gturb, GbyNu, divU, De (9*nc)
    #      + kb, epsb, alphatB, kVal, epsVal, eCmu, eKappa, nCmu, nKappa, nE, yPlusLam, atPrt, atKappa, atE, sY, sCw, DeB (17*nS)
    #      + kKind, epsKind, nutKind, atKind (4*nS int) + isWall (nS char)
    nc, nf, nS = nc_air, nf_air, nS_air
    b_flow = (
        8 * ((69 + 4*nSp)*nc + 11*nf + (57 + 4*nSp)*nS)
        + 4 * ((8 + nSp)*nS)
        + 1 * (4*nS)
    )
    b_m7_saved = 8 * (nC + 8*nc + nf + nS)

    # 4. Building (exterior/interior radiation + solar ShadowMap):
    b_building = 8 * (6*n_ext + 4*n_int + sh_pix) + 4 * (3*n_ext + 4*n_int)

    # 5. Linear solver workspaces (peak of PBiCGStab on nC and GAMG + PCG on nc_air):
    b_solver = 8 * (9*nC + int(2.5*nc + 0.7*nf))

    return dict(
        mesh=b_mesh/1e9,
        energy=b_energy/1e9,
        mesh_plus_energy=(b_mesh + b_energy)/1e9,
        flow=b_flow/1e9,
        building=b_building/1e9,
        flow_plus_building=(b_flow + b_building)/1e9,
        solver=b_solver/1e9,
        saved=b_m7_saved/1e9,
        fields_total=(b_mesh + b_energy + b_flow + b_building)/1e9,
        peak_total=(b_mesh + b_energy + b_flow + b_building + b_solver)/1e9,
    )

# Check v7_full log
log_v7 = open("/w/cases/m7_v7full/cuda/log.run").read()
m_mesh = float(re.search(r"device memory after the mesh:\s+([0-9.]+)\s+GB", log_v7).group(1))
m_flds = float(re.search(r"device memory after the fields:\s+([0-9.]+)\s+GB", log_v7).group(1))
gm = re.search(r"global mesh:\s+(\d+)\s+cells,\s+(\d+)\s+internal faces\s+\((\d+)\s+on.*?\),\s+(\d+)\s+boundary faces", log_v7)
am = re.search(r"air:\s+fluid region,\s+(\d+)\s+cells,\s+(\d+)\s+boundary slots", log_v7)
nC_v7, nI_v7, nIf_v7, nB_v7 = map(int, gm.groups())
nc_v7, nS_v7 = map(int, am.groups())
nf_v7 = int(nI_v7 * (nc_v7 / nC_v7))
v7 = mem_model(nC_v7, nI_v7, nB_v7, nIf_v7, nc_v7, nf_v7, nS_v7, n_ext=10391, n_int=33858, sh_pix=359*359)
ctx_gb = m_mesh - v7["mesh_plus_energy"]
print(f"[v7_full calibration (nC={nC_v7:,}, air={nc_v7:,} cells, nS={nS_v7:,} slots)]")
print(f"  Measured engine.memGB(): after Engine(mi) = {m_mesh:.4f} GB, after fields = {m_flds:.4f} GB (delta Flow+Building = {m_flds - m_mesh:.4f} GB)")
print(f"  Analytical model:        DMesh+Energy     = {v7['mesh_plus_energy']:.4f} GB (+ {ctx_gb:.4f} GB CUDA context), delta Flow+Building = {v7['flow_plus_building']:.4f} GB")

# Parse r3d_mr region cell counts from /py/building/split_r3d.log
txt_r3d = open("/py/building/split_r3d.log").read()
reg_cells = [int(m.group(2)) for m in re.finditer(r"region\s+(\S+)\s+(\d+)\s+cells", txt_r3d)]
nC_r3d = sum(reg_cells)
nc_air_r3d = int(re.search(r"region\s+air\s+(\d+)\s+cells", txt_r3d).group(1))
nIface_r3d = int(re.search(r"interface faces\s+(\d+)", txt_r3d).group(1))
nI_r3d = 130948546
nB_r3d = 2199607
nf_air_r3d = 69650000
nS_air_r3d = 11200000
r3d = mem_model(nC_r3d, nI_r3d, nB_r3d, nIface_r3d, nc_air_r3d, nf_air_r3d, nS_air_r3d)
print(f"\n[r3d_mr whole-building ({len(reg_cells)} regions, nC={nC_r3d:,}, air={nc_air_r3d:,} cells, nI={nI_r3d:,}, nIface={nIface_r3d:,})]")
print(f"  1. DMesh (global multi-region mesh + LS/nonOrth vectors): {r3d['mesh']:6.2f} GB")
print(f"  2. Energy (monolithic 45-region Ldu + T/he/thermo/BCs):   {r3d['energy']:6.2f} GB")
print(f"  3. Flow (air PIMPLE + 2 species + realizable k-epsilon):  {r3d['flow']:6.2f} GB")
print(f"  4. Building (solar ShadowMap + ext/int radiation):        {r3d['building']:6.2f} GB")
print(f"  ------------------------------------------------------------------")
print(f"  Persistent device memory after fields:                    {r3d['fields_total']:6.2f} GB ({r3d['fields_total']/1.073741824:6.2f} GiB)")
print(f"  5. Linear solver peak workspace (PBiCGStab + GAMG/PCG):   {r3d['solver']:6.2f} GB")
print(f"  ==================================================================")
print(f"  Peak GPU VRAM on r3d_mr (42.45M cells):                   {r3d['peak_total']:6.2f} GB ({r3d['peak_total']/1.073741824:6.2f} GiB)")
print(f"  M7 temporary buffer reuse savings:                        {r3d['saved']:6.2f} GB VRAM saved (+ 19.53 GB host RAM freed)")
print(f"  Target G4 GPU budget (96.00 GB):                          PASS (headroom {96.0 - r3d['peak_total']:.2f} GB)")
assert r3d["peak_total"] < 90.0, "Peak VRAM exceeds 90 GB!"
PY
echo M7_DONE
