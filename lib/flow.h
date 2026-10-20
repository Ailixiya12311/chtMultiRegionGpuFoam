// One fluid region of chtMultiRegionFoam (v2606 fluid/UEqn.H, pEqn.H, rhoEqn.H), rho form (heRhoThermo with an
// equation of state of psi = 0: Boussinesq / rhoConst; psi*correction(ddt(p_rgh)) vanishes).
//
// Indexing: cells local i (global c0 + i); region internal faces local j (global f0 + j); boundary "slots" s:
// the region's boundary faces first (global b0 + s), then its interface faces (walls for the flow), in the order
// the application lists them. Slot geometry is outward from the fluid cell.
#pragma once

#include "energy.h"

namespace gc
{

// U slot kinds: U_b = a + b U_P per component
enum UKind { U_FIXED = 0, U_ZEROGRAD = 1, U_INLETOUTLET = 2, U_PIOV = 3, U_SLIP = 4 };
// p_rgh slot kinds
enum PKind { P_FIXED = 0, P_ZEROGRAD = 1, P_FLUX = 2, P_PRGH_P = 3, P_PRGH_TOTALP = 4 };

struct Flow
{
    const DMesh* m = nullptr;
    Energy* E = nullptr;
    int region = 0, c0 = 0, nc = 0, f0 = 0, nf = 0, b0 = 0, nBr = 0, nS = 0;

    // slots
    dvec<int> sFace;                          // global face id (internal face for interface slots)
    dvec<int> sCell;                          // local cell
    dvec<double> sSf, sCf, sMagSf, sDc, sLs;  // outward Sf, Cf, |Sf|, patch deltaCoeffs, leastSquares vector
    dvec<char> sSideA;                        // interface slots: fluid is side A (owner)
    dvec<int> ifSlot;                         // global interface index -> slot (-1: not this region)
    dvec<double> Rinv;                        // 6 per cell: safeInv(sum SfHat Sf) for fvc::reconstruct

    // fields
    dvec<double> U, U0, Ub, Ub0;              // 3 nc / 3 nS
    dvec<double> p, pPrev, pAbs, pAbsB;       // p_rgh, its prevIter, p (cells, slots)
    dvec<double> rho, rho0f, rhoB, rhoB0;     // rho (cells), rho.oldTime(), slot values (current, old)
    dvec<double> phi, phiB, phi0, phiB0;      // mass flux: internal faces, slots (current, old)
    dvec<double> K0, pOld, dpdt;              // K.oldTime(), p.oldTime() (for dpdt), dpdt
    dvec<double> mu, muB, nut, nutB;          // laminar dynamic viscosity, turbulent viscosity
    // equation of state per cell / slot: rho = rho0 (1 - beta (T - T0))
    dvec<double> eRho0, eT0, eBeta, eRho0B, eT0B, eBetaB;
    // boundary conditions
    dvec<int> uKind, pKind;
    dvec<double> uVal, uVf;                   // 3 nS, nS
    dvec<char> uAssign, uFixes;               // assignable(), fixesValue()
    dvec<double> pVal, p0Val, pGrad;          // fixed value, p/p0 of prgh*Pressure, fixedFluxPressure gradient
    int nFP = 0;                              // flowRateInletVelocity / flowRateOutletVelocity patches
    dvec<int> fpS0, fpN, fpKind;              // kind: bit 0 = massFlowRate, bit 1 = outlet
    dvec<double> fpQ;

    // settings
    double g[3] = {0, 0, 0};
    double ghRef = 0;
    bool solveDir[3] = {true, true, true};
    int conv = 1;                             // div(phi,U): 0 upwind, 1 linearUpwind grad(U)
    bool bounded = false;
    double limitKU = 0;                       // grad(U) cellLimited coefficient (0: none)
    double psiU = 1, psiP = 1, psiSnP = 1, psiSnRho = 1;   // laplacian / snGrad limited-corrected coefficients
    int pRefCell = -1;                        // local cell, -1: p_rgh does not need a reference
    double pRefValue = 0;
    bool kOldReady = false, pOldReady = false;

    // work
    dvec<double> gU, gP, gRho;                // grad(U) (9 nc, [i*3+j] = d u_j / d x_i), grad(p_rgh), grad(rho)
    dvec<double> src3, bD3, bS3, bMin;        // UEqn: sources, boundary diagonal / source per component
    dvec<double> rAU, HbyA, HbyAB;
    dvec<double> rhorAUf, rhorAUfB, phiHbyA, phiHbyAB, phig, phigB, pCorr;
    Ldu MU, Mp;
    AmgCache amgP;
    bool haveUEqn = false;

    // species (multiComponentMixture; nSp == 0: pureMixture)
    int nSp = 0, inertIndex = -1, convY = 0;  // convY: 0 upwind, 3 limitedLinear
    double convYK = 1.0, limitKH = 0.0, Tref0 = 298.15;
    bool boundedY = false;
    std::vector<SpecieIn> sp;
    dvec<double> spData;                      // 7 nSp: Cp, Hsref, mu, Pr, rho0, T0, beta per specie
    dvec<double> Y, Y0, SuY, SpY;             // nSp * nc (specie m at m*nc)
    dvec<double> Yb, yVal, yGrad, yVf;        // nSp * nS (specie m at m*nS)
    dvec<int> yKind;                          // nSp * nS
    dvec<double> cpB, eB, kappaB;             // slot Cp, e, kappa (all nS slots)

    // turbulence (RNGkEpsilon; turb.enabled == false: laminar)
    TurbIn turb;
    dvec<double> k, k0, eps, eps0, alphat;    // cells (nc)
    dvec<double> kb, epsb, alphatB;           // slots (nS)
    dvec<int> kKind, epsKind, nutKind, atKind;
    dvec<double> kVal, epsVal;
    dvec<double> eCmu, eKappa, nCmu, nKappa, nE, yPlusLam;
    dvec<double> atPrt, atKappa, atE, sY, sCw;
    dvec<char> isWall;
    dvec<int> wallCells, wallCellsGlobal;     // unique wall-adjacent cells (local, global)
    dvec<double> epsWall, Gturb, GbyNu, divU, De, DeB;

    void init(const DMesh& mesh, Energy& energy, int regionIndex, const std::vector<int>& ifFaces,
              const std::vector<char>& ifSideA);
    void initSpecies(const FluidIn& fi);
    void initTurbulence(const FluidIn& fi);
    void setupTurbulence(const std::vector<double>& kIn, const std::vector<double>& epsIn,
                         const std::vector<double>& nutIn, const std::vector<double>& alphatIn,
                         const std::vector<int>& kK, const std::vector<double>& kV,
                         const std::vector<int>& eK, const std::vector<double>& eV,
                         const std::vector<int>& nK, const std::vector<int>& aK,
                         const std::vector<double>& eCmuIn, const std::vector<double>& eKappaIn,
                         const std::vector<double>& nCmuIn, const std::vector<double>& nKappaIn,
                         const std::vector<double>& nEIn, const std::vector<double>& atPrtIn,
                         const std::vector<double>& atKappaIn, const std::vector<double>& atEIn,
                         const std::vector<double>& sYIn, const std::vector<double>& alphatB0);
    void validateTurbulence();
    void correctNut();
    void updateAlphat();
    void updateMixture(bool evalYb = true);   // composition -> mu, rho0/T0/beta, E->cp/e/kappa (cells & slots)
    // derived boundary values (U_b, rho_b) from the current cell fields
    void updateBoundary(bool evalFlowPatches = true);
    void initPhi();                            // phi = interpolate(rho U) . Sf
    void storeOld(double dt, bool frozen, bool storePrevIter);
    void momentum(double dt, double alpha, const SolverCtl& ctl, bool predictor, SolveInfo info[3]);
    void species(double dt, const std::vector<double>& alphaY, const SolverCtl& ctl, std::vector<SolveInfo>& info);
    void pressure(double dt, const SolverCtl& ctl, double pRelax, int nNonOrth, SolveInfo& info,
                  double contErr[2]);
    void turbulence(double dt, double alphaK, double alphaEps, const SolverCtl& ctlK, const SolverCtl& ctlEps,
                    SolveInfo& infoEps, SolveInfo& infoK);
    void updateRho();                          // rho = thermo rho(T)
    // energy coupling: rho, rho.oldTime(), phi on the region faces, explicit EEqn terms into E->Su
    void toEnergy(double dt, bool dpdtOn);
    void updatePAbs();                         // p = p_rgh + rho gh (+ closed-volume level)
    double courantNo(double dt, double& meanCo);
};

} // namespace gc
