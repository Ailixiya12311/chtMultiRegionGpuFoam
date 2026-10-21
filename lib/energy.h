// Monolithic energy equation over all regions, solved in T for the OpenFOAM he equations (he = cp T + e per cell,
// hConst / Boussinesq mixtures): fluid EEqn.H and solid solveSolid.H multiplied out, region interfaces coupled
// through the series conductance of the two patch sides (turbulentTemperatureCoupledBaffleMixed at convergence).
// Interface faces act as patch faces of each region for gradients, limiters, non-orthogonal corrections and
// relaxation, as they are in OpenFOAM's region meshes.
#pragma once

#include "linsolve.h"

namespace gc
{

struct Energy
{
    const DMesh* m = nullptr;
    int nR = 0;

    // cells
    dvec<double> T, rho, rhoOld, hOld, cp, e, kappa, alphat, Su, Sp;
    dvec<double> Sx;                          // explicit fluid EEqn terms (rho U.g, K terms, dpdt) [W]
    dvec<char> fluid;
    // regions: cellLimited coefficient of grad(he) (<= 0: none), laplacian limited-corrected coefficient psi
    // (1: corrected, 0: uncorrected), equation relaxation factor (<= 0: no relaxation)
    dvec<double> rLimitK, rPsi, rRelax;
    // boundary faces
    dvec<int> bKind;
    dvec<double> bVal, bGrad, bVf, cpB, eB, kappaB, alphatB;
    // interface faces
    dvec<double> Rlayer, qIface, kappaWall;   // kappaWall < 0: fluid side uses kappa of the cell
    // linearised surface exchange w [W/K] of a face without layers: its surface node takes w (Tref - T_s) besides
    // qIface, solved with the two sides (T_s = (gA T_A + gB T_B + qIface)/(gA + gB + w)), so Tref = Tmrt enters
    // through qIface (Building, interior longwave)
    dvec<double> wRad;
    // fluid mass flux on fluid faces (global face ids; zero elsewhere), convection scheme
    dvec<double> phi;                         // nI + nB
    int conv = 0;                             // 0 upwind, 1 linearUpwind, 2 linear, 3 limitedLinear
    double convK = 1.0;                       // limitedLinear coefficient
    bool bounded = false;                     // bounded Gauss (-Sp(div(phi)))
    // fixedTemperatureConstraint
    dvec<int> fixCells;
    dvec<double> fixVals;

    // work
    dvec<double> aB, bB, TbB, GbB;            // boundary: T_b = aB + bB T_P, T_b, conductance alphaEff |S| dc
    dvec<double> gA, gB, G, TA, TB;           // interface: side conductances, series conductance, patch T
    dvec<double> h, grad;                     // he per cell, (limited) grad(he)
    dvec<double> fS;                          // region faces: explicit face terms to the owner row (assemble)
    Ldu M;
    AmgCache amg;

    void init(const DMesh& mesh, int nRegions);
    void boundaryState();                     // aB, bB, TbB, GbB, interface conductances and temperatures
    void gradient();
    void assemble(double dt);
    SolveInfo solve(double dt, const SolverCtl& ctl, bool relaxOn);
    void storeOld();
};

} // namespace gc
