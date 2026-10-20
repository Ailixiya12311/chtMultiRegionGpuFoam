// Linear solvers on an Ldu system. Residuals use OpenFOAM's normalisation (L1 norm / normFactor), and
// convergence follows lduMatrix::solver (tolerance, relTol, maxIter, minIter), so logs compare directly.
//   PCG       flexible (Polak-Ribiere) conjugate gradients, Jacobi, DILU (= DIC on a symmetric matrix) or AMG
//             preconditioner (symmetric matrices)
//   PBICGSTAB BiCGStab, Jacobi, DILU or AMG preconditioner
//   SMOOTH    OpenFOAM's smoothSolver: Gauss-Seidel, symmetric Gauss-Seidel or DILU smoother sweeps, residual
//             checked every nSweeps
// DILU and Gauss-Seidel run in a multi-colour order (no two neighbours share a colour; colours in turn, the cells
// of one colour in parallel), built once per system and kept in DMesh::colourings.
// AMG: aggregation multigrid (double pairwise matching per level, piecewise-constant transfer, l1-Jacobi
// smoothing, minimal-residual scaling of the coarse correction, dense inverse on the coarsest level). The finest
// level works on the Ldu matrix itself; coarse levels are CSR.
#pragma once

#include <memory>

#include "ldu.h"

namespace gc
{

struct AmgHierarchy;

struct AmgCache
{
    std::unique_ptr<AmgHierarchy> h;
    int left = 0;
    int reuseEvery = 10;                      // rebuild the coarse levels every N solves
    AmgCache();
    ~AmgCache();
};

// x: local field (cell - c0) on input the initial guess, on output the solution
SolveInfo solve(const DMesh& m, const Ldu& M, double* x, const SolverCtl& ctl, AmgCache* cache = nullptr);

} // namespace gc
