// LDU matrix on a cell range [c0, c0 + nc) with coefficients on the internal faces [f0, f0 + nf) (global face ids),
// OpenFOAM layout: upper[f] = coefficient of the neighbour value in the owner row, lower[f] = coefficient of the
// owner value in the neighbour row. Faces of the range with global id >= fBL are "boundary-like": in OpenFOAM they
// are patch faces (region interfaces seen from each region), so relax() treats their diagonal part like
// non-coupled patch coefficients and leaves them out of the off-diagonal sum.
//
// Boundary contributions are kept apart from the interior part, as fvMatrix does with internalCoeffs and
// boundaryCoeffs: per cell bDiag = sum of internal coefficients, bAbs = sum of their magnitudes, bSrc = sum of the
// boundary source. The solved system is (diag + bDiag) x = src + bSrc.
#pragma once

#include "mesh.h"

namespace gc
{

struct Ldu
{
    int c0 = 0, nc = 0, f0 = 0, nf = 0, fBL = 0x7fffffff;
    dvec<double> diag, upper, lower, src;     // interior parts
    dvec<double> bDiag, bAbs, bSrc;           // boundary parts per cell

    void init(int c0_, int nc_, int f0_, int nf_, int fBL_ = 0x7fffffff);
    void zero();
};

struct LduView
{
    int c0, nc, f0, nf, fBL;
    double *diag, *upper, *lower, *src, *bDiag, *bAbs, *bSrc;
};
LduView view(Ldu& M);

// y = A x (A = diag + bDiag + off-diagonals), x and y indexed locally (cell - c0)
void amul(const DMesh& m, const Ldu& M, const double* x, double* y);
// row sums of A
void sumA(const DMesh& m, const Ldu& M, double* s);
// OpenFOAM fvMatrix::relax(alpha) with per-cell alpha (alpha <= 0: that cell is not relaxed). psi: current field.
// bMin: per-cell sum of the smallest boundary coefficient component (vector equations; nullptr = bDiag)
void relax(const DMesh& m, Ldu& M, const double* alpha, const double* psi, const double* bMin = nullptr);
// fvMatrix::setValues: cells (global ids) held at values; psi (local) is set too
void setValues(const DMesh& m, Ldu& M, const dvec<int>& cells, const dvec<double>& values, double* psi);
// effective diagonal and source of the solved system
void effective(const Ldu& M, double* d, double* b);

} // namespace gc
