#include "ldu.h"

namespace gc
{

void Ldu::init(int c0_, int nc_, int f0_, int nf_, int fBL_)
{
    c0 = c0_; nc = nc_; f0 = f0_; nf = nf_; fBL = fBL_;
    diag.resize(nc); src.resize(nc); bDiag.resize(nc); bAbs.resize(nc); bSrc.resize(nc);
    upper.resize(nf); lower.resize(nf);
    zero();
}

void Ldu::zero()
{
    fill(diag, 0.0); fill(src, 0.0); fill(bDiag, 0.0); fill(bAbs, 0.0); fill(bSrc, 0.0);
    fill(upper, 0.0); fill(lower, 0.0);
}

LduView view(Ldu& M)
{
    LduView v;
    v.c0 = M.c0; v.nc = M.nc; v.f0 = M.f0; v.nf = M.nf; v.fBL = M.fBL;
    v.diag = ptr(M.diag); v.upper = ptr(M.upper); v.lower = ptr(M.lower); v.src = ptr(M.src);
    v.bDiag = ptr(M.bDiag); v.bAbs = ptr(M.bAbs); v.bSrc = ptr(M.bSrc);
    return v;
}

void amul(const DMesh& m, const Ldu& M, const double* x, double* y)
{
    const MeshView g = view(m);
    const int c0 = M.c0, f0 = M.f0, f1 = M.f0 + M.nf, nI = g.nI;
    const double *d = ptr(M.diag), *bd = ptr(M.bDiag), *up = ptr(M.upper), *lo = ptr(M.lower);
    for_n(M.nc, [=] HD (long i) {
        const long c = c0 + i;
        double s = (d[i] + bd[i])*x[i];
        for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
        {
            const int e = g.cfEnt[k];
            if (e >= 0)
            {
                if (e < f0 || e >= f1 || e >= nI) continue;
                s += up[e - f0]*x[g.nei[e] - c0];
            }
            else
            {
                const int f = -e - 1;
                if (f < f0 || f >= f1) continue;
                s += lo[f - f0]*x[g.own[f] - c0];
            }
        }
        y[i] = s;
    });
}

void sumA(const DMesh& m, const Ldu& M, double* out)
{
    const MeshView g = view(m);
    const int c0 = M.c0, f0 = M.f0, f1 = M.f0 + M.nf, nI = g.nI;
    const double *d = ptr(M.diag), *bd = ptr(M.bDiag), *up = ptr(M.upper), *lo = ptr(M.lower);
    for_n(M.nc, [=] HD (long i) {
        const long c = c0 + i;
        double s = d[i] + bd[i];
        for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
        {
            const int e = g.cfEnt[k];
            if (e >= 0)
            {
                if (e < f0 || e >= f1 || e >= nI) continue;
                s += up[e - f0];
            }
            else
            {
                const int f = -e - 1;
                if (f < f0 || f >= f1) continue;
                s += lo[f - f0];
            }
        }
        out[i] = s;
    });
}

void relax(const DMesh& m, Ldu& M, const double* alpha, const double* psi, const double* bMin)
{
    const MeshView g = view(m);
    LduView L = view(M);
    const int c0 = L.c0, f0 = L.f0, f1 = L.f0 + L.nf, fBL = L.fBL, nI = g.nI;
    for_n(L.nc, [=] HD (long i) {
        const double a = alpha[i];
        if (!(a > 0.0)) return;
        const long c = c0 + i;
        // sum of |off-diagonal| over interior faces (boundary-like faces count as patch coefficients)
        double sumOff = 0.0;
        for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
        {
            const int e = g.cfEnt[k];
            if (e >= 0)
            {
                if (e < f0 || e >= f1 || e >= nI || e >= fBL) continue;
                sumOff += fabs(L.upper[e - f0]);
            }
            else
            {
                const int f = -e - 1;
                if (f < f0 || f >= f1 || f >= fBL) continue;
                sumOff += fabs(L.lower[f - f0]);
            }
        }
        const double D0 = L.diag[i];
        double D = D0 + L.bAbs[i];
        D = dmax(fabs(D), sumOff);
        D /= a;
        D -= (bMin ? bMin[i] : L.bDiag[i]);
        L.src[i] += (D - D0)*psi[i];
        L.diag[i] = D;
    });
}

void setValues(const DMesh& m, Ldu& M, const dvec<int>& cells, const dvec<double>& values, double* psi)
{
    const long n = long(cells.size());
    if (n == 0) return;
    const MeshView g = view(m);
    LduView L = view(M);
    const int c0 = L.c0, f0 = L.f0, f1 = L.f0 + L.nf, nI = g.nI;
    // marker: value per constrained cell (NaN elsewhere)
    dvec<double> mark(L.nc);
    fill(mark, double(NAN));
    double* mk = ptr(mark);
    const int* cl = ptr(cells);
    const double* vl = ptr(values);
    for_n(n, [=] HD (long j) { mk[cl[j] - c0] = vl[j]; });
    // unconstrained rows: the known values of constrained neighbours go to the source
    for_n(L.nc, [=] HD (long i) {
        if (!nanq(mk[i])) return;
        const long c = c0 + i;
        double s = 0.0;
        for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
        {
            const int e = g.cfEnt[k];
            if (e >= 0)
            {
                if (e < f0 || e >= f1 || e >= nI) continue;
                const double v = mk[g.nei[e] - c0];
                if (!nanq(v)) s += L.upper[e - f0]*v;
            }
            else
            {
                const int f = -e - 1;
                if (f < f0 || f >= f1) continue;
                const double v = mk[g.own[f] - c0];
                if (!nanq(v)) s += L.lower[f - f0]*v;
            }
        }
        L.src[i] -= s;
    });
    // cut the connections of constrained cells
    const int* own = g.own;
    const int* nei = g.nei;
    for_n(L.nf, [=] HD (long j) {
        const long f = f0 + j;
        if (!nanq(mk[own[f] - c0]) || !nanq(mk[nei[f] - c0]))
        {
            L.upper[j] = 0.0;
            L.lower[j] = 0.0;
        }
    });
    // constrained rows: diagonal only (interior part), source = value * diagonal, no patch contributions
    for_n(L.nc, [=] HD (long i) {
        const double v = mk[i];
        if (nanq(v)) return;
        L.src[i] = v*L.diag[i];
        L.bDiag[i] = 0.0;
        L.bAbs[i] = 0.0;
        L.bSrc[i] = 0.0;
        psi[i] = v;
    });
}

void effective(const Ldu& M, double* d, double* b)
{
    const double *dg = ptr(M.diag), *bd = ptr(M.bDiag), *s = ptr(M.src), *bs = ptr(M.bSrc);
    for_n(M.nc, [=] HD (long i) {
        d[i] = dg[i] + bd[i];
        b[i] = s[i] + bs[i];
    });
}

} // namespace gc
