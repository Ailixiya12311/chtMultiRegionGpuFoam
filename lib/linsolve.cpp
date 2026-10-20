#include "linsolve.h"

#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gc
{

namespace
{

// ------------------------------------------------------------------------------------------- vector helpers
double dotv(long n, const double* a, const double* b)
{
    return sum_n(n, [=] HD (long i) { return a[i]*b[i]; });
}

double sumMag(long n, const double* a)
{
    return sum_n(n, [=] HD (long i) { return fabs(a[i]); });
}

// ------------------------------------------------------------------------------------------- row access
// local indices (cell - c0); fn(j, a_ij) for the off-diagonal entries of row i
struct LduRows
{
    MeshView g;
    int c0, f0, f1;
    const double *up, *lo;
};

template<class Fn> HD inline void forRow(const LduRows& R, long i, Fn fn)
{
    const long c = R.c0 + i;
    for (int k = R.g.cfPtr[c]; k < R.g.cfPtr[c + 1]; ++k)
    {
        const int e = R.g.cfEnt[k];
        if (e >= 0)
        {
            if (e < R.f0 || e >= R.f1 || e >= R.g.nI) continue;
            fn(long(R.g.nei[e] - R.c0), R.up[e - R.f0]);
        }
        else
        {
            const int f = -e - 1;
            if (f < R.f0 || f >= R.f1) continue;
            fn(long(R.g.own[f] - R.c0), R.lo[f - R.f0]);
        }
    }
}

struct CsrRows
{
    const int *rp, *ci;
    const double* v;
};

template<class Fn> HD inline void forRow(const CsrRows& R, long i, Fn fn)
{
    for (int k = R.rp[i]; k < R.rp[i + 1]; ++k)
        if (R.ci[k] != i) fn(long(R.ci[k]), R.v[k]);
}

struct Csr
{
    int n = 0;
    dvec<int> rp, ci;
    dvec<double> v, d;                        // values, diagonal
    CsrRows rows() const { return CsrRows{ptr(rp), ptr(ci), ptr(v)}; }
};

template<class Rows> void apply(const Rows& R, long n, const double* d, const double* x, double* y)
{
    for_n(n, [=] HD (long i) {
        double s = d[i]*x[i];
        forRow(R, i, [&](long j, double a) { s += a*x[j]; });
        y[i] = s;
    });
}

// ------------------------------------------------------------------------------------------- aggregation
// One pass of pairwise matching (cht/amg.py pairwise): mutual strongest pairs in up to 3 rounds, leftovers join
// their strongest aggregated neighbour, the rest stay single. Strength s_ij = -a_ij / sqrt(|d_i d_j|) > 0.
template<class Rows> int pairwise(const Rows& R, long n, const double* d, dvec<int>& aggv)
{
    aggv.resize(n);
    fill(aggv, -1);
    int* agg = ptr(aggv);
    dvec<int> bestv(n), flagv(n), idsv(n);
    int* best = ptr(bestv);
    int* flag = ptr(flagv);
    int* ids = ptr(idsv);
    int nc = 0;
    for (int round = 0; round < 3; ++round)
    {
        for_n(n, [=] HD (long i) {
            best[i] = -1;
            if (agg[i] >= 0) return;
            double smax = 0.0;
            forRow(R, i, [&](long j, double a) {
                if (agg[j] >= 0) return;
                const double s = -a/sqrt(fabs(d[i]*d[j]));
                if (s > smax) smax = s;
            });
            if (!(smax > 0.0)) return;
            int bj = -1;
            forRow(R, i, [&](long j, double a) {
                if (agg[j] >= 0) return;
                const double s = -a/sqrt(fabs(d[i]*d[j]));
                if (s > 0.0 && s >= smax*(1 - 1e-12) && (bj < 0 || j < bj)) bj = int(j);
            });
            best[i] = bj;
        });
        for_n(n, [=] HD (long i) {
            const int j = best[i];
            flag[i] = (j >= 0 && i < j && best[j] == i) ? 1 : 0;
        });
        thrust::exclusive_scan(thrust::device, flagv.begin(), flagv.end(), idsv.begin());
        const int k = int(sum_n(n, [=] HD (long i) { return double(flag[i]); }));
        if (k == 0) break;
        const int base = nc;
        for_n(n, [=] HD (long i) {
            if (flag[i])
            {
                agg[i] = base + ids[i];
                agg[best[i]] = base + ids[i];
            }
        });
        nc += k;
    }
    // leftovers: strongest already aggregated neighbour
    for_n(n, [=] HD (long i) {
        best[i] = -1;
        if (agg[i] >= 0) return;
        double smax = 0.0;
        forRow(R, i, [&](long j, double a) {
            if (agg[j] < 0) return;
            const double s = -a/sqrt(fabs(d[i]*d[j]));
            if (s > smax) smax = s;
        });
        if (!(smax > 0.0)) return;
        int bj = -1;
        forRow(R, i, [&](long j, double a) {
            if (agg[j] < 0) return;
            const double s = -a/sqrt(fabs(d[i]*d[j]));
            if (s > 0.0 && s >= smax*(1 - 1e-12) && (bj < 0 || j < bj)) bj = int(j);
        });
        best[i] = bj;
    });
    for_n(n, [=] HD (long i) {
        if (agg[i] < 0 && best[i] >= 0) flag[i] = agg[best[i]];
        else flag[i] = -2;
    });
    for_n(n, [=] HD (long i) { if (flag[i] >= 0) agg[i] = flag[i]; });
    // singletons
    for_n(n, [=] HD (long i) { flag[i] = agg[i] < 0 ? 1 : 0; });
    thrust::exclusive_scan(thrust::device, flagv.begin(), flagv.end(), idsv.begin());
    const int ns = int(sum_n(n, [=] HD (long i) { return double(flag[i]); }));
    const int base = nc;
    for_n(n, [=] HD (long i) { if (flag[i]) agg[i] = base + ids[i]; });
    return nc + ns;
}

// COO (key = row*nc + col) -> CSR with summed duplicates
void cooToCsr(dvec<long long>& keys, dvec<double>& vals, int nc, Csr& out)
{
    thrust::sort_by_key(thrust::device, keys.begin(), keys.end(), vals.begin());
    dvec<long long> uk(keys.size());
    dvec<double> uv(keys.size());
    auto ends = thrust::reduce_by_key(thrust::device, keys.begin(), keys.end(), vals.begin(), uk.begin(), uv.begin());
    const long nnz = long(ends.first - uk.begin());
    keys.clear(); keys.shrink_to_fit();
    vals.clear(); vals.shrink_to_fit();
    out.n = nc;
    out.ci.resize(nnz);
    out.v.resize(nnz);
    dvec<int> rows(nnz);
    {
        const long long* k = ptr(uk);
        const double* v = ptr(uv);
        int* ci = ptr(out.ci);
        double* vv = ptr(out.v);
        int* rw = ptr(rows);
        const long long ncl = nc;
        for_n(nnz, [=] HD (long i) {
            rw[i] = int(k[i]/ncl);
            ci[i] = int(k[i]%ncl);
            vv[i] = v[i];
        });
    }
    out.rp.resize(nc + 1);
    thrust::lower_bound(thrust::device, rows.begin(), rows.end(), thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>(nc + 1), out.rp.begin());
    out.d.resize(nc);
    {
        const int* rp = ptr(out.rp);
        const int* ci = ptr(out.ci);
        const double* v = ptr(out.v);
        double* d = ptr(out.d);
        for_n(nc, [=] HD (long i) {
            double s = 0.0;
            for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] == i) s += v[k];
            d[i] = s;
        });
    }
}

void galerkinLdu(const LduRows& R, long n, long nf, const double* d, const int* agg, int nc, Csr& out)
{
    dvec<long long> keys(n + 2*nf);
    dvec<double> vals(n + 2*nf);
    long long* k = ptr(keys);
    double* v = ptr(vals);
    const long long ncl = nc;
    for_n(n, [=] HD (long i) {
        k[i] = agg[i]*ncl + agg[i];
        v[i] = d[i];
    });
    const MeshView g = R.g;
    const int c0 = R.c0, f0 = R.f0;
    const double *up = R.up, *lo = R.lo;
    for_n(nf, [=] HD (long j) {
        const long f = f0 + j;
        const int o = agg[g.own[f] - c0], nb = agg[g.nei[f] - c0];
        k[n + j] = o*ncl + nb;
        v[n + j] = up[j];
        k[n + nf + j] = nb*ncl + o;
        v[n + nf + j] = lo[j];
    });
    cooToCsr(keys, vals, nc, out);
}

void galerkinCsr(const Csr& A, const int* agg, int nc, Csr& out)
{
    const long nnz = long(A.ci.size());
    dvec<long long> keys(nnz);
    dvec<double> vals(nnz);
    long long* k = ptr(keys);
    double* v = ptr(vals);
    const int* rp = ptr(A.rp);
    const int* ci = ptr(A.ci);
    const double* av = ptr(A.v);
    const long long ncl = nc;
    for_n(A.n, [=] HD (long i) {
        for (int q = rp[i]; q < rp[i + 1]; ++q)
        {
            k[q] = agg[i]*ncl + agg[ci[q]];
            v[q] = av[q];
        }
    });
    cooToCsr(keys, vals, nc, out);
}

template<class Rows> void l1Dinv(const Rows& R, long n, const double* d, dvec<double>& Dinv)
{
    Dinv.resize(n);
    double* di = ptr(Dinv);
    for_n(n, [=] HD (long i) {
        double l1 = fabs(d[i]);
        forRow(R, i, [&](long, double a) { l1 += fabs(a); });
        const double m = dmax(l1, fabs(d[i]));
        di[i] = m > 0.0 ? 1.0/m : 1.0;
    });
}

} // namespace

// ============================================================================================ AMG hierarchy
struct AmgHierarchy
{
    struct Level
    {
        long n = 0;
        Csr A;                                 // coarse levels (level 0: the Ldu)
        dvec<double> Dinv;
        dvec<int> agg;                         // to the next level
        int nc = 0;
        dvec<int> mPtr, mIdx;                  // members of each coarse cell (restriction by gather)
        dvec<double> x, b, r, e, t;
    };
    std::vector<Level> L;
    dvec<double> coarseInv;                    // dense inverse (row-major) of the coarsest level, or empty
    int nsweep = 2;
    // level 0 operator
    LduRows R0;
    const double* d0 = nullptr;

    static constexpr int coarseSize = 500;
    static constexpr int denseMax = 1000;

    void build(const LduRows& R, long n, long nf, const double* d);
    void setFine(const LduRows& R, const double* d);
    void op(int k, const double* x, double* y) const;
    void cycle(int k, const double* b, double* x);
    void members(Level& lv);
    void denseInverse(const Csr& A);
};

void AmgHierarchy::members(Level& lv)
{
    // stable order of the fine cells by aggregate
    lv.mIdx.resize(lv.n);
    thrust::sequence(thrust::device, lv.mIdx.begin(), lv.mIdx.end());
    dvec<int> key(lv.agg);
    thrust::stable_sort_by_key(thrust::device, key.begin(), key.end(), lv.mIdx.begin());
    lv.mPtr.resize(lv.nc + 1);
    thrust::lower_bound(thrust::device, key.begin(), key.end(), thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>(lv.nc + 1), lv.mPtr.begin());
}

void AmgHierarchy::denseInverse(const Csr& A)
{
    const int n = A.n;
    std::vector<int> rp = to_host(A.rp), ci = to_host(A.ci);
    std::vector<double> v = to_host(A.v);
    std::vector<double> a(size_t(n)*n, 0.0), inv(size_t(n)*n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int k = rp[i]; k < rp[i + 1]; ++k) a[size_t(i)*n + ci[k]] += v[k];
        inv[size_t(i)*n + i] = 1.0;
    }
    // Gauss-Jordan with partial pivoting; a (near-)singular pivot (pure Neumann system) is regularised
    double scale = 0.0;
    for (int i = 0; i < n; ++i) scale = std::max(scale, std::fabs(a[size_t(i)*n + i]));
    for (int c = 0; c < n; ++c)
    {
        int pr = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(a[size_t(r)*n + c]) > std::fabs(a[size_t(pr)*n + c])) pr = r;
        if (pr != c)
            for (int j = 0; j < n; ++j)
            {
                std::swap(a[size_t(c)*n + j], a[size_t(pr)*n + j]);
                std::swap(inv[size_t(c)*n + j], inv[size_t(pr)*n + j]);
            }
        double p = a[size_t(c)*n + c];
        if (std::fabs(p) < 1e-13*scale) p = (p < 0 ? -1.0 : 1.0)*1e-13*scale;
        const double rp_ = 1.0/p;
        for (int j = 0; j < n; ++j) { a[size_t(c)*n + j] *= rp_; inv[size_t(c)*n + j] *= rp_; }
        for (int r = 0; r < n; ++r)
        {
            if (r == c) continue;
            const double f = a[size_t(r)*n + c];
            if (f == 0.0) continue;
            for (int j = 0; j < n; ++j)
            {
                a[size_t(r)*n + j] -= f*a[size_t(c)*n + j];
                inv[size_t(r)*n + j] -= f*inv[size_t(c)*n + j];
            }
        }
    }
    coarseInv = to_dev(inv);
}

void AmgHierarchy::setFine(const LduRows& R, const double* d)
{
    R0 = R;
    d0 = d;
    l1Dinv(R0, L[0].n, d0, L[0].Dinv);
}

void AmgHierarchy::build(const LduRows& R, long n, long nf, const double* d)
{
    L.clear();
    L.emplace_back();
    L[0].n = n;
    setFine(R, d);
    const int passes = 2;
    while (L.back().n > coarseSize && L.size() < 25)
    {
        Level& lv = L.back();
        const int k = int(L.size()) - 1;
        Csr Ac;
        dvec<int> aggTot;
        int nc = 0;
        for (int pass = 0; pass < passes; ++pass)
        {
            dvec<int> agg;
            Csr An;
            if (pass == 0 && k == 0)
            {
                nc = pairwise(R0, n, d0, agg);
                galerkinLdu(R0, n, nf, d0, ptr(agg), nc, An);
                aggTot = agg;
            }
            else
            {
                const Csr& Ain = pass == 0 ? lv.A : Ac;
                nc = pairwise(Ain.rows(), Ain.n, ptr(Ain.d), agg);
                galerkinCsr(Ain, ptr(agg), nc, An);
                if (pass == 0) aggTot = agg;
                else
                {
                    int* at = ptr(aggTot);
                    const int* ag = ptr(agg);
                    for_n(long(aggTot.size()), [=] HD (long i) { at[i] = ag[at[i]]; });
                }
            }
            Ac = std::move(An);
        }
        if (nc > 0.9*lv.n) break;
        lv.agg = std::move(aggTot);
        lv.nc = nc;
        members(lv);
        L.emplace_back();
        Level& nx = L.back();
        nx.n = nc;
        nx.A = std::move(Ac);
        l1Dinv(nx.A.rows(), nx.n, ptr(nx.A.d), nx.Dinv);
    }
    for (Level& lv : L)
    {
        lv.x.resize(lv.n); lv.b.resize(lv.n); lv.r.resize(lv.n); lv.e.resize(lv.n); lv.t.resize(lv.n);
    }
    coarseInv.clear();
    if (L.size() > 1 && L.back().n <= denseMax) denseInverse(L.back().A);
    else if (L.size() == 1 && n <= denseMax)
    {
        // tiny system: no hierarchy, Jacobi smoothing only
    }
}

void AmgHierarchy::op(int k, const double* x, double* y) const
{
    if (k == 0) apply(R0, L[0].n, d0, x, y);
    else apply(L[k].A.rows(), L[k].n, ptr(L[k].A.d), x, y);
}

void AmgHierarchy::cycle(int k, const double* b, double* x)
{
    Level& lv = L[k];
    const long n = lv.n;
    const double* Di = ptr(lv.Dinv);
    double* r = ptr(lv.r);
    double* t = ptr(lv.t);
    auto smooth = [&](int sweeps) {
        for (int s = 0; s < sweeps; ++s)
        {
            op(k, x, t);
            for_n(n, [=] HD (long i) { x[i] += Di[i]*(b[i] - t[i]); });
        }
    };
    if (k == int(L.size()) - 1)
    {
        if (!coarseInv.empty())
        {
            const double* A = ptr(coarseInv);
            for_n(n, [=] HD (long i) {
                double s = 0.0;
                for (long j = 0; j < n; ++j) s += A[i*n + j]*b[j];
                x[i] = s;
            });
        }
        else
        {
            for_n(n, [=] HD (long i) { x[i] = Di[i]*b[i]; });
            smooth(49);
        }
        return;
    }
    for_n(n, [=] HD (long i) { x[i] = Di[i]*b[i]; });
    smooth(nsweep - 1);
    op(k, x, t);
    for_n(n, [=] HD (long i) { r[i] = b[i] - t[i]; });
    Level& nx = L[k + 1];
    {
        const int* mp = ptr(lv.mPtr);
        const int* mi = ptr(lv.mIdx);
        double* bc = ptr(nx.b);
        for_n(nx.n, [=] HD (long c) {
            double s = 0.0;
            for (int q = mp[c]; q < mp[c + 1]; ++q) s += r[mi[q]];
            bc[c] = s;
        });
    }
    cycle(k + 1, ptr(nx.b), ptr(nx.x));
    double* e = ptr(lv.e);
    {
        const int* agg = ptr(lv.agg);
        const double* xc = ptr(nx.x);
        for_n(n, [=] HD (long i) { e[i] = xc[agg[i]]; });
    }
    op(k, e, t);
    const double eAe = dotv(n, e, t);
    const double alpha = eAe > 0 ? dotv(n, e, r)/eAe : 1.0;
    for_n(n, [=] HD (long i) { x[i] += alpha*e[i]; });
    smooth(nsweep);
}

AmgCache::AmgCache() = default;
AmgCache::~AmgCache() = default;

// ============================================================================================ Krylov solvers
namespace
{

struct System
{
    const DMesh* m;
    const Ldu* M;
    LduRows R;
    long n, nf;
    dvec<double> d, b;                         // effective diagonal and source
    void A(const double* x, double* y) const { apply(R, n, ptr(d), x, y); }
};

bool converged(double res, double res0, const SolverCtl& c)
{
    return res < c.tol || (c.relTol > 1e-20 && res < c.relTol*res0);
}

} // namespace

SolveInfo solve(const DMesh& m, const Ldu& M, double* x, const SolverCtl& ctl, AmgCache* cache)
{
    System S;
    S.m = &m;
    S.M = &M;
    S.R = LduRows{view(m), M.c0, M.f0, M.f0 + M.nf, ptr(M.upper), ptr(M.lower)};
    S.n = M.nc;
    S.nf = M.nf;
    S.d.resize(S.n);
    S.b.resize(S.n);
    effective(M, ptr(S.d), ptr(S.b));
    const long n = S.n;
    double* d = ptr(S.d);
    double* b = ptr(S.b);

    // normFactor (lduMatrix::solver::normFactor)
    dvec<double> rv(n), t1(n);
    double* r = ptr(rv);
    double* t = ptr(t1);
    S.A(x, r);                                 // r = A x
    sumA(m, M, t);                             // row sums
    const double xRef = sum_n(n, [=] HD (long i) { return x[i]; })/double(n);
    const double nf = sum_n(n, [=] HD (long i) {
        const double tr = t[i]*xRef;
        return fabs(r[i] - tr) + fabs(b[i] - tr);
    }) + 1e-20;
    for_n(n, [=] HD (long i) { r[i] = b[i] - r[i]; });
    SolveInfo info;
    info.res0 = sumMag(n, r)/nf;
    info.res = info.res0;
    if (ctl.minIter <= 0 && info.res0 < ctl.tol) return info;

    // preconditioner
    AmgHierarchy* amg = nullptr;
    std::unique_ptr<AmgHierarchy> local;
    if (ctl.precond == SolverCtl::AMG)
    {
        if (cache && cache->h && cache->left > 0)
        {
            cache->left--;
            amg = cache->h.get();
            amg->setFine(S.R, d);
        }
        else
        {
            local.reset(new AmgHierarchy());
            local->build(S.R, n, S.nf, d);
            amg = local.get();
            if (cache)
            {
                cache->h = std::move(local);
                cache->left = cache->reuseEvery;
                amg = cache->h.get();
            }
        }
    }
    auto precond = [&](const double* in, double* out) {
        if (amg) { amg->cycle(0, in, out); return; }
        if (ctl.precond == SolverCtl::JACOBI)
            for_n(n, [=] HD (long i) { out[i] = in[i]/(d[i] != 0.0 ? d[i] : 1.0); });
        else
            for_n(n, [=] HD (long i) { out[i] = in[i]; });
    };

    int it = 0;
    if (ctl.method == SolverCtl::SMOOTH)
    {
        // point-Jacobi sweeps, residual checked every 5 sweeps
        while ((it < ctl.maxIter && !converged(info.res, info.res0, ctl)) || it < ctl.minIter)
        {
            for_n(n, [=] HD (long i) { x[i] += r[i]/d[i]; });
            S.A(x, t);
            for_n(n, [=] HD (long i) { r[i] = b[i] - t[i]; });
            ++it;
            if (it % 5 == 0 || it == ctl.maxIter) info.res = sumMag(n, r)/nf;
        }
        info.res = sumMag(n, r)/nf;
    }
    else if (ctl.method == SolverCtl::PCG)
    {
        dvec<double> zv(n), zo(n), pv(n), qv(n);
        double *z = ptr(zv), *zold = ptr(zo), *p = ptr(pv), *q = ptr(qv);
        precond(r, z);
        thrust::copy(thrust::device, zv.begin(), zv.end(), pv.begin());
        double rz = dotv(n, r, z);
        while ((it < ctl.maxIter && !converged(info.res, info.res0, ctl)) || it < ctl.minIter)
        {
            S.A(p, q);
            const double pq = dotv(n, p, q);
            if (pq == 0.0) break;
            const double alpha = rz/pq;
            for_n(n, [=] HD (long i) { x[i] += alpha*p[i]; r[i] -= alpha*q[i]; });
            ++it;
            info.res = sumMag(n, r)/nf;
            if (converged(info.res, info.res0, ctl) && it >= ctl.minIter) break;
            std::swap(zv, zo);
            z = ptr(zv);
            zold = ptr(zo);
            precond(r, z);
            const double rzn = dotv(n, r, z);
            const double beta = dmax((rzn - dotv(n, r, zold))/rz, 0.0);
            for_n(n, [=] HD (long i) { p[i] = z[i] + beta*p[i]; });
            rz = rzn;
        }
    }
    else
    {
        // BiCGStab
        dvec<double> rhv(rv), vv(n), pv(n), yv(n), sv(n), zv(n), tv(n);
        double *rh = ptr(rhv), *v = ptr(vv), *p = ptr(pv), *y = ptr(yv), *s = ptr(sv), *z = ptr(zv), *tt = ptr(tv);
        fill(vv, 0.0);
        fill(pv, 0.0);
        double rho = 1.0, alpha = 1.0, omega = 1.0;
        while ((it < ctl.maxIter && !converged(info.res, info.res0, ctl)) || it < ctl.minIter)
        {
            const double rhoN = dotv(n, rh, r);
            if (rhoN == 0.0) break;
            if (it == 0)
                thrust::copy(thrust::device, rv.begin(), rv.end(), pv.begin());
            else
            {
                const double beta = (rhoN/rho)*(alpha/omega);
                const double om = omega;
                for_n(n, [=] HD (long i) { p[i] = r[i] + beta*(p[i] - om*v[i]); });
            }
            precond(p, y);
            S.A(y, v);
            const double rv_ = dotv(n, rh, v);
            if (rv_ == 0.0) break;
            alpha = rhoN/rv_;
            const double al = alpha;
            for_n(n, [=] HD (long i) { s[i] = r[i] - al*v[i]; });
            ++it;
            info.res = sumMag(n, s)/nf;
            if (converged(info.res, info.res0, ctl) && it >= ctl.minIter)
            {
                for_n(n, [=] HD (long i) { x[i] += al*y[i]; r[i] = s[i]; });
                break;
            }
            precond(s, z);
            S.A(z, tt);
            const double t2 = dotv(n, tt, tt);
            omega = t2 > 0 ? dotv(n, tt, s)/t2 : 0.0;
            const double om = omega;
            for_n(n, [=] HD (long i) { x[i] += al*y[i] + om*z[i]; r[i] = s[i] - om*tt[i]; });
            rho = rhoN;
            info.res = sumMag(n, r)/nf;
            if (omega == 0.0) break;
        }
    }
    info.iters = it;
    return info;
}

} // namespace gc
