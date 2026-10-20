#include "linsolve.h"

#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#if defined(GC_CUDA)
#include <cub/device/device_reduce.cuh>
#endif

#include <algorithm>
#include <cstdlib>
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
    dvec<float> vf, df;                       // single-precision copies for the AMG V-cycle (toFloat)
    CsrRows rows() const { return CsrRows{ptr(rp), ptr(ci), ptr(v)}; }
    void toFloat()
    {
        vf.resize(v.size());
        df.resize(d.size());
        const double *a = ptr(v), *b = ptr(d);
        float *af = ptr(vf), *bf = ptr(df);
        for_n(long(v.size()), [=] HD (long k) { af[k] = float(a[k]); });
        for_n(long(d.size()), [=] HD (long i) { bf[i] = float(b[i]); });
    }
};

template<class Rows> void apply(const Rows& R, long n, const double* d, const double* x, double* y)
{
    for_n(n, [=] HD (long i) {
        double s = d[i]*x[i];
        forRow(R, i, [&](long j, double a) { s += a*x[j]; });
        y[i] = s;
    });
}

#if defined(GC_CUDA)
// Csr y = d x + (off-diagonal) x with W threads per row (a warp holds whole groups; the sum of each group is
// reduced with shuffles): the entries of a row are read by neighbouring threads, coalesced, where one thread per
// row would stride through memory. Coarse AMG levels have 10-30 entries per row. Entries with ci == row (the
// diagonal kept in Galerkin products) are skipped: d carries the diagonal.
template<int W, class T> __global__ void csrApplyW(int n, const int* rp, const int* ci, const T* v, const T* d,
                                                   const T* x, T* y)
{
    const long t = long(blockIdx.x)*blockDim.x + threadIdx.x;
    const long row = t/W;
    const int lane = int(t%W);
    T s = 0;
    if (row < n)
        for (int k = rp[row] + lane; k < rp[row + 1]; k += W)
        {
            const int j = ci[k];
            if (j != row) s += v[k]*x[j];
        }
    for (int o = W/2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o, W);
    if (row < n && lane == 0) y[row] = d[row]*x[row] + s;
}

template<int W, class T> void csrApplyLaunch(int n, const int* rp, const int* ci, const T* v, const T* d,
                                             const T* x, T* y)
{
    const int block = 256;
    const long threads = long(n)*W;
    csrApplyW<W, T><<<int((threads + block - 1)/block), block>>>(n, rp, ci, v, d, x, y);
}
#endif

template<class T> void csrApplyRaw(int n, long nnz, const int* rp, const int* ci, const T* v, const T* d,
                                   const T* x, T* y)
{
#if defined(GC_CUDA)
    const long perRow = nnz/std::max(n, 1);
    if (perRow > 24) csrApplyLaunch<32>(n, rp, ci, v, d, x, y);
    else if (perRow > 12) csrApplyLaunch<16>(n, rp, ci, v, d, x, y);
    else if (perRow > 6) csrApplyLaunch<8>(n, rp, ci, v, d, x, y);
    else csrApplyLaunch<4>(n, rp, ci, v, d, x, y);
#else
    (void)nnz;
    for_n(n, [=] HD (long i) {
        T s = 0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] != i) s += v[k]*x[ci[k]];
        y[i] = d[i]*x[i] + s;
    });
#endif
}

void csrApply(const Csr& A, const double* d, const double* x, double* y)
{
#if defined(GC_CUDA)
    csrApplyRaw(A.n, long(A.ci.size()), ptr(A.rp), ptr(A.ci), ptr(A.v), d, x, y);
#else
    apply(A.rows(), A.n, d, x, y);
#endif
}

// two dot products in one reduction (one device -> host round trip instead of two)
struct D2
{
    double a, b;
};
struct D2Plus
{
    HD D2 operator()(const D2& p, const D2& q) const { return D2{p.a + q.a, p.b + q.b}; }
};
template<class T> void dot2(long n, const T* x, const T* y, const T* u, const T* w, double& xy, double& uw)
{
    const D2 r = thrust::transform_reduce(thrust::device, thrust::counting_iterator<long>(0),
                                          thrust::counting_iterator<long>(n),
                                          [=] HD (long i) { return D2{double(x[i])*double(y[i]), double(u[i])*double(w[i])}; },
                                          D2{0.0, 0.0}, D2Plus());
    xy = r.a;
    uw = r.b;
}

// ---------------------------------------------------------------------- single-precision CSR rows (AMG V-cycle)
// One pass over the rows of a CSR in float: s = sum of the off-diagonal a_ij x_j (entries with ci == row skipped,
// the diagonal comes with the epilogue), then ep(row, s) finishes the row: y = A x, r = b - A x, or a Jacobi update
// written to a second vector, so that a smoothing sweep is one kernel instead of a product and an update.
struct EpAx
{
    const float *d, *x;
    float* y;
    HD void operator()(long i, float s) const { y[i] = d[i]*x[i] + s; }
};
struct EpRes
{
    const float *d, *x, *b;
    float* r;
    HD void operator()(long i, float s) const { r[i] = b[i] - (d[i]*x[i] + s); }
};
struct EpJac
{
    const float *d, *x, *b, *di;
    float* xn;
    HD void operator()(long i, float s) const { xn[i] = x[i] + di[i]*(b[i] - (d[i]*x[i] + s)); }
};

#if defined(GC_CUDA)
template<int W, class Ep> __global__ void csrRowsW(int n, const int* rp, const int* ci, const float* v,
                                                   const float* x, Ep ep)
{
    const long t = long(blockIdx.x)*blockDim.x + threadIdx.x;
    const long row = t/W;
    const int lane = int(t%W);
    float s = 0;
    if (row < n)
        for (int k = rp[row] + lane; k < rp[row + 1]; k += W)
        {
            const int j = ci[k];
            if (j != row) s += v[k]*x[j];
        }
    for (int o = W/2; o > 0; o /= 2) s += __shfl_down_sync(0xffffffffu, s, o, W);
    if (row < n && lane == 0) ep(row, s);
}

template<int W, class Ep> void csrRowsLaunch(int n, const int* rp, const int* ci, const float* v, const float* x,
                                             Ep ep)
{
    const int block = 256;
    const long threads = long(n)*W;
    csrRowsW<W, Ep><<<int((threads + block - 1)/block), block>>>(n, rp, ci, v, x, ep);
}
#endif

template<class Ep> void csrRows(int n, long nnz, const int* rp, const int* ci, const float* v, const float* x, Ep ep)
{
    if (n <= 0) return;
#if defined(GC_CUDA)
    const long perRow = nnz/std::max(n, 1);
    if (perRow > 24) csrRowsLaunch<32>(n, rp, ci, v, x, ep);
    else if (perRow > 12) csrRowsLaunch<16>(n, rp, ci, v, x, ep);
    else if (perRow > 6) csrRowsLaunch<8>(n, rp, ci, v, x, ep);
    else csrRowsLaunch<4>(n, rp, ci, v, x, ep);
#else
    (void)nnz;
    for_n(n, [=] HD (long i) {
        float s = 0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] != i) s += v[k]*x[ci[k]];
        ep(i, s);
    });
#endif
}

// out = {sum e t, sum e r} (double sums of float vectors) into device memory, without a host round trip: the
// V-cycle's kernels read it there. tmp: CUB work space, kept (grows only).
struct D2Fn
{
    const float *e, *t, *r;
    HD D2 operator()(long i) const { return D2{double(e[i])*double(t[i]), double(e[i])*double(r[i])}; }
};
void dot2Dev(long n, const float* e, const float* t, const float* r, D2* out, dvec<char>& tmp)
{
#if defined(GC_CUDA)
    auto it = thrust::make_transform_iterator(thrust::counting_iterator<long>(0), D2Fn{e, t, r});
    size_t bytes = 0;
    cub::DeviceReduce::Reduce(nullptr, bytes, it, out, int(n), D2Plus(), D2{0.0, 0.0});
    if (tmp.size() < bytes) tmp.resize(bytes);
    cub::DeviceReduce::Reduce(static_cast<void*>(ptr(tmp)), bytes, it, out, int(n), D2Plus(), D2{0.0, 0.0});
#else
    (void)tmp;
    *out = thrust::transform_reduce(thrust::device, thrust::counting_iterator<long>(0),
                                    thrust::counting_iterator<long>(n), D2Fn{e, t, r}, D2{0.0, 0.0}, D2Plus());
#endif
}

// ------------------------------------------------------------------------------- multi-colour DILU, Gauss-Seidel
// OpenFOAM's DILU / DIC preconditioners and Gauss-Seidel smoothers sweep the cells in index order, which is
// sequential. Here the cells are coloured so that no two neighbours share a colour and swept colour by colour,
// all cells of one colour in parallel: the same formulas (DILU: rD_i = a_ii - sum_{j before i} a_ij a_ji / rD_j,
// forward and backward substitution; GS: x_i = (b_i - sum_j a_ij x_j) / a_ii with the newest x_j) in the colour
// order instead of the index order. Iteration counts are therefore close to, not equal to, OpenFOAM's.
//
// With the colouring the system is also stored as a CSR in colour order: row r is cell order[r], the rows of one
// colour are contiguous, and in each row the entries of earlier colours come first (mid[r] splits them from the
// later ones). Its values are gathered from upper / lower once per solve; the sweeps and A x then read contiguous
// rows instead of going through the cell-face lists. solve() runs such a system in that order throughout: the
// vectors are permuted to the rows (ci holds row indices), so a colour reads and writes contiguous vector entries
// instead of cells spread over the mesh, and x is permuted back at the end.

// fn(j, s) for the off-diagonal entries of row i, s: upper index (>= 0) or -(lower index) - 1 of a_ij
template<class Fn> HD inline void forRowIdx(const LduRows& R, long i, Fn fn)
{
    const long c = R.c0 + i;
    for (int k = R.g.cfPtr[c]; k < R.g.cfPtr[c + 1]; ++k)
    {
        const int e = R.g.cfEnt[k];
        if (e >= 0)
        {
            if (e < R.f0 || e >= R.f1 || e >= R.g.nI) continue;
            fn(long(R.g.nei[e] - R.c0), e - R.f0);
        }
        else
        {
            const int f = -e - 1;
            if (f < R.f0 || f >= R.f1) continue;
            fn(long(R.g.own[f] - R.c0), -(f - R.f0) - 1);
        }
    }
}

struct Colouring
{
    long n = 0, nnz = 0;
    int nCol = 0;
    dvec<int> col;                             // colour of each (local) cell
    dvec<int> order;                           // cells sorted by colour (by index within a colour)
    std::vector<long> start;                   // host: colour c is rows start[c] .. start[c + 1]
    dvec<int> rp, mid, ci, src;                // colour-ordered CSR (ci: row of the neighbour); src as forRowIdx
    dvec<double> v;                            // a_ij of the current matrix (fillValues)
};

HD inline unsigned cellRank(long i)
{
    unsigned long long h = (unsigned long long)i*0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27;
    return unsigned(h >> 32);
}

// Jones-Plassmann colouring with the smallest free colour: each round every uncoloured cell whose (hashed) rank
// beats all its uncoloured neighbours takes the smallest colour none of its coloured neighbours has. A round
// reads the colours of the previous round only, so the result does not depend on thread scheduling.
void buildColouring(const LduRows& R, long n, Colouring& C)
{
    C.n = n;
    C.col.assign(n, -1);
    dvec<int> prev(n, -1);
    int* col = ptr(C.col);
    const int* pc = ptr(prev);
    for (int round = 0; ; ++round)
    {
        if (round == 10000) fatal("multi-colour ordering: no progress after %d rounds", round);
        thrust::copy(thrust::device, C.col.begin(), C.col.end(), prev.begin());
        for_n(n, [=] HD (long i) {
            if (pc[i] >= 0) return;
            const unsigned wi = cellRank(i);
            bool top = true;
            unsigned long long used = 0;
            forRow(R, i, [&](long j, double) {
                const int cj = pc[j];
                if (cj >= 0)
                {
                    if (cj < 64) used |= 1ull << cj;
                    return;
                }
                const unsigned wj = cellRank(j);
                if (wj > wi || (wj == wi && j > i)) top = false;
            });
            if (!top) return;
            int c = 0;
            while (c < 63 && ((used >> c) & 1ull)) ++c;
            col[i] = c;
        });
        const double left = sum_n(n, [=] HD (long i) { return col[i] < 0 ? 1.0 : 0.0; });
        if (left == 0.0) break;
    }
    C.nCol = int(max_n(n, [=] HD (long i) { return double(col[i]); }, -1.0)) + 1;
    if (C.nCol >= 64) fatal("multi-colour ordering: %d colours (at most 63)", C.nCol);
    C.order.resize(n);
    thrust::sequence(thrust::device, C.order.begin(), C.order.end());
    {
        dvec<int> key(C.col);
        thrust::stable_sort_by_key(thrust::device, key.begin(), key.end(), C.order.begin());
        dvec<int> st(C.nCol + 1);
        thrust::lower_bound(thrust::device, key.begin(), key.end(), thrust::counting_iterator<int>(0),
                            thrust::counting_iterator<int>(C.nCol + 1), st.begin());
        const std::vector<int> h = to_host(st);
        C.start.assign(h.begin(), h.end());
    }
    // colour-ordered CSR pattern
    const int* ord = ptr(C.order);
    C.rp.assign(n + 1, 0);
    C.mid.resize(n);
    int* rp = ptr(C.rp);
    for_n(n, [=] HD (long r) {
        int k = 0;
        forRowIdx(R, ord[r], [&](long, int) { ++k; });
        rp[r + 1] = k;
    });
    thrust::inclusive_scan(thrust::device, C.rp.begin() + 1, C.rp.end(), C.rp.begin() + 1);
    C.nnz = C.rp.back();
    C.ci.resize(C.nnz);
    C.src.resize(C.nnz);
    C.v.resize(C.nnz);
    int* ci = ptr(C.ci);
    int* sr = ptr(C.src);
    int* md = ptr(C.mid);
    for_n(n, [=] HD (long r) {
        const long i = ord[r];
        const int c = col[i];
        int k = rp[r];
        for (int pass = 0; pass < 2; ++pass)
        {
            if (pass == 1) md[r] = k;
            forRowIdx(R, i, [&](long j, int s) {
                if ((col[j] < c) != (pass == 0)) return;
                ci[k] = int(j);
                sr[k] = s;
                ++k;
            });
        }
    });
    // neighbour cells -> their rows
    {
        dvec<int> pos(n);
        int* ps = ptr(pos);
        for_n(n, [=] HD (long r) { ps[ord[r]] = int(r); });
        for_n(C.nnz, [=] HD (long k) { ci[k] = ps[ci[k]]; });
    }
    if (prof_on()) std::printf("    multi-colour ordering: %ld cells, %d colours\n", n, C.nCol);
}

Colouring& colouring(const DMesh& m, const LduRows& R, long n, long nf)
{
    const std::array<long, 4> key{long(R.c0), n, long(R.f0), nf};
    for (const auto& e : m.colourings)
        if (e.first == key) return *static_cast<Colouring*>(e.second.get());
    ProfScope ps("colouring");
    std::shared_ptr<Colouring> c = std::make_shared<Colouring>();
    buildColouring(R, n, *c);
    m.colourings.emplace_back(key, c);
    return *c;
}

// a_ij of the current matrix into the colour-ordered CSR
void fillValues(Colouring& C, const LduRows& R)
{
    const int* sr = ptr(C.src);
    const double *up = R.up, *lo = R.lo;
    double* v = ptr(C.v);
    for_n(C.nnz, [=] HD (long k) {
        const int q = sr[k];
        v[k] = q >= 0 ? up[q] : lo[-q - 1];
    });
}

// for each colour (ascending, or descending with rev), all rows r of that colour in parallel: fn(r)
template<class Fn> void forColourRows(const Colouring& C, bool rev, Fn fn)
{
    for (int q = 0; q < C.nCol; ++q)
    {
        const int c = rev ? C.nCol - 1 - q : q;
        const long a = C.start[c], cnt = C.start[c + 1] - a;
        for_n(cnt, [=] HD (long k) { fn(a + k); });
    }
}

// The vectors below are in row (colour) order.
// y = A x (d: diagonal) from the colour-ordered CSR
void colourApply(const Colouring& C, const double* d, const double* x, double* y)
{
    const int *rp = ptr(C.rp), *ci = ptr(C.ci);
    const double* v = ptr(C.v);
    for_n(C.n, [=] HD (long r) {
        double s = d[r]*x[r];
        for (int k = rp[r]; k < rp[r + 1]; ++k) s += v[k]*x[ci[k]];
        y[r] = s;
    });
}

// DILU (DIC for a symmetric matrix): rD = 1 / (a_ii - sum over earlier-coloured j of a_ij a_ji rD_j)
void diluFactor(const Colouring& C, const LduRows& R, const double* d, double* rD)
{
    const int *rp = ptr(C.rp), *md = ptr(C.mid), *ci = ptr(C.ci), *sr = ptr(C.src);
    const double* v = ptr(C.v);
    const double *up = R.up, *lo = R.lo;
    forColourRows(C, false, [=] HD (long r) {
        double s = d[r];
        for (int k = rp[r]; k < md[r]; ++k)
        {
            const int q = sr[k];
            s -= v[k]*(q >= 0 ? lo[q] : up[-q - 1])*rD[ci[k]];
        }
        rD[r] = 1.0/s;
    });
}

// w = M^-1 r: forward then backward substitution in colour order
void diluApply(const Colouring& C, const double* rD, const double* rr, double* w)
{
    const int *rp = ptr(C.rp), *md = ptr(C.mid), *ci = ptr(C.ci);
    const double* v = ptr(C.v);
    forColourRows(C, false, [=] HD (long r) {
        double s = rr[r];
        for (int k = rp[r]; k < md[r]; ++k) s -= v[k]*w[ci[k]];
        w[r] = rD[r]*s;
    });
    forColourRows(C, true, [=] HD (long r) {
        double s = 0.0;
        for (int k = md[r]; k < rp[r + 1]; ++k) s += v[k]*w[ci[k]];
        w[r] -= rD[r]*s;
    });
}

// one Gauss-Seidel sweep over the colours (symmetric: forward, then backward)
void gsSweep(const Colouring& C, const double* d, const double* b, double* x, bool sym)
{
    const int *rp = ptr(C.rp), *ci = ptr(C.ci);
    const double* v = ptr(C.v);
    auto relaxRow = [=] HD (long r) {
        double s = b[r];
        for (int k = rp[r]; k < rp[r + 1]; ++k) s -= v[k]*x[ci[k]];
        x[r] = s/d[r];
    };
    forColourRows(C, false, relaxRow);
    if (sym) forColourRows(C, true, relaxRow);
}

// ------------------------------------------------------------------------------------------- aggregation
// Symmetric priority of edge (i, j) for breaking ties between equally strong neighbours: both ends rank the edge
// the same, so on a uniform mesh the locally dominant edges are mutual choices. (Breaking ties by the smaller
// index instead makes every cell point the same way along the mesh, chains rather than pairs: a uniform 3-D
// region then coarsened by only 1.4x per two passes.)
HD inline unsigned edgeRank(long i, long j)
{
    unsigned long long a = (unsigned long long)(i < j ? i : j), b = (unsigned long long)(i < j ? j : i);
    unsigned long long h = a*0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull + (a << 6) + (a >> 2));
    h ^= h >> 31;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27;
    return unsigned(h >> 32);
}

// One pass of pairwise matching: in up to 5 rounds every unmatched cell picks, among its unmatched neighbours with
// a strong connection (s_ij >= strongFrac * the cell's largest s; 0.85, GPUCHT_AMG_STRONG overrides), the one
// with the highest edgeRank; mutual
// picks pair. Leftovers join their strongest aggregated neighbour, the rest stay single. Strength
// s_ij = -a_ij / sqrt(|d_i d_j|) > 0. (Picking the single strongest neighbour, cht/amg.py, makes every cell of a
// graded mesh point towards the finer side: chains again.)
// V-cycle shape: 0 pre-smoothing sweeps (default 1, GPUCHT_AMG_PRE), 1 post-smoothing sweeps (default 2,
// GPUCHT_AMG_POST), 2 minimal-residual scaling of the coarse correction (default 1, GPUCHT_AMG_MR)
int cycleSetting(int which)
{
    static const int v[3] = {
        [] { const char* e = std::getenv("GPUCHT_AMG_PRE"); return e ? std::atoi(e) : 1; }(),
        [] { const char* e = std::getenv("GPUCHT_AMG_POST"); return e ? std::atoi(e) : 2; }(),
        [] { const char* e = std::getenv("GPUCHT_AMG_MR"); return e ? std::atoi(e) : 1; }()};
    return v[which];
}

double strongFracSetting()
{
    static const double v = [] { const char* e = std::getenv("GPUCHT_AMG_STRONG"); return e ? std::atof(e) : 0.85; }();
    return v;
}
template<class Rows> int pairwise(const Rows& R, long n, const double* d, dvec<int>& aggv)
{
    aggv.resize(n);
    fill(aggv, -1);
    int* agg = ptr(aggv);
    dvec<int> bestv(n), flagv(n), idsv(n);
    int* best = ptr(bestv);
    int* flag = ptr(flagv);
    int* ids = ptr(idsv);
    const double strongFrac = strongFracSetting();
    int nc = 0;
    for (int round = 0; round < 5; ++round)
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
            unsigned br = 0;
            forRow(R, i, [&](long j, double a) {
                if (agg[j] >= 0) return;
                const double s = -a/sqrt(fabs(d[i]*d[j]));
                if (!(s > 0.0 && s >= strongFrac*smax)) return;
                const unsigned rk = edgeRank(i, j);
                if (bj < 0 || rk > br || (rk == br && j < bj)) { bj = int(j); br = rk; }
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
        unsigned br = 0;
        forRow(R, i, [&](long j, double a) {
            if (agg[j] < 0) return;
            const double s = -a/sqrt(fabs(d[i]*d[j]));
            if (!(s > 0.0 && s >= smax*(1 - 1e-12))) return;
            const unsigned rk = edgeRank(i, j);
            if (bj < 0 || rk > br || (rk == br && j < bj)) { bj = int(j); br = rk; }
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

template<class Rows> void l1Dinv(const Rows& R, long n, const double* d, dvec<float>& Dinv)
{
    Dinv.resize(n);
    float* di = ptr(Dinv);
    for_n(n, [=] HD (long i) {
        double l1 = fabs(d[i]);
        forRow(R, i, [&](long, double a) { l1 += fabs(a); });
        const double m = dmax(l1, fabs(d[i]));
        di[i] = float(m > 0.0 ? 1.0/m : 1.0);
    });
}

// Fixed Galerkin pattern: keys (coarse row * nc + coarse col) of the fine entries -> CSR pattern of `out` and the
// gather map src / seg (coarse value q is the sum of fine entries src[p], p in [seg[q], seg[q + 1]), in ascending
// fine-entry order), so that new fine values give the coarse ones without sorting again.
void galerkinPattern(dvec<long long>& keys, int nc, Csr& out, dvec<int>& src, dvec<int>& seg)
{
    const long ne = long(keys.size());
    src.resize(ne);
    thrust::sequence(thrust::device, src.begin(), src.end());
    thrust::stable_sort_by_key(thrust::device, keys.begin(), keys.end(), src.begin());
    dvec<int> head(ne), pos(ne);
    const long long* k = ptr(keys);
    int* h = ptr(head);
    for_n(ne, [=] HD (long p) { h[p] = (p == 0 || k[p] != k[p - 1]) ? 1 : 0; });
    thrust::exclusive_scan(thrust::device, head.begin(), head.end(), pos.begin());
    const long nnz = ne > 0 ? long(pos.back()) + long(head.back()) : 0;
    out.n = nc;
    out.ci.resize(nnz);
    out.v.resize(nnz);
    out.d.resize(nc);
    seg.resize(nnz + 1);
    dvec<int> rows(nnz);
    {
        const int* ps = ptr(pos);
        int* sg = ptr(seg);
        int* ci = ptr(out.ci);
        int* rw = ptr(rows);
        const long long ncl = nc;
        for_n(ne, [=] HD (long p) {
            if (!h[p]) return;
            const int q = ps[p];
            sg[q] = int(p);
            rw[q] = int(k[p]/ncl);
            ci[q] = int(k[p]%ncl);
        });
        thrust::fill(thrust::device, seg.begin() + nnz, seg.end(), int(ne));
    }
    out.rp.resize(nc + 1);
    thrust::lower_bound(thrust::device, rows.begin(), rows.end(), thrust::counting_iterator<int>(0),
                        thrust::counting_iterator<int>(nc + 1), out.rp.begin());
    keys.clear();
    keys.shrink_to_fit();
}

// fine entry e of an Ldu level (layout of galerkinLdu: diagonal, then upper, then lower) / of a Csr level
struct LduVal
{
    long n, nf;
    const double *d, *up, *lo;
    HD double operator()(int e) const { return e < n ? d[e] : (e < n + nf ? up[e - n] : lo[e - n - nf]); }
};
struct CsrVal
{
    const double* v;
    HD double operator()(int e) const { return v[e]; }
};

template<class Val> void galerkinValues(const dvec<int>& srcv, const dvec<int>& segv, Val fv, Csr& A)
{
    const int* src = ptr(srcv);
    const int* seg = ptr(segv);
    double* v = ptr(A.v);
    for_n(long(A.v.size()), [=] HD (long q) {
        double s = 0.0;
        for (int p = seg[q]; p < seg[q + 1]; ++p) s += fv(src[p]);
        v[q] = s;
    });
    const int* rp = ptr(A.rp);
    const int* ci = ptr(A.ci);
    double* d = ptr(A.d);
    for_n(A.n, [=] HD (long i) {
        double s = 0.0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) if (ci[k] == i) s += v[k];
        d[i] = s;
    });
}

// coarse keys of the fine entries for aggregation agg (Ldu level: galerkinLdu layout; Csr level: CSR order)
void coarseKeysLdu(const LduRows& R, long n, long nf, const int* agg, int nc, dvec<long long>& keys)
{
    keys.resize(n + 2*nf);
    long long* k = ptr(keys);
    const long long ncl = nc;
    for_n(n, [=] HD (long i) { k[i] = agg[i]*ncl + agg[i]; });
    const MeshView g = R.g;
    const int c0 = R.c0, f0 = R.f0;
    for_n(nf, [=] HD (long j) {
        const long f = f0 + j;
        const int o = agg[g.own[f] - c0], nb = agg[g.nei[f] - c0];
        k[n + j] = o*ncl + nb;
        k[n + nf + j] = nb*ncl + o;
    });
}

void coarseKeysCsr(const Csr& A, const int* agg, int nc, dvec<long long>& keys)
{
    keys.resize(A.ci.size());
    long long* k = ptr(keys);
    const int* rp = ptr(A.rp);
    const int* ci = ptr(A.ci);
    const long long ncl = nc;
    for_n(A.n, [=] HD (long i) {
        for (int q = rp[i]; q < rp[i + 1]; ++q) k[q] = agg[i]*ncl + agg[ci[q]];
    });
}

} // namespace

// ============================================================================================ AMG hierarchy
// The V-cycle runs in single precision (the Krylov iteration around it stays double): float copies of the level
// operators, float vectors, and each Jacobi sweep fused with its matrix product (csrRows). The minimal-residual
// scaling of the coarse correction reduces into device memory, so a cycle needs no host round trip.
struct AmgHierarchy
{
    struct Level
    {
        long n = 0;
        Csr A;                                 // coarse levels (level 0: the Ldu)
        dvec<float> Dinv;                      // l1-Jacobi
        dvec<int> agg;                         // to the next level
        int nc = 0;
        dvec<int> gSrc, gSeg;                  // fixed Galerkin gather map to the next level's operator
        dvec<int> mPtr, mIdx;                  // members of each coarse cell (restriction by gather)
        dvec<float> x, xn, b, r, e, t;
    };
    std::vector<Level> L;
    dvec<double> coarseInv;                    // dense inverse (column-major: thread i reads A[j n + i],
                                               // coalesced) of the coarsest level, or empty
    // level 0 operator
    LduRows R0;
    const double* d0 = nullptr;
    long nf0 = 0;
    // level 0 also as CSR for the products (the most applied operator): pattern from the Ldu once, values gathered
    // from upper / lower each solve (double for the Krylov products, float for the V-cycle); src0: upper index
    // (>= 0) or -(lower index) - 1
    dvec<int> rp0, ci0, src0;
    dvec<double> v0;
    dvec<float> v0f, d0f;
    dvec<D2> d2;                               // per level: e.Ae, e.r of the coarse correction
    dvec<char> redTmp;
    void buildCsr0();
    void fillCsr0();

    static constexpr int coarseSize = 200;      // coarsest level: dense inverse (cost ~ n^3 on the host)
    static constexpr int denseMax = 1000;

    // aggregation (pairwise matching on this matrix) and the coarse patterns, once
    void build(const LduRows& R, long n, long nf, const double* d);
    // new fine matrix, same aggregation: coarse values by the gather maps, smoothers, coarsest inverse
    void refresh(const LduRows& R, const double* d);
    void setFine(const LduRows& R, const double* d);
    void coarseValues(int k);
    void op(int k, const double* x, double* y) const;           // level 0, double (Krylov products)
    template<class Ep> void rowsF(int k, const float* x, Ep ep) const;
    // V-cycle on level k from L[k].b; returns the vector holding the result (L[k].x or L[k].xn)
    float* cycle(int k);
    // out = M^-1 in (one V-cycle)
    void precondition(const double* in, double* out);
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
    std::vector<double> invT(size_t(n)*n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) invT[size_t(j)*n + i] = inv[size_t(i)*n + j];
    coarseInv = to_dev(invT);
}

void AmgHierarchy::setFine(const LduRows& R, const double* d)
{
    R0 = R;
    d0 = d;
    l1Dinv(R0, L[0].n, d0, L[0].Dinv);
    fillCsr0();
    d0f.resize(L[0].n);
    float* df = ptr(d0f);
    for_n(L[0].n, [=] HD (long i) { df[i] = float(d[i]); });
}

void AmgHierarchy::buildCsr0()
{
    const long n = L[0].n;
    const LduRows R = R0;
    rp0.assign(n + 1, 0);
    int* rp = ptr(rp0);
    for_n(n, [=] HD (long i) {
        int c = 0;
        forRow(R, i, [&](long, double) { ++c; });
        rp[i + 1] = c;
    });
    thrust::inclusive_scan(thrust::device, rp0.begin() + 1, rp0.end(), rp0.begin() + 1);
    const int nnz = rp0.back();
    ci0.resize(nnz);
    src0.resize(nnz);
    v0.resize(nnz);
    v0f.resize(nnz);
    int* ci = ptr(ci0);
    int* sr = ptr(src0);
    for_n(n, [=] HD (long i) {
        int k = rp[i];
        const long c = R.c0 + i;
        for (int q = R.g.cfPtr[c]; q < R.g.cfPtr[c + 1]; ++q)
        {
            const int e = R.g.cfEnt[q];
            if (e >= 0)
            {
                if (e < R.f0 || e >= R.f1 || e >= R.g.nI) continue;
                ci[k] = R.g.nei[e] - R.c0;
                sr[k++] = e - R.f0;
            }
            else
            {
                const int f = -e - 1;
                if (f < R.f0 || f >= R.f1) continue;
                ci[k] = R.g.own[f] - R.c0;
                sr[k++] = -(f - R.f0) - 1;
            }
        }
    });
}

void AmgHierarchy::fillCsr0()
{
    if (ci0.empty()) return;
    const int* sr = ptr(src0);
    const double* up = R0.up;
    const double* lo = R0.lo;
    double* v = ptr(v0);
    float* vf = ptr(v0f);
    for_n(long(v0.size()), [=] HD (long k) {
        const int q = sr[k];
        const double a = q >= 0 ? up[q] : lo[-q - 1];
        v[k] = a;
        vf[k] = float(a);
    });
}

void AmgHierarchy::build(const LduRows& R, long n, long nf, const double* d)
{
    L.clear();
    L.emplace_back();
    L[0].n = n;
    nf0 = nf;
    setFine(R, d);
    buildCsr0();
    fillCsr0();
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
                { ProfScope pp("pairwise"); nc = pairwise(R0, n, d0, agg); }
                ProfScope pg("galerkin");
                galerkinLdu(R0, n, nf, d0, ptr(agg), nc, An);
                aggTot = agg;
            }
            else
            {
                const Csr& Ain = pass == 0 ? lv.A : Ac;
                { ProfScope pp("pairwise"); nc = pairwise(Ain.rows(), Ain.n, ptr(Ain.d), agg); }
                // the last pass only aggregates: the operator comes from the fixed pattern below
                ProfScope pg("galerkin");
                if (pass < passes - 1) galerkinCsr(Ain, ptr(agg), nc, An);
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
        Csr Anext;
        {
            ProfScope pg("pattern");
            dvec<long long> keys;
            if (k == 0) coarseKeysLdu(R0, n, nf, ptr(lv.agg), nc, keys);
            else coarseKeysCsr(lv.A, ptr(lv.agg), nc, keys);
            galerkinPattern(keys, nc, Anext, lv.gSrc, lv.gSeg);
        }
        ProfScope pm("members");
        members(lv);
        L.emplace_back();
        Level& nx = L.back();
        nx.n = nc;
        nx.A = std::move(Anext);
        coarseValues(k);
        l1Dinv(nx.A.rows(), nx.n, ptr(nx.A.d), nx.Dinv);
        nx.A.toFloat();
    }
    for (Level& lv : L)
    {
        lv.x.resize(lv.n); lv.xn.resize(lv.n); lv.b.resize(lv.n); lv.r.resize(lv.n); lv.e.resize(lv.n);
        lv.t.resize(lv.n);
    }
    d2.resize(L.size());
    if (prof_on())
    {
        std::printf("    AMG hierarchy:");
        for (size_t k = 0; k < L.size(); ++k)
            std::printf(" L%zu %ld%s", k, L[k].n,
                        k ? (" (" + std::to_string(L[k].A.ci.size()/std::max<long>(L[k].n, 1)) + "/row)").c_str() : "");
        std::printf("\n");
    }
    coarseInv.clear();
    ProfScope pd("dense inverse");
    if (L.size() > 1 && L.back().n <= denseMax) denseInverse(L.back().A);
    else if (L.size() == 1 && n <= denseMax)
    {
        // tiny system: no hierarchy, Jacobi smoothing only
    }
}

void AmgHierarchy::coarseValues(int k)
{
    const Level& lv = L[k];
    Csr& A = L[k + 1].A;
    if (k == 0) galerkinValues(lv.gSrc, lv.gSeg, LduVal{lv.n, nf0, d0, R0.up, R0.lo}, A);
    else galerkinValues(lv.gSrc, lv.gSeg, CsrVal{ptr(lv.A.v)}, A);
}

void AmgHierarchy::refresh(const LduRows& R, const double* d)
{
    setFine(R, d);
    for (int k = 0; k + 1 < int(L.size()); ++k)
    {
        coarseValues(k);
        Level& nx = L[k + 1];
        l1Dinv(nx.A.rows(), nx.n, ptr(nx.A.d), nx.Dinv);
        nx.A.toFloat();
    }
    coarseInv.clear();
    ProfScope pd("dense inverse");
    if (L.size() > 1 && L.back().n <= denseMax) denseInverse(L.back().A);
}

void AmgHierarchy::op(int k, const double* x, double* y) const
{
    if (k == 0)
    {
        ProfScope ps("L0 op");
        if (!ci0.empty()) csrApplyRaw(int(L[0].n), long(ci0.size()), ptr(rp0), ptr(ci0), ptr(v0), d0, x, y);
        else apply(R0, L[0].n, d0, x, y);
    }
    else csrApply(L[k].A, ptr(L[k].A.d), x, y);
}

template<class Ep> void AmgHierarchy::rowsF(int k, const float* x, Ep ep) const
{
    if (k == 0)
    {
        ProfScope ps("L0 op");
        csrRows(int(L[0].n), long(ci0.size()), ptr(rp0), ptr(ci0), ptr(v0f), x, ep);
    }
    else csrRows(L[k].A.n, long(L[k].A.ci.size()), ptr(L[k].A.rp), ptr(L[k].A.ci), ptr(L[k].A.vf), x, ep);
}

float* AmgHierarchy::cycle(int k)
{
    Level& lv = L[k];
    const long n = lv.n;
    const float* Di = ptr(lv.Dinv);
    const float* d = k == 0 ? ptr(d0f) : ptr(lv.A.df);
    const float* b = ptr(lv.b);
    float* x = ptr(lv.x);
    float* xn = ptr(lv.xn);
    // Jacobi sweeps x -> xn -> x ..., the result in x
    auto smooth = [&](int sweeps) {
        for (int s = 0; s < sweeps; ++s)
        {
            rowsF(k, x, EpJac{d, x, b, Di, xn});
            std::swap(x, xn);
        }
    };
    if (k == int(L.size()) - 1)
    {
        if (!coarseInv.empty())
        {
            const double* A = ptr(coarseInv);
            for_n(n, [=] HD (long i) {
                double s = 0.0;
                for (long j = 0; j < n; ++j) s += A[j*n + i]*double(b[j]);
                x[i] = float(s);
            });
        }
        else
        {
            for_n(n, [=] HD (long i) { x[i] = Di[i]*b[i]; });
            smooth(49);
        }
        return x;
    }
    for_n(n, [=] HD (long i) { x[i] = Di[i]*b[i]; });
    smooth(cycleSetting(0));
    float* r = ptr(lv.r);
    rowsF(k, x, EpRes{d, x, b, r});
    Level& nx = L[k + 1];
    {
        const int* mp = ptr(lv.mPtr);
        const int* mi = ptr(lv.mIdx);
        float* bc = ptr(nx.b);
        for_n(nx.n, [=] HD (long c) {
            float s = 0;
            for (int q = mp[c]; q < mp[c + 1]; ++q) s += r[mi[q]];
            bc[c] = s;
        });
    }
    const float* xc;
    if (k == 0)
    {
        ProfScope pc("coarse levels");
        xc = cycle(k + 1);
    }
    else xc = cycle(k + 1);
    // coarse correction e = P xc, scaled by alpha = (e.r)/(e.Ae)
    const int* agg = ptr(lv.agg);
    if (cycleSetting(2))
    {
        float* e = ptr(lv.e);
        float* t = ptr(lv.t);
        for_n(n, [=] HD (long i) { e[i] = xc[agg[i]]; });
        rowsF(k, e, EpAx{d, e, t});
        D2* dd = ptr(d2) + k;
        dot2Dev(n, e, t, r, dd, redTmp);
        for_n(n, [=] HD (long i) {
            const float al = float(dd->a > 0 ? dd->b/dd->a : 1.0);
            x[i] += al*e[i];
        });
    }
    else for_n(n, [=] HD (long i) { x[i] += xc[agg[i]]; });
    smooth(cycleSetting(1));
    return x;
}

void AmgHierarchy::precondition(const double* in, double* out)
{
    const long n = L[0].n;
    float* b = ptr(L[0].b);
    for_n(n, [=] HD (long i) { b[i] = float(in[i]); });
    const float* x = cycle(0);
    for_n(n, [=] HD (long i) { out[i] = double(x[i]); });
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
    double *d, *b;                             // effective diagonal and source
    const Colouring* cs = nullptr;             // colour-ordered CSR with this matrix's values, if built: the
                                               // vectors are then in its row order
    const AmgHierarchy* amg = nullptr;         // AMG level 0 (CSR of this matrix), if built
    void A(const double* x, double* y) const
    {
        if (cs) colourApply(*cs, d, x, y);
        else if (amg) amg->op(0, x, y);
        else apply(R, n, d, x, y);
    }
};

// work slots of DMesh::workBuf used by solve()
enum Slot { W_D, W_B, W_R, W_T, W_1, W_2, W_3, W_4, W_5, W_6, W_7, W_8 };

bool converged(double res, double res0, const SolverCtl& c)
{
    return res < c.tol || (c.relTol > 1e-20 && res < c.relTol*res0);
}

} // namespace

SolveInfo solve(const DMesh& m, const Ldu& M, double* x0, const SolverCtl& ctl, AmgCache* cache)
{
    ProfScope prof("linsolve");
    double* x = x0;
    System S;
    S.m = &m;
    S.M = &M;
    S.R = LduRows{view(m), M.c0, M.f0, M.f0 + M.nf, ptr(M.upper), ptr(M.lower)};
    S.n = M.nc;
    S.nf = M.nf;
    const long n = S.n;
    std::unique_ptr<ProfScope> pn(new ProfScope("normFactor"));
    S.d = m.workBuf(W_D, n);
    S.b = m.workBuf(W_B, n);
    double* d = S.d;
    double* b = S.b;

    // normFactor (lduMatrix::solver::normFactor). One pass over the rows gives the effective diagonal and source,
    // r = A x and the row sums t (the arithmetic of effective(), amul() and sumA(), in their order)
    double* r = m.workBuf(W_R, n);
    double* t = m.workBuf(W_T, n);
    {
        const LduRows R = S.R;
        const double *dg = ptr(M.diag), *bd = ptr(M.bDiag), *sr = ptr(M.src), *bs = ptr(M.bSrc);
        for_n(n, [=] HD (long i) {
            const double di = dg[i] + bd[i];
            double ax = di*x[i], sa = di;
            forRow(R, i, [&](long j, double a) { ax += a*x[j]; sa += a; });
            d[i] = di;
            b[i] = sr[i] + bs[i];
            r[i] = ax;
            t[i] = sa;
        });
    }
    const double xRef = sum_n(n, [=] HD (long i) { return x[i]; })/double(n);
    const double nf = sum_n(n, [=] HD (long i) {
        const double tr = t[i]*xRef;
        return fabs(r[i] - tr) + fabs(b[i] - tr);
    }) + 1e-20;
    for_n(n, [=] HD (long i) { r[i] = b[i] - r[i]; });
    SolveInfo info;
    info.res0 = sumMag(n, r)/nf;
    info.res = info.res0;
    pn.reset();
    if (ctl.minIter <= 0 && info.res0 < ctl.tol) return info;

    // preconditioner
    AmgHierarchy* amg = nullptr;
    std::unique_ptr<AmgHierarchy> local;
    if (ctl.precond == SolverCtl::AMG)
    {
        if (cache && cache->h && cache->h->L[0].n != n) cache->h.reset();
        if (cache && cache->h && cache->left > 0)
        {
            cache->left--;
            amg = cache->h.get();
            amg->setFine(S.R, d);
        }
        else if (cache && cache->h)
        {
            ProfScope pb("amg refresh");
            amg = cache->h.get();
            amg->refresh(S.R, d);
            cache->left = cache->reuseEvery;
        }
        else
        {
            ProfScope pb("amg build");
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
    // multi-colour ordering (DILU preconditioner, smoothSolver) and the DILU factor of this matrix
    Colouring* col = nullptr;
    if (ctl.precond == SolverCtl::DILU || ctl.method == SolverCtl::SMOOTH)
    {
        col = &colouring(m, S.R, n, S.nf);
        ProfScope pv("fill values");
        fillValues(*col, S.R);
        S.cs = col;
        // the system to the colour-ordered rows: d -> t, b -> d, x -> b, r -> x0 (each buffer read before it is
        // overwritten; Gauss-Seidel does not use r); the old r buffer becomes the work vector t. x0 gets the
        // solution back at the end.
        const int* ord = ptr(col->order);
        double *dC = t, *bC = d, *xC = b, *rC = x0, *tC = r;
        for_n(n, [=] HD (long q) { dC[q] = d[ord[q]]; });
        for_n(n, [=] HD (long q) { bC[q] = b[ord[q]]; });
        for_n(n, [=] HD (long q) { xC[q] = x0[ord[q]]; });
        if (!(ctl.method == SolverCtl::SMOOTH && ctl.smoother != SolverCtl::DILUSM))
            for_n(n, [=] HD (long q) { rC[q] = r[ord[q]]; });
        S.d = d = dC;
        S.b = b = bC;
        x = xC;
        r = rC;
        t = tC;
    }
    S.amg = amg;
    double* rD = nullptr;
    if (ctl.precond == SolverCtl::DILU || (ctl.method == SolverCtl::SMOOTH && ctl.smoother == SolverCtl::DILUSM))
    {
        ProfScope pf("dilu factor");
        rD = m.workBuf(W_8, n);
        diluFactor(*col, S.R, d, rD);
    }
    auto precond = [&](const double* in, double* out) {
        if (amg) { amg->precondition(in, out); return; }
        if (rD) { diluApply(*col, rD, in, out); return; }
        if (ctl.precond == SolverCtl::JACOBI)
            for_n(n, [=] HD (long i) { out[i] = in[i]/(d[i] != 0.0 ? d[i] : 1.0); });
        else
            for_n(n, [=] HD (long i) { out[i] = in[i]; });
    };

    ProfScope pi("iterate");
    int it = 0;
    if (ctl.method == SolverCtl::SMOOTH)
    {
        // smoothSolver: nSweeps smoother sweeps between residual checks
        const int ns = std::max(ctl.nSweeps, 1);
        double* z = m.workBuf(W_1, n);
        while ((it < ctl.maxIter && !converged(info.res, info.res0, ctl)) || it < ctl.minIter)
        {
            for (int sw = 0; sw < ns; ++sw)
            {
                if (ctl.smoother == SolverCtl::DILUSM)
                {
                    // DILUSmoother: x += M^-1 (b - A x)
                    if (sw > 0)
                    {
                        S.A(x, t);
                        for_n(n, [=] HD (long i) { r[i] = b[i] - t[i]; });
                    }
                    diluApply(*col, rD, r, z);
                    for_n(n, [=] HD (long i) { x[i] += z[i]; });
                }
                else gsSweep(*col, d, b, x, ctl.smoother == SolverCtl::SYMGS);
            }
            it += ns;
            if (ctl.smoother == SolverCtl::DILUSM)
            {
                S.A(x, t);
                for_n(n, [=] HD (long i) { r[i] = b[i] - t[i]; });
                info.res = sumMag(n, r)/nf;
            }
            else
            {
                // Gauss-Seidel needs no residual vector: sum |b - A x| row by row in one reduction
                const int *rp = ptr(col->rp), *ci = ptr(col->ci);
                const double* v = ptr(col->v);
                info.res = sum_n(n, [=] HD (long q) {
                    double s = d[q]*x[q];
                    for (int k = rp[q]; k < rp[q + 1]; ++k) s += v[k]*x[ci[k]];
                    return fabs(b[q] - s);
                })/nf;
            }
        }
    }
    else if (ctl.method == SolverCtl::PCG)
    {
        double *z = m.workBuf(W_1, n), *zold = m.workBuf(W_2, n), *p = m.workBuf(W_3, n), *q = m.workBuf(W_4, n);
        { ProfScope pp("precond"); precond(r, z); }
        thrust::copy(pol(), z, z + n, p);
        double rz = dotv(n, r, z);
        while ((it < ctl.maxIter && !converged(info.res, info.res0, ctl)) || it < ctl.minIter)
        {
            { ProfScope ps("spmv"); S.A(p, q); }
            const double pq = dotv(n, p, q);
            if (pq == 0.0) break;
            const double alpha = rz/pq;
            for_n(n, [=] HD (long i) { x[i] += alpha*p[i]; r[i] -= alpha*q[i]; });
            ++it;
            info.res = sumMag(n, r)/nf;
            if (converged(info.res, info.res0, ctl) && it >= ctl.minIter) break;
            std::swap(z, zold);
            { ProfScope pp("precond"); precond(r, z); }
            double rzn, rzo;
            dot2(n, r, z, r, zold, rzn, rzo);
            const double beta = dmax((rzn - rzo)/rz, 0.0);
            for_n(n, [=] HD (long i) { p[i] = z[i] + beta*p[i]; });
            rz = rzn;
        }
    }
    else
    {
        // BiCGStab
        double *rh = m.workBuf(W_1, n), *v = m.workBuf(W_2, n), *p = m.workBuf(W_3, n), *y = m.workBuf(W_4, n);
        double *s = m.workBuf(W_5, n), *z = m.workBuf(W_6, n), *tt = m.workBuf(W_7, n);
        thrust::copy(pol(), r, r + n, rh);
        thrust::fill(pol(), v, v + n, 0.0);
        thrust::fill(pol(), p, p + n, 0.0);
        double rho = 1.0, alpha = 1.0, omega = 1.0;
        while ((it < ctl.maxIter && !converged(info.res, info.res0, ctl)) || it < ctl.minIter)
        {
            const double rhoN = dotv(n, rh, r);
            if (rhoN == 0.0) break;
            if (it == 0)
                thrust::copy(pol(), r, r + n, p);
            else
            {
                const double beta = (rhoN/rho)*(alpha/omega);
                const double om = omega;
                for_n(n, [=] HD (long i) { p[i] = r[i] + beta*(p[i] - om*v[i]); });
            }
            { ProfScope pp("precond"); precond(p, y); }
            { ProfScope ps("spmv"); S.A(y, v); }
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
            { ProfScope pp("precond"); precond(s, z); }
            { ProfScope ps("spmv"); S.A(z, tt); }
            double t2, ts;
            dot2(n, tt, tt, tt, s, t2, ts);
            omega = t2 > 0 ? ts/t2 : 0.0;
            const double om = omega;
            for_n(n, [=] HD (long i) { x[i] += al*y[i] + om*z[i]; r[i] = s[i] - om*tt[i]; });
            rho = rhoN;
            info.res = sumMag(n, r)/nf;
            if (omega == 0.0) break;
        }
    }
    info.iters = it;
    if (col)
    {
        const int* ord = ptr(col->order);
        const double* xC = x;
        for_n(n, [=] HD (long q) { x0[ord[q]] = xC[q]; });
    }
    return info;
}

} // namespace gc
