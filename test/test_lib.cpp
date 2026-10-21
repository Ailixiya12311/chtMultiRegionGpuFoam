// Library self-test without OpenFOAM:
//  1. two-material bar (two regions, one interface), steady conduction, fixed ends: exact linear profiles;
//     1b. the same with a linearised surface exchange and a heat flow at the interface face
//  2. n^3 cube with a heat source, steady: PCG+Jacobi vs PCG+AMG vs BiCGStab+AMG (same solution, iterations, time)
//  3. ShadowMap synthetic tests (cube on ground & room with glazing window)
#include <chrono>
#include <cmath>
#include <cstdio>

#include "api.h"
#include "building.h"

using namespace gc;

// structured box mesh, cells ordered x fastest; regions split along x at cell index ix = splitX
static MeshIn box(int nx, int ny, int nz, double L, int splitX)
{
    MeshIn m;
    const double dx = L/nx, dy = L/ny, dz = L/nz;
    auto cid = [&](int i, int j, int k) { return i + nx*(j + ny*k); };
    // cells: region A (i < splitX) first, then region B
    std::vector<int> map(nx*ny*nz);
    int c = 0;
    const int nreg = splitX > 0 && splitX < nx ? 2 : 1;
    for (int r = 0; r < nreg; ++r)
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    if ((nreg == 1) || (r == 0 ? i < splitX : i >= splitX)) map[cid(i, j, k)] = c++;
    m.nC = c;
    m.V.assign(m.nC, dx*dy*dz);
    m.C.resize(3*m.nC);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const int q = map[cid(i, j, k)];
                m.C[3*q] = (i + 0.5)*dx; m.C[3*q + 1] = (j + 0.5)*dy; m.C[3*q + 2] = (k + 0.5)*dz;
            }
    auto regionOf = [&](int i) { return nreg == 1 ? 0 : (i < splitX ? 0 : 1); };
    struct F { int o, n; double S[3], Cf[3]; double dc; };
    std::vector<F> faces[2], iface;
    struct Bf { int o; double S[3], Cf[3]; double dc; int patch; };
    std::vector<Bf> bf[2];
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const int p = map[cid(i, j, k)];
                const double cx = (i + 0.5)*dx, cy = (j + 0.5)*dy, cz = (k + 0.5)*dz;
                if (i + 1 < nx)
                {
                    F f{p, map[cid(i + 1, j, k)], {dy*dz, 0, 0}, {(i + 1)*dx, cy, cz}, 1/dx};
                    if (regionOf(i) == regionOf(i + 1)) faces[regionOf(i)].push_back(f); else iface.push_back(f);
                }
                if (j + 1 < ny) faces[regionOf(i)].push_back(F{p, map[cid(i, j + 1, k)], {0, dx*dz, 0}, {cx, (j + 1)*dy, cz}, 1/dy});
                if (k + 1 < nz) faces[regionOf(i)].push_back(F{p, map[cid(i, j, k + 1)], {0, 0, dx*dy}, {cx, cy, (k + 1)*dz}, 1/dz});
                if (i == 0) bf[regionOf(i)].push_back(Bf{p, {-dy*dz, 0, 0}, {0, cy, cz}, 2/dx, 0});
                if (i == nx - 1) bf[regionOf(i)].push_back(Bf{p, {dy*dz, 0, 0}, {L, cy, cz}, 2/dx, 1});
                if (j == 0) bf[regionOf(i)].push_back(Bf{p, {0, -dx*dz, 0}, {cx, 0, cz}, 2/dy, 2});
                if (j == ny - 1) bf[regionOf(i)].push_back(Bf{p, {0, dx*dz, 0}, {cx, L, cz}, 2/dy, 2});
                if (k == 0) bf[regionOf(i)].push_back(Bf{p, {0, 0, -dx*dy}, {cx, cy, 0}, 2/dz, 2});
                if (k == nz - 1) bf[regionOf(i)].push_back(Bf{p, {0, 0, dx*dy}, {cx, cy, L}, 2/dz, 2});
            }
    int c0 = 0, f0 = 0;
    for (int r = 0; r < nreg; ++r)
    {
        MeshIn::Region R;
        R.name = r == 0 ? "A" : "B";
        R.c0 = c0;
        R.nC = r == 0 ? (nreg == 1 ? m.nC : splitX*ny*nz) : m.nC - c0;
        c0 += R.nC;
        R.f0 = f0;
        R.nI = int(faces[r].size());
        f0 += R.nI;
        m.regions.push_back(R);
    }
    auto pushI = [&](const F& f) {
        m.own.push_back(f.o); m.nei.push_back(f.n);
        for (int d = 0; d < 3; ++d) { m.Sf.push_back(f.S[d]); m.Cf.push_back(f.Cf[d]); }
        m.w.push_back(0.5); m.nonOrthDC.push_back(f.dc);
        for (int d = 0; d < 3; ++d) { m.corrVec.push_back(0); m.lsP.push_back(0); m.lsN.push_back(0); }
    };
    for (int r = 0; r < nreg; ++r) for (auto& f : faces[r]) pushI(f);
    for (auto& f : iface) { pushI(f); m.dcI_A.push_back(2/dx); m.dcI_B.push_back(2/dx); }
    m.nIface = int(iface.size());
    m.nI = int(m.nei.size());
    if (m.nIface) m.pairs.push_back(MeshIn::Pair{0, 1, "toB", "toA", m.nI - m.nIface, m.nIface});
    int b0 = 0;
    for (int r = 0; r < nreg; ++r)
    {
        m.regions[r].b0 = b0;
        for (int pt = 0; pt < 3; ++pt)
        {
            MeshIn::Patch P;
            P.region = m.regions[r].name;
            P.name = pt == 0 ? "left" : (pt == 1 ? "right" : "sides");
            P.type = "patch";
            P.b0 = b0;
            for (auto& f : bf[r])
                if (f.patch == pt)
                {
                    m.own.push_back(f.o);
                    for (int d = 0; d < 3; ++d) { m.Sf.push_back(f.S[d]); m.Cf.push_back(f.Cf[d]); m.lsB.push_back(0); }
                    m.dcB.push_back(f.dc);
                    ++b0;
                }
            P.n = b0 - P.b0;
            m.patches.push_back(P);
        }
        m.regions[r].nB = b0 - m.regions[r].b0;
    }
    m.nB = b0;
    return m;
}

static double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void setup(Engine& E, const MeshIn& m, const std::vector<double>& kap, double Tleft, double Tright)
{
    const int nC = m.nC, nB = m.nB, nR = int(m.regions.size());
    std::vector<double> one(nC, 1.0), zero(nC, 0.0), k(nC);
    for (int r = 0; r < nR; ++r)
        for (int c = m.regions[r].c0; c < m.regions[r].c0 + m.regions[r].nC; ++c) k[c] = kap[r];
    E.put(A::T, std::vector<double>(nC, 0.5*(Tleft + Tright)));
    E.put(A::rho, one);
    E.put(A::cp, one);
    E.put(A::e, zero);
    E.put(A::kappa, k);
    E.put(A::rLimitK, std::vector<double>(nR, 0.0));
    E.put(A::rPsi, std::vector<double>(nR, 0.0));
    E.put(A::rRelax, std::vector<double>(nR, 0.0));
    std::vector<int> kind(nB, B_ZEROGRAD);
    std::vector<double> val(nB, 0.0), ones(nB, 1.0), kb(nB);
    for (auto& P : m.patches)
    {
        int r = P.region == "A" ? 0 : 1;
        for (int b = P.b0; b < P.b0 + P.n; ++b)
        {
            kb[b] = kap[r];
            if (P.name == "left") { kind[b] = B_FIXED; val[b] = Tleft; }
            if (P.name == "right") { kind[b] = B_FIXED; val[b] = Tright; }
        }
    }
    E.putBKind(kind);
    E.put(A::bVal, val);
    E.put(A::cpB, ones);
    E.put(A::eB, std::vector<double>(nB, 0.0));
    E.put(A::kappaB, kb);
    E.energyInit();
}

struct PatchGrid
{
    std::vector<v3> Cf, nrm;
    std::vector<double> A;
    void append(const PatchGrid& o)
    {
        Cf.insert(Cf.end(), o.Cf.begin(), o.Cf.end());
        nrm.insert(nrm.end(), o.nrm.begin(), o.nrm.end());
        A.insert(A.end(), o.A.begin(), o.A.end());
    }
};

static PatchGrid makeGrid(double x0, double x1, double y0, double y1, double h, char axis, double c, double normal)
{
    PatchGrid g;
    for (double a = x0 + 0.5*h; a < x1 - 1e-12; a += h)
        for (double b = y0 + 0.5*h; b < y1 - 1e-12; b += h)
        {
            if (axis == 'z') { g.Cf.push_back(v3{a, b, c}); g.nrm.push_back(v3{0, 0, normal}); }
            else if (axis == 'y') { g.Cf.push_back(v3{a, c, b}); g.nrm.push_back(v3{0, normal, 0}); }
            else { g.Cf.push_back(v3{c, a, b}); g.nrm.push_back(v3{normal, 0, 0}); }
            g.A.push_back(h*h);
        }
    return g;
}

int main()
{
    int fails = 0;
    // ---------------------------------------------------------------- 1. bar
    {
        const int nx = 10;
        MeshIn m = box(nx, 1, 1, 1.0, nx/2);
        Engine E(m);
        setup(E, m, {1.0, 4.0}, 100.0, 0.0);
        SolverCtl s;
        s.method = SolverCtl::PBICGSTAB;
        s.precond = SolverCtl::JACOBI;
        s.tol = 1e-14;
        s.maxIter = 1000;
        SolveInfo inf = E.energySolve(0.0, s, false);
        std::vector<double> T = E.get(A::T);
        const double q = 100.0/(0.5/1.0 + 0.5/4.0);
        double err = 0;
        for (int c = 0; c < m.nC; ++c)
        {
            const double x = m.C[3*c];
            const double ex = x < 0.5 ? 100.0 - q*x/1.0 : 100.0 - q*0.5 - q*(x - 0.5)/4.0;
            err = std::max(err, std::fabs(T[c] - ex));
        }
        std::vector<double> TA, TB, gA, gB;
        E.interfaceState(TA, TB, gA, gB);
        printf("bar: res0 %.3e res %.3e iters %d  max |T - exact| = %.3e  interface T %.10f (exact 20)\n",
               inf.res0, inf.res, inf.iters, err, TA[0]);
        if (err > 1e-9 || std::fabs(TA[0] - 20.0) > 1e-9) { printf("  FAIL\n"); ++fails; }
    }
    // ---------------------------------------------------------------- 1b. bar with surface exchange at the interface
    // the interface face also exchanges w (Tref - Ts) and takes q0 (Energy::wRad, as the interior longwave star node):
    // Ts = (2*100 + w Tref + q0)/(2 + 8 + w), the two sides conducting k/0.5 = 2 and 8; w well above both
    {
        const int nx = 10;
        MeshIn m = box(nx, 1, 1, 1.0, nx/2);
        Engine E(m);
        setup(E, m, {1.0, 4.0}, 100.0, 0.0);
        const double w = 25.0, Tref = 60.0, q0 = 30.0;
        E.put(A::wRad, std::vector<double>(m.nIface, w));
        E.put(A::qIface, std::vector<double>(m.nIface, q0 + w*Tref));
        SolverCtl s;
        s.method = SolverCtl::PBICGSTAB;
        s.precond = SolverCtl::JACOBI;
        s.tol = 1e-14;
        s.maxIter = 1000;
        SolveInfo inf = E.energySolve(0.0, s, false);
        std::vector<double> T = E.get(A::T);
        const double Ts = (200.0 + w*Tref + q0)/(10.0 + w);
        double err = 0;
        for (int c = 0; c < m.nC; ++c)
        {
            const double x = m.C[3*c];
            const double ex = x < 0.5 ? 100.0 - (100.0 - Ts)*x/0.5 : Ts*(1.0 - (x - 0.5)/0.5);
            err = std::max(err, std::fabs(T[c] - ex));
        }
        std::vector<double> TA, TB, gA, gB;
        E.interfaceState(TA, TB, gA, gB);
        printf("bar + surface exchange: iters %d  max |T - exact| = %.3e  interface T %.10f (exact %.10f)\n",
               inf.iters, err, TA[0], Ts);
        if (err > 1e-9 || std::fabs(TA[0] - Ts) > 1e-9 || std::fabs(TB[0] - Ts) > 1e-9) { printf("  FAIL\n"); ++fails; }
    }
    // ---------------------------------------------------------------- 2. cube
    {
        const int n = 48;
        MeshIn m = box(n, n, n, 1.0, 0);
        std::vector<double> Su(m.nC, 1.0*m.V[0]*100.0);
        std::vector<double> ref;
        const char* names[] = {"PCG+Jacobi", "PCG+AMG", "BiCGStab+AMG", "PCG+DIC", "BiCGStab+DILU", "smooth GS",
                               "smooth symGS"};
        const SolverCtl::Method meth[] = {SolverCtl::PCG, SolverCtl::PCG, SolverCtl::PBICGSTAB, SolverCtl::PCG,
                                          SolverCtl::PBICGSTAB, SolverCtl::SMOOTH, SolverCtl::SMOOTH};
        const SolverCtl::Precond pre[] = {SolverCtl::JACOBI, SolverCtl::AMG, SolverCtl::AMG, SolverCtl::DILU,
                                          SolverCtl::DILU, SolverCtl::NONE, SolverCtl::NONE};
        for (int t = 0; t < 7; ++t)
        {
            Engine E(m);
            setup(E, m, {1.0}, 1.0, 0.0);
            E.put(A::Su, Su);
            SolverCtl s;
            s.method = meth[t];
            s.precond = pre[t];
            s.smoother = t == 6 ? SolverCtl::SYMGS : SolverCtl::GS;
            s.tol = 1e-12;
            s.maxIter = t >= 5 ? 100000 : 5000;
            const double t0 = now();
            SolveInfo inf = E.energySolve(0.0, s, false);
            const double t1 = now();
            std::vector<double> T = E.get(A::T);
            double d = 0;
            if (ref.empty()) ref = T;
            for (int c = 0; c < m.nC; ++c) d = std::max(d, std::fabs(T[c] - ref[c]));
            printf("cube %d^3 %-13s res0 %.3e res %.3e iters %5d  %.3f s  max diff to PCG+Jacobi %.2e\n", n, names[t],
                   inf.res0, inf.res, inf.iters, t1 - t0, d);
            if (d > 1e-7) { printf("  FAIL\n"); ++fails; }
        }
    }
    // ---------------------------------------------------------------- 3. ShadowMap
    {
        const double ps = 0.05;
        ShadowMap sm{ps};
        // test 3a: 2 m cube on 10x10 ground, sun 45 deg from south
        PatchGrid g0 = makeGrid(0, 10, 0, 10, 0.1, 'z', 0.0, +1);
        PatchGrid all;
        for (size_t i = 0; i < g0.A.size(); ++i)
            if (!(g0.Cf[i].x > 4 && g0.Cf[i].x < 6 && g0.Cf[i].y > 4 && g0.Cf[i].y < 6))
            {
                all.Cf.push_back(g0.Cf[i]); all.nrm.push_back(g0.nrm[i]); all.A.push_back(g0.A[i]);
            }
        const int ng = int(all.A.size());
        all.append(makeGrid(4, 6, 4, 6, 0.1, 'z', 2.0, +1));
        all.append(makeGrid(4, 6, 0, 2, 0.1, 'y', 4.0, -1));
        all.append(makeGrid(4, 6, 0, 2, 0.1, 'y', 6.0, +1));
        all.append(makeGrid(4, 6, 0, 2, 0.1, 'x', 4.0, -1));
        all.append(makeGrid(4, 6, 0, 2, 0.1, 'x', 6.0, +1));
        Samples S;
        S.build(all.Cf, all.nrm, all.A, ps);
        std::vector<ShadowGroup> grp(1);
        grp[0].S = &S; grp[0].nrm = &all.nrm; grp[0].glass = false;
        const double s45 = std::sqrt(0.5);
        sm.compute(v3{0.0, -s45, s45}, grp);
        double shArea = 0, topLit = 0, sLit = 0, nLit = 0;
        for (int i = 0; i < ng; ++i) shArea += (1.0 - grp[0].vis[i])*all.A[i];
        for (int i = 0; i < 400; ++i)
        {
            topLit += grp[0].vis[ng + i]/400.0;
            sLit += grp[0].vis[ng + 400 + i]/400.0;
            nLit += grp[0].vis[ng + 800 + i]/400.0;
        }
        printf("shadow1: ground shadow %.3f m2 (expected 4.000), top %.3f, south %.3f, north %.3f\n",
               shArea, topLit, sLit, nLit);
        if (std::fabs(shArea - 4.0) > 0.05 || topLit < 0.99 || sLit < 0.99 || nLit > 1e-9) { printf("  FAIL\n"); ++fails; }

        // test 3b: room with south window
        const double rad40 = 40.0*3.14159265358979323846/180.0;
        const v3 sun{0.0, -std::cos(rad40), std::sin(rad40)};
        PatchGrid sw = makeGrid(0, 4, 0, 3, 0.05, 'y', -0.2, -1);
        PatchGrid ext, glass;
        for (size_t i = 0; i < sw.A.size(); ++i)
        {
            const bool w = sw.Cf[i].x > 1 && sw.Cf[i].x < 3 && sw.Cf[i].z > 1 && sw.Cf[i].z < 2;
            PatchGrid& dst = w ? glass : ext;
            dst.Cf.push_back(sw.Cf[i]); dst.nrm.push_back(sw.nrm[i]); dst.A.push_back(sw.A[i]);
        }
        ext.append(makeGrid(-0.2, 4.2, -0.2, 4.2, 0.1, 'z', 3.2, +1));
        PatchGrid floor = makeGrid(0, 4, 0, 4, 0.05, 'z', 0.0, +1);
        PatchGrid nwall = makeGrid(0, 4, 0, 3, 0.05, 'y', 4.0, -1);
        PatchGrid is0 = makeGrid(0, 4, 0, 3, 0.05, 'y', 0.0, +1);
        PatchGrid inner = floor;
        inner.append(nwall);
        for (size_t i = 0; i < is0.A.size(); ++i)
            if (!(is0.Cf[i].x > 1 && is0.Cf[i].x < 3 && is0.Cf[i].z > 1 && is0.Cf[i].z < 2))
            {
                inner.Cf.push_back(is0.Cf[i]); inner.nrm.push_back(is0.nrm[i]); inner.A.push_back(is0.A[i]);
            }
        Samples Se, Sg, Si;
        Se.build(ext.Cf, ext.nrm, ext.A, ps);
        Sg.build(glass.Cf, glass.nrm, glass.A, ps);
        Si.build(inner.Cf, inner.nrm, inner.A, ps);
        std::vector<ShadowGroup> g2(3);
        g2[0] = {&Se, &ext.nrm, false, {}, {}};
        g2[1] = {&Sg, &glass.nrm, true, {}, {}};
        g2[2] = {&Si, &inner.nrm, false, {}, {}};
        sm.compute(sun, g2);
        double inc = 0, land = 0, glMean = 0;
        const double cosG = dot(glass.nrm[0], sun);
        for (size_t i = 0; i < glass.A.size(); ++i) { inc += g2[1].vis[i]*glass.A[i]*cosG; glMean += g2[1].vis[i]; }
        glMean /= double(glass.A.size());
        for (size_t i = 0; i < inner.A.size(); ++i) land += g2[2].glf[i]*inner.A[i]*dot(inner.nrm[i], sun);
        printf("shadow2: glass lit %.3f, beam through glass %.4f, landing inside %.4f (ratio %.3f)\n",
               glMean, inc, land, land/inc);
        if (glMean < 0.99 || std::fabs(land/inc - 1.0) > 0.05) { printf("  FAIL\n"); ++fails; }
    }
    printf(fails ? "TESTS FAILED: %d\n" : "ALL TESTS PASSED\n", fails);
    return fails ? 1 : 0;
}

