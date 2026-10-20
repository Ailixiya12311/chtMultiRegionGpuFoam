#include "energy.h"

namespace gc
{

namespace
{

constexpr double SMALL = 1e-15;

struct EV                                     // raw view for kernels
{
    MeshView g;
    int nR;
    const double *T, *rho, *rhoOld, *hOld, *cp, *e, *kappa, *alphat, *Su, *Sp, *Sx;
    const char* fluid;
    const double *rLimitK, *rPsi, *rRelax;
    const int* bKind;
    const double *bVal, *bGrad, *cpB, *eB, *kappaB, *alphatB;
    const double *Rlayer, *qIface, *kappaWall, *phi;
    double *bVf, *aB, *bB, *TbB, *GbB, *gA, *gB, *G, *TA, *TB, *h, *grad;
    int conv;
    double convK;
    bool bounded;
};

EV ev(Energy& E)
{
    EV v;
    v.g = view(*E.m);
    v.nR = E.nR;
    v.T = ptr(E.T); v.rho = ptr(E.rho); v.rhoOld = ptr(E.rhoOld); v.hOld = ptr(E.hOld); v.cp = ptr(E.cp);
    v.e = ptr(E.e); v.kappa = ptr(E.kappa); v.alphat = ptr(E.alphat); v.Su = ptr(E.Su); v.Sp = ptr(E.Sp); v.Sx = ptr(E.Sx);
    v.fluid = ptr(E.fluid);
    v.rLimitK = ptr(E.rLimitK); v.rPsi = ptr(E.rPsi); v.rRelax = ptr(E.rRelax);
    v.bKind = ptr(E.bKind);
    v.bVal = ptr(E.bVal); v.bGrad = ptr(E.bGrad); v.bVf = ptr(E.bVf); v.cpB = ptr(E.cpB); v.eB = ptr(E.eB);
    v.kappaB = ptr(E.kappaB); v.alphatB = ptr(E.alphatB);
    v.Rlayer = ptr(E.Rlayer); v.qIface = ptr(E.qIface); v.kappaWall = ptr(E.kappaWall); v.phi = ptr(E.phi);
    v.aB = ptr(E.aB); v.bB = ptr(E.bB); v.TbB = ptr(E.TbB); v.GbB = ptr(E.GbB);
    v.gA = ptr(E.gA); v.gB = ptr(E.gB); v.G = ptr(E.G); v.TA = ptr(E.TA); v.TB = ptr(E.TB);
    v.h = ptr(E.h); v.grad = ptr(E.grad);
    v.conv = E.conv; v.convK = E.convK; v.bounded = E.bounded;
    return v;
}

HD inline double keff(const EV& v, long c)
{
    return v.kappa[c] + (v.fluid[c] ? v.cp[c]*v.alphat[c] : 0.0);
}

// he value seen by cell c across the cell-face entry e (neighbour cell, interface patch value, boundary value)
HD inline double nbrValue(const EV& v, long c, int ent)
{
    const MeshView& g = v.g;
    if (ent >= 0)
    {
        if (ent < g.nIr) return v.h[g.nei[ent]];
        if (ent < g.nI) return v.cp[c]*v.TA[ent - g.nIr] + v.e[c];
        const int b = ent - g.nI;
        return v.cpB[b]*v.TbB[b] + v.eB[b];
    }
    const int f = -ent - 1;
    if (f < g.nIr) return v.h[g.own[f]];
    return v.cp[c]*v.TB[f - g.nIr] + v.e[c];
}

// face terms of a region face f (f < nIr): T-form off-diagonals and the explicit parts to owner / neighbour
struct FaceTerms
{
    double up, lo;                            // matrix coefficients
    double dO, dN;                            // diagonal contributions (owner row, neighbour row)
    double sO, sN;                            // source contributions
};

HD inline FaceTerms faceTerms(const EV& v, long f)
{
    const MeshView& g = v.g;
    const int o = g.own[f], n = g.nei[f];
    const double w = g.w[f];
    const double magSf = g.magSf[f];
    const double dc = g.nonOrthDC[f];
    FaceTerms t{0, 0, 0, 0, 0, 0};
    // diffusion: laplacian(alphaEff, he), alphaEff = keff/cp interpolated linearly
    const double gi = (w*keff(v, o)/v.cp[o] + (1 - w)*keff(v, n)/v.cp[n])*magSf*dc;
    t.up = -gi*v.cp[n];
    t.lo = -gi*v.cp[o];
    t.dO = gi*v.cp[o];
    t.dN = gi*v.cp[n];
    const double de = gi*(v.e[n] - v.e[o]);
    t.sO += de;
    t.sN -= de;
    // non-orthogonal correction (corrected / limited corrected psi) with the grad(he) scheme gradient
    const double psi = v.rPsi[g.cellRegion[o]];
    if (psi > 0)
    {
        const v3 go = ld3(v.grad, o), gn = ld3(v.grad, n);
        const v3 gf = w*go + (1 - w)*gn;
        double corr = dot(ld3(g.corrVec, f), gf);
        if (psi < 1)
        {
            const double sn = dc*(v.h[n] - v.h[o]);
            const double lim = dmin(psi*fabs(sn)/((1 - psi)*fabs(corr) + SMALL), 1.0);
            corr *= lim;
        }
        const double F = gi/dc*corr;
        t.sO += F;
        t.sN -= F;
    }
    // convection of he (fluid faces)
    const double F = v.phi[f];
    if (F != 0.0)
    {
        if (v.conv <= 1 || !v.fluid[o] || !v.fluid[n])
        {
            const double Fp = F > 0 ? F : 0.0, Fm = F > 0 ? 0.0 : F;
            t.dO += Fp*v.cp[o];
            t.dN -= Fm*v.cp[n];
            t.up += Fm*v.cp[n];
            t.lo -= Fp*v.cp[o];
            double eup = F >= 0 ? v.e[o] : v.e[n];
            if (v.conv == 1 && v.fluid[o] && v.fluid[n])   // linearUpwind grad(he)
            {
                const v3 cf = ld3(g.Cf, f);
                eup += F >= 0 ? dot(ld3(v.grad, o), cf - ld3(g.C, o)) : dot(ld3(v.grad, n), cf - ld3(g.C, n));
            }
            t.sO -= F*eup;
            t.sN += F*eup;
        }
        else                                  // linear (2) or limitedLinear k (3): implicit blended face weight
        {
            double lam = 1.0;
            if (v.conv == 3)
            {
                const v3 d = ld3(g.C, n) - ld3(g.C, o);
                const double gradcf = F > 0 ? dot(d, ld3(v.grad, o)) : dot(d, ld3(v.grad, n));
                const double dphi = v.h[n] - v.h[o];
                double r;
                if (fabs(gradcf) >= 1000.0*fabs(dphi))
                    r = 2000.0*(gradcf >= 0 ? 1.0 : -1.0)*(dphi >= 0 ? 1.0 : -1.0) - 1.0;
                else
                    r = 2.0*gradcf/dphi - 1.0;
                lam = dmax(dmin(2.0/dmax(v.convK, SMALL)*r, 1.0), 0.0);
            }
            const double wf = lam*w + (1.0 - lam)*(F >= 0 ? 1.0 : 0.0);
            t.dO += wf*F*v.cp[o];
            t.dN -= (1.0 - wf)*F*v.cp[n];
            t.up += (1.0 - wf)*F*v.cp[n];
            t.lo -= wf*F*v.cp[o];
            const double ef = wf*v.e[o] + (1.0 - wf)*v.e[n];
            t.sO -= F*ef;
            t.sN += F*ef;
        }
    }
    return t;
}

} // namespace

void Energy::init(const DMesh& mesh, int nRegions)
{
    m = &mesh;
    nR = nRegions;
    const long nC = m->nC, nB = m->nB, nIf = m->nIface, nF = long(m->nI) + m->nB;
    for (auto* a : {&T, &rho, &rhoOld, &hOld, &cp, &e, &kappa, &alphat, &Su, &Sp, &Sx, &h})
    {
        a->resize(nC);
        fill(*a, 0.0);
    }
    grad.resize(3*nC);
    fluid.resize(nC);
    fill(fluid, char(0));
    for (auto* a : {&rLimitK, &rPsi, &rRelax})
    {
        a->resize(nR);
        fill(*a, 0.0);
    }
    bKind.resize(nB);
    fill(bKind, int(B_ZEROGRAD));
    for (auto* a : {&bVal, &bGrad, &bVf, &cpB, &eB, &kappaB, &alphatB, &aB, &bB, &TbB, &GbB})
    {
        a->resize(nB);
        fill(*a, 0.0);
    }
    for (auto* a : {&Rlayer, &qIface, &gA, &gB, &G, &TA, &TB})
    {
        a->resize(nIf);
        fill(*a, 0.0);
    }
    kappaWall.resize(nIf);
    fill(kappaWall, -1.0);
    phi.resize(nF);
    fill(phi, 0.0);
    fS.resize(m->nIr);
    M.init(0, int(nC), 0, m->nI, m->nIr);
}

void Energy::boundaryState()
{
    EV v = ev(*this);
    const MeshView g = v.g;
    for_n(g.nB, [=] HD (long b) {
        const int c = g.own[g.nI + b];
        const double dc = g.dcB[b];
        double a, bb;
        switch (v.bKind[b])
        {
            case B_FIXED: a = v.bVal[b]; bb = 0.0; break;
            case B_FIXEDGRAD: a = v.bGrad[b]/dc; bb = 1.0; break;
            case B_MIXED: a = v.bVf[b]*v.bVal[b] + (1 - v.bVf[b])*v.bGrad[b]/dc; bb = 1 - v.bVf[b]; break;
            case B_INLETOUTLET:
            {
                const double vf = v.phi[g.nI + b] < 0.0 ? 1.0 : 0.0;
                v.bVf[b] = vf;
                a = vf*v.bVal[b];
                bb = 1.0 - vf;
                break;
            }
            default: a = 0.0; bb = 1.0;
        }
        v.aB[b] = a;
        v.bB[b] = bb;
        v.TbB[b] = a + bb*v.T[c];
        const double kb = v.kappaB[b] + (v.fluid[c] ? v.cpB[b]*v.alphatB[b] : 0.0);
        v.GbB[b] = kb/v.cpB[b]*g.magSf[g.nI + b]*dc;
    });
    for_n(g.nIface, [=] HD (long j) {
        const long f = g.nIr + j;
        const int o = g.own[f], n = g.nei[f];
        const double S = g.magSf[f];
        double ko = v.kappa[o], kn = v.kappa[n];
        const double kw = v.kappaWall[j];
        if (kw >= 0)
        {
            if (v.fluid[o]) ko = kw;
            else if (v.fluid[n]) kn = kw;
        }
        const double go = ko*g.dcIA[j]*S, gn = kn*g.dcIB[j]*S;
        const double R = v.Rlayer[j];
        const double Gs = 1.0/(1.0/go + 1.0/gn + R/S);
        v.gA[j] = go;
        v.gB[j] = gn;
        v.G[j] = Gs;
        if (R == 0.0)
        {
            const double t = (go*v.T[o] + gn*v.T[n] + v.qIface[j])/(go + gn);
            v.TA[j] = t;
            v.TB[j] = t;
        }
        else
        {
            const double Q = (v.T[o] - v.T[n])*Gs;
            v.TA[j] = v.T[o] - Q/go;
            v.TB[j] = v.T[n] + Q/gn;
        }
    });
}

void Energy::gradient()
{
    ProfScope prof("gradient");
    EV v = ev(*this);
    const MeshView g = v.g;
    for_n(g.nC, [=] HD (long c) { v.h[c] = v.cp[c]*v.T[c] + v.e[c]; });
    for_n(g.nC, [=] HD (long c) {
        const double hc = v.h[c];
        v3 gr = V3(0, 0, 0);
        double mx = hc, mn = hc;
        for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
        {
            const int ent = g.cfEnt[k];
            const double hv = nbrValue(v, c, ent);
            mx = dmax(mx, hv);
            mn = dmin(mn, hv);
            if (ent >= 0)
            {
                // region face (owner side): pVectors (h_nei - h_own); interface side A / boundary face: patch
                // vector (h_patch - h_cell)
                const v3 ls = ent < g.nI ? ld3(g.lsP, ent) : ld3(g.lsB, ent - g.nI);
                gr += ls*(hv - hc);
            }
            else
            {
                const int f = -ent - 1;
                if (f < g.nIr) gr -= ld3(g.lsN, f)*(hc - hv);   // lsGrad[nei] -= nVectors (h_nei - h_own)
                else gr += ld3(g.lsN, f)*(hv - hc);              // interface side B: patch vector
            }
        }
        // cellLimited (minmod) on the region stencil
        const double kL = v.rLimitK[g.cellRegion[c]];
        if (kL > 0)
        {
            double maxD = mx - hc, minD = mn - hc;
            if (kL < 1)
            {
                const double ext = (1.0/kL - 1.0)*(maxD - minD);
                maxD += ext;
                minD -= ext;
            }
            double lim = 1.0;
            const v3 Cc = ld3(g.C, c);
            for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
            {
                const int f = cfFace(g.cfEnt[k]);
                const double ex = dot(ld3(g.Cf, f) - Cc, gr);
                double r = 1.0;
                if (ex > SMALL) r = maxD/ex;
                else if (ex < -SMALL) r = minD/ex;
                lim = dmin(lim, r);
            }
            gr = gr*lim;
        }
        st3(v.grad, c, gr);
    });
}

void Energy::assemble(double dt)
{
    EV v = ev(*this);
    const MeshView g = v.g;
    LduView L = view(M);
    // off-diagonal coefficients, and the explicit part per face: faceTerms gives dO = -lo, dN = -up, sN = -sO
    // exactly (each pair is built with mirrored operations), so the cell rows below need only up, lo and sO
    std::unique_ptr<ProfScope> pf(new ProfScope("faces"));
    double* fs = ptr(fS);
    for_n(g.nIr, [=] HD (long f) {
        const FaceTerms t = faceTerms(v, f);
        L.upper[f] = t.up;
        L.lower[f] = t.lo;
        fs[f] = t.sO;
    });
    for_n(g.nIface, [=] HD (long j) {
        L.upper[g.nIr + j] = -v.G[j];
        L.lower[g.nIr + j] = -v.G[j];
    });
    // cell rows: interior diagonal / source, boundary(-like) parts
    pf.reset();
    ProfScope pc("cells");
    const double rdt = dt > 0 ? 1.0/dt : 0.0;
    for_n(g.nC, [=] HD (long c) {
        double D = 0, S = 0, bD = 0, bA = 0, bS = 0, divPhi = 0;
        for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
        {
            const int ent = g.cfEnt[k];
            if (ent >= 0 && ent < g.nIr)
            {
                D += -L.lower[ent];
                S += fs[ent];
                divPhi += v.phi[ent];
            }
            else if (ent < 0 && -ent - 1 < g.nIr)
            {
                const int f = -ent - 1;
                D += -L.upper[f];
                S += -fs[f];
                divPhi -= v.phi[f];
            }
            else if (ent >= 0 && ent < g.nI)  // interface, side A
            {
                const int j = ent - g.nIr;
                bD += v.G[j];
                bA += v.G[j];
                bS += v.qIface[j]*v.gA[j]/(v.gA[j] + v.gB[j]);
            }
            else if (ent < 0)                 // interface, side B
            {
                const int j = -ent - 1 - g.nIr;
                bD += v.G[j];
                bA += v.G[j];
                bS += v.qIface[j]*v.gB[j]/(v.gA[j] + v.gB[j]);
            }
            else                              // boundary face
            {
                const int b = ent - g.nI;
                const double Gb = v.GbB[b];
                double iC = Gb*(v.cp[c] - v.cpB[b]*v.bB[b]);
                double bC = Gb*(v.cpB[b]*v.aB[b] + v.eB[b] - v.e[c]);
                const double F = v.phi[g.nI + b];
                if (F != 0.0)
                {
                    iC += F*v.cpB[b]*v.bB[b];
                    bC -= F*(v.cpB[b]*v.aB[b] + v.eB[b]);
                    divPhi += F;
                }
                bD += iC;
                bA += fabs(iC);
                bS += bC;
            }
        }
        if (v.bounded && v.fluid[c])          // bounded Gauss: - Sp(div(phi)) he
        {
            D -= divPhi*v.cp[c];
            S += divPhi*v.e[c];
        }
        const double V = g.V[c];
        if (rdt > 0)
        {
            D += v.rho[c]*v.cp[c]*V*rdt;
            S += (v.rhoOld[c]*v.hOld[c] - v.rho[c]*v.e[c])*V*rdt;
        }
        // fvOptions on he: Su + Sp he (volume-integrated)
        S += v.Su[c] + v.Sp[c]*v.e[c] + v.Sx[c];
        D -= v.Sp[c]*v.cp[c];
        L.diag[c] = D;
        L.src[c] = S;
        L.bDiag[c] = bD;
        L.bAbs[c] = bA;
        L.bSrc[c] = bS;
    });
}

SolveInfo Energy::solve(double dt, const SolverCtl& ctl, bool relaxOn)
{
    ProfScope prof("energy");
    {
        ProfScope pa("assemble");
        { ProfScope pb("boundaryState"); boundaryState(); }
        gradient();
        assemble(dt);
    }
    EV v = ev(*this);
    const MeshView g = v.g;
    LduView L = view(M);
    if (relaxOn)
    {
        ProfScope pr("relax");
        // fvMatrix::relax on the he matrix of each region, carried out in T: the T matrix is the he matrix with
        // column j scaled by cp_j, so off-diagonal magnitudes and patch coefficients are divided back by cp
        double* T = ptr(this->T);
        for_n(g.nC, [=] HD (long c) {
            const double a = v.rRelax[g.cellRegion[c]];
            if (!(a > 0.0)) return;
            double sumOff = 0.0;
            for (int k = g.cfPtr[c]; k < g.cfPtr[c + 1]; ++k)
            {
                const int ent = g.cfEnt[k];
                if (ent >= 0 && ent < g.nIr) sumOff += fabs(L.upper[ent])/v.cp[g.nei[ent]];
                else if (ent < 0 && -ent - 1 < g.nIr) sumOff += fabs(L.lower[-ent - 1])/v.cp[g.own[-ent - 1]];
            }
            const double cpc = v.cp[c];
            const double D0 = L.diag[c]/cpc;
            double D = dmax(fabs(D0 + L.bAbs[c]/cpc), sumOff);
            D = D/a - L.bDiag[c]/cpc;
            const double Dt = D*cpc;
            L.src[c] += (Dt - L.diag[c])*T[c];
            L.diag[c] = Dt;
        });
    }
    if (!fixCells.empty()) setValues(*m, M, fixCells, fixVals, ptr(T));
    const SolveInfo si = gc::solve(*m, M, ptr(T), ctl, ctl.precond == SolverCtl::AMG ? &amg : nullptr);
    boundaryState();
    return si;
}

void Energy::storeOld()
{
    copy(rho, rhoOld);
    const double *t = ptr(T), *c = ptr(cp), *ee = ptr(e);
    double* ho = ptr(hOld);
    for_n(m->nC, [=] HD (long i) { ho[i] = c[i]*t[i] + ee[i]; });
}

} // namespace gc
