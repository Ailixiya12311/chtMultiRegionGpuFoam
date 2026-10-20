#include "flow.h"

namespace gc
{

namespace
{

constexpr double SMALL = 1e-15;

struct FW                                     // raw view for kernels
{
    MeshView g;
    int c0, nc, f0, nf, b0, nBr, nS, nIrG;
    const int *sFace, *sCell, *ifSlot;
    const double *sSf, *sCf, *sMagSf, *sDc, *sLs, *Rinv;
    const char* sSideA;
    double *U, *U0, *Ub, *Ub0, *p, *pPrev, *pAbs, *pAbsB, *rho, *rho0f, *rhoB, *rhoB0;
    double *phi, *phiB, *phi0, *phiB0, *K0, *pOld, *dpdt, *mu, *muB, *nut, *nutB;
    const double *eRho0, *eT0, *eBeta, *eRho0B, *eT0B, *eBetaB;
    const int *uKind, *pKind;
    double *uVal, *uVf;
    const char *uAssign, *uFixes;
    double *pVal, *pGrad;
    const double* p0Val;
    double *gU, *gP, *gRho, *src3, *bD3, *bS3, *bMin, *rAU, *HbyA, *HbyAB;
    double *rhorAUf, *rhorAUfB, *phiHbyA, *phiHbyAB, *phig, *phigB, *pCorr;
    const double *T, *TbB, *TA, *TB;
    double gx, gy, gz, ghRef;
};

FW fw(Flow& F)
{
    FW v;
    v.g = view(*F.m);
    v.c0 = F.c0; v.nc = F.nc; v.f0 = F.f0; v.nf = F.nf; v.b0 = F.b0; v.nBr = F.nBr; v.nS = F.nS;
    v.nIrG = F.m->nIr;
    v.sFace = ptr(F.sFace); v.sCell = ptr(F.sCell); v.ifSlot = ptr(F.ifSlot);
    v.sSf = ptr(F.sSf); v.sCf = ptr(F.sCf); v.sMagSf = ptr(F.sMagSf); v.sDc = ptr(F.sDc); v.sLs = ptr(F.sLs);
    v.Rinv = ptr(F.Rinv); v.sSideA = ptr(F.sSideA);
    v.U = ptr(F.U); v.U0 = ptr(F.U0); v.Ub = ptr(F.Ub); v.Ub0 = ptr(F.Ub0);
    v.p = ptr(F.p); v.pPrev = ptr(F.pPrev); v.pAbs = ptr(F.pAbs); v.pAbsB = ptr(F.pAbsB);
    v.rho = ptr(F.rho); v.rho0f = ptr(F.rho0f); v.rhoB = ptr(F.rhoB); v.rhoB0 = ptr(F.rhoB0);
    v.phi = ptr(F.phi); v.phiB = ptr(F.phiB); v.phi0 = ptr(F.phi0); v.phiB0 = ptr(F.phiB0);
    v.K0 = ptr(F.K0); v.pOld = ptr(F.pOld); v.dpdt = ptr(F.dpdt);
    v.mu = ptr(F.mu); v.muB = ptr(F.muB); v.nut = ptr(F.nut); v.nutB = ptr(F.nutB);
    v.eRho0 = ptr(F.eRho0); v.eT0 = ptr(F.eT0); v.eBeta = ptr(F.eBeta);
    v.eRho0B = ptr(F.eRho0B); v.eT0B = ptr(F.eT0B); v.eBetaB = ptr(F.eBetaB);
    v.uKind = ptr(F.uKind); v.pKind = ptr(F.pKind); v.uVal = ptr(F.uVal); v.uVf = ptr(F.uVf);
    v.uAssign = ptr(F.uAssign); v.uFixes = ptr(F.uFixes); v.pVal = ptr(F.pVal); v.pGrad = ptr(F.pGrad);
    v.p0Val = ptr(F.p0Val);
    v.gU = ptr(F.gU); v.gP = ptr(F.gP); v.gRho = ptr(F.gRho);
    v.src3 = ptr(F.src3); v.bD3 = ptr(F.bD3); v.bS3 = ptr(F.bS3); v.bMin = ptr(F.bMin);
    v.rAU = ptr(F.rAU); v.HbyA = ptr(F.HbyA); v.HbyAB = ptr(F.HbyAB);
    v.rhorAUf = ptr(F.rhorAUf); v.rhorAUfB = ptr(F.rhorAUfB); v.phiHbyA = ptr(F.phiHbyA);
    v.phiHbyAB = ptr(F.phiHbyAB); v.phig = ptr(F.phig); v.phigB = ptr(F.phigB); v.pCorr = ptr(F.pCorr);
    v.T = ptr(F.E->T); v.TbB = ptr(F.E->TbB); v.TA = ptr(F.E->TA); v.TB = ptr(F.E->TB);
    v.gx = F.g[0]; v.gy = F.g[1]; v.gz = F.g[2]; v.ghRef = F.ghRef;
    return v;
}

// cell-face entry q of a flow cell: 0 region face (owner side), 1 region face (neighbour side), 2 boundary slot
HD inline int ent(const FW& v, int q, int& j, int& other)
{
    const MeshView& g = v.g;
    const int e = g.cfEnt[q];
    if (e >= 0)
    {
        if (e < g.nI)
        {
            if (e >= v.f0 && e < v.f0 + v.nf) { j = e - v.f0; other = g.nei[e] - v.c0; return 0; }
            j = v.ifSlot[e - v.nIrG];
            return 2;
        }
        j = e - g.nI - v.b0;
        return 2;
    }
    const int f = -e - 1;
    if (f >= v.f0 && f < v.f0 + v.nf) { j = f - v.f0; other = g.own[f] - v.c0; return 1; }
    j = v.ifSlot[f - v.nIrG];
    return 2;
}

HD inline double Tslot(const FW& v, long s)
{
    const int f = v.sFace[s];
    if (f >= v.g.nI) return v.TbB[f - v.g.nI];
    const int j = f - v.nIrG;
    return v.sSideA[s] ? v.TA[j] : v.TB[j];
}

HD inline double ghC(const FW& v, long i)
{
    const v3 C = ld3(v.g.C, v.c0 + i);
    return v.gx*C.x + v.gy*C.y + v.gz*C.z - v.ghRef;
}

HD inline double ghF(const FW& v, const v3& Cf) { return v.gx*Cf.x + v.gy*Cf.y + v.gz*Cf.z - v.ghRef; }

// U_b = a + b U_P per component on slot s
HD inline void uCoeffs(const FW& v, long s, const v3& UP, double a[3], double b[3])
{
    switch (v.uKind[s])
    {
        case U_FIXED:
            for (int k = 0; k < 3; ++k) { a[k] = v.uVal[3*s + k]; b[k] = 0.0; }
            return;
        case U_INLETOUTLET:
        {
            const double f = v.uVf[s];
            for (int k = 0; k < 3; ++k) { a[k] = f*v.uVal[3*s + k]; b[k] = 1.0 - f; }
            return;
        }
        case U_PIOV:
            if (v.uVf[s] > 0.5)
            {
                // directionMixed, valueFraction I - nn, refValue 0: U_b = n (n.U_P)
                const v3 n = ld3(v.sSf, s)*(1.0/v.sMagSf[s]);
                const double un = dot(n, UP);
                const double nn[3] = {n.x, n.y, n.z};
                const double up[3] = {UP.x, UP.y, UP.z};
                for (int k = 0; k < 3; ++k) { b[k] = nn[k]*nn[k]; a[k] = nn[k]*(un - nn[k]*up[k]); }
                return;
            }
            for (int k = 0; k < 3; ++k) { a[k] = 0.0; b[k] = 1.0; }
            return;
        case U_SLIP:
        {
            const v3 n = ld3(v.sSf, s)*(1.0/v.sMagSf[s]);
            const double un = dot(n, UP);
            const double nn[3] = {n.x, n.y, n.z};
            const double up[3] = {UP.x, UP.y, UP.z};
            for (int k = 0; k < 3; ++k) { b[k] = 1.0 - nn[k]*nn[k]; a[k] = -nn[k]*(un - nn[k]*up[k]); }
            return;
        }
        default:
            for (int k = 0; k < 3; ++k) { a[k] = 0.0; b[k] = 1.0; }
    }
}

HD inline bool isPFixed(int pk)
{
    return pk == P_FIXED || pk == P_PRGH_P || pk == P_PRGH_TOTALP || pk == P_TOTALP;
}

HD inline double pSlot(const FW& v, long s, double pP)
{
    const int pk = v.pKind[s];
    if (isPFixed(pk)) return v.pVal[s];
    if (pk == P_FLUX) return pP + v.pGrad[s]/v.sDc[s];
    return pP;
}

// boundary snGrad of p_rgh as the patch evaluates it
HD inline double pSnGradSlot(const FW& v, long s, double pP)
{
    const int pk = v.pKind[s];
    if (isPFixed(pk)) return v.sDc[s]*(v.pVal[s] - pP);
    if (pk == P_FLUX) return v.pGrad[s];
    return 0.0;
}

// scalar slot coefficients: Y_b = a + bb Y_P
HD inline void ySlotCoeffs(int kind, double val, double grad, double vf, double dc, double phiB, double& a, double& bb)
{
    switch (kind)
    {
        case B_FIXED: a = val; bb = 0.0; return;
        case B_FIXEDGRAD: a = grad/dc; bb = 1.0; return;
        case B_MIXED: a = vf*val + (1.0 - vf)*grad/dc; bb = 1.0 - vf; return;
        case B_INLETOUTLET:
        {
            const double f = phiB < 0.0 ? 1.0 : 0.0;
            a = f*val;
            bb = 1.0 - f;
            return;
        }
        default: a = 0.0; bb = 1.0; return;
    }
}

// k / epsilon slot coefficients: phi_b = a + bb phi_P
HD inline void tSlotCoeffs(const FW& v, long s, int kind, double val, double kb_s, double& a, double& bb)
{
    switch (kind)
    {
        case B_FIXED: a = val; bb = 0.0; return;
        case B_INLETOUTLET:
        {
            const double f = v.phiB[s] < 0.0 ? 1.0 : 0.0;
            a = f*val;
            bb = 1.0 - f;
            return;
        }
        case B_TURB_INTENSITY:
        {
            const double f = v.phiB[s] < 0.0 ? 1.0 : 0.0;
            const v3 ub = ld3(v.Ub, s);
            a = f*1.5*val*val*dot(ub, ub);
            bb = 1.0 - f;
            return;
        }
        case B_TURB_MIXINGLEN:
        {
            const double f = v.phiB[s] < 0.0 ? 1.0 : 0.0;
            const double kp = dmax(kb_s, 0.0);
            a = f*val*kp*sqrt(kp);
            bb = 1.0 - f;
            return;
        }
        default: a = 0.0; bb = 1.0; return;
    }
}

HD inline double calcYPlusLam(double kappa, double E)
{
    double yp = 11.0;
    for (int i = 0; i < 10; ++i) yp = log(dmax(E*yp, 1.0))/kappa;
    return yp;
}

HD inline double calcYPlusTherm(double P, double Prat, double kappa, double E)
{
    double ypt = 11.0;
    for (int i = 0; i < 10; ++i)
    {
        const double f = ypt - (log(E*ypt)/kappa + P)/Prat;
        const double df = 1.0 - 1.0/(ypt*kappa*Prat);
        const double nw = ypt - f/df;
        if (nw < 1e-300) return 0.0;
        if (fabs(nw - ypt) < 0.01) return nw;
        ypt = nw;
    }
    return ypt;
}

// OpenFOAM limitedLinear k (NVDTVD::r) limiter on region internal face j
HD inline double limLinearFace(const FW& v, long j, const double* phi, const double* gphi, double k)
{
    const long f = v.f0 + j;
    const int o = v.g.own[f] - v.c0, n = v.g.nei[f] - v.c0;
    const v3 d = ld3(v.g.C, v.c0 + n) - ld3(v.g.C, v.c0 + o);
    const double F = v.phi[j];
    const double gradcf = F > 0.0 ? dot(d, ld3(gphi, o)) : dot(d, ld3(gphi, n));
    const double dphi = phi[n] - phi[o];
    double r;
    if (fabs(gradcf) >= 1000.0*fabs(dphi))
        r = 2000.0*(gradcf >= 0.0 ? 1.0 : -1.0)*(dphi >= 0.0 ? 1.0 : -1.0) - 1.0;
    else
        r = 2.0*gradcf/dphi - 1.0;
    return dmax(dmin(2.0/dmax(k, SMALL)*r, 1.0), 0.0);
}

// OpenFOAM multiComponentMixture operator+= rules (constTransport<hConstThermo<Boussinesq<specie>>>)
HD inline void mixProps(int nSp, const double* spD, double Tref0, const double* Yarr, long idx, long stride,
                        double& Cp, double& e, double& kappa, double& mu, double& r0, double& T0, double& beta)
{
    double tot = 0.0;
    for (int m = 0; m < nSp; ++m) tot += Yarr[m*stride + idx];
    const bool ok = fabs(tot) > SMALL;
    double cp = 0, hs = 0, mu_ = 0, pr = 0, r0_ = 0, t0_ = 0, bt_ = 0;
    for (int m = 0; m < nSp; ++m)
    {
        const double w = ok ? Yarr[m*stride + idx]/tot : (m == 0 ? 1.0 : 0.0);
        const double* d = spD + 7*m;
        cp += w*d[0];
        hs += w*d[1];
        mu_ += w*d[2];
        pr += w*d[3];
        r0_ += w*d[4];
        t0_ += w*d[5];
        bt_ += w*d[6];
    }
    Cp = cp;
    e = hs - cp*Tref0;
    mu = mu_;
    kappa = cp*mu_/pr;
    r0 = r0_;
    T0 = t0_;
    beta = bt_;
}

// least-squares gradient of a cell field phi[i*st + off] with slot values phib[s*bst + boff]; optional cellLimited
// (minmod) coefficient k; result into out[i*ost + ooff + d*dst]
void lsGradient(const FW& v, const double* phi, int st, int off, const double* phib, int bst, int boff, double k,
                double* out, int ost, int ooff, int dst)
{
    ProfScope prof("lsGradient");
    for_n(v.nc, [=] HD (long i) {
        const long c = v.c0 + i;
        const double hc = phi[i*st + off];
        v3 gr = V3(0, 0, 0);
        double mx = hc, mn = hc;
        for (int q = v.g.cfPtr[c]; q < v.g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            double hv;
            if (t == 0) { hv = phi[long(o)*st + off]; gr += ld3(v.g.lsP, v.f0 + j)*(hv - hc); }
            else if (t == 1) { hv = phi[long(o)*st + off]; gr -= ld3(v.g.lsN, v.f0 + j)*(hc - hv); }
            else { hv = phib[long(j)*bst + boff]; gr += ld3(v.sLs, j)*(hv - hc); }
            mx = dmax(mx, hv);
            mn = dmin(mn, hv);
        }
        if (k > 0)
        {
            double maxD = mx - hc, minD = mn - hc;
            if (k < 1)
            {
                const double ext = (1.0/k - 1.0)*(maxD - minD);
                maxD += ext;
                minD -= ext;
            }
            double lim = 1.0;
            const v3 Cc = ld3(v.g.C, c);
            for (int q = v.g.cfPtr[c]; q < v.g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                const v3 Cf = t == 2 ? ld3(v.sCf, j) : ld3(v.g.Cf, v.f0 + j);
                const double ex = dot(Cf - Cc, gr);
                double r = 1.0;
                if (ex > SMALL) r = maxD/ex;
                else if (ex < -SMALL) r = minD/ex;
                lim = dmin(lim, r);
            }
            gr = gr*lim;
        }
        out[i*ost + ooff] = gr.x;
        out[i*ost + ooff + dst] = gr.y;
        out[i*ost + ooff + 2*dst] = gr.z;
    });
}

// lsGradient of the three components of a vector field (U[3 i + k], slot values Ub[3 s + k]) in one pass over the
// cell faces, each component with its own cellLimited limiter (as three lsGradient calls, same arithmetic):
// out[9 i + k + 3 d] = d U_k / d x_d
void lsGradient3(const FW& v, const double* phi, const double* phib, double k, double* out)
{
    ProfScope prof("lsGradient");
    for_n(v.nc, [=] HD (long i) {
        const long c = v.c0 + i;
        const double hc[3] = {phi[3*i], phi[3*i + 1], phi[3*i + 2]};
        v3 gr[3] = {V3(0, 0, 0), V3(0, 0, 0), V3(0, 0, 0)};
        double mx[3] = {hc[0], hc[1], hc[2]}, mn[3] = {hc[0], hc[1], hc[2]};
        for (int q = v.g.cfPtr[c]; q < v.g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            const double* hv = t == 2 ? phib + 3L*j : phi + 3L*o;
            const v3 ls = t == 0 ? ld3(v.g.lsP, v.f0 + j) : (t == 1 ? ld3(v.g.lsN, v.f0 + j) : ld3(v.sLs, j));
            for (int m = 0; m < 3; ++m)
            {
                if (t == 1) gr[m] -= ls*(hc[m] - hv[m]);
                else gr[m] += ls*(hv[m] - hc[m]);
                mx[m] = dmax(mx[m], hv[m]);
                mn[m] = dmin(mn[m], hv[m]);
            }
        }
        if (k > 0)
        {
            double maxD[3], minD[3], lim[3] = {1.0, 1.0, 1.0};
            for (int m = 0; m < 3; ++m)
            {
                maxD[m] = mx[m] - hc[m];
                minD[m] = mn[m] - hc[m];
                if (k < 1)
                {
                    const double ext = (1.0/k - 1.0)*(maxD[m] - minD[m]);
                    maxD[m] += ext;
                    minD[m] -= ext;
                }
            }
            const v3 Cc = ld3(v.g.C, c);
            for (int q = v.g.cfPtr[c]; q < v.g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                const v3 dC = (t == 2 ? ld3(v.sCf, j) : ld3(v.g.Cf, v.f0 + j)) - Cc;
                for (int m = 0; m < 3; ++m)
                {
                    const double ex = dot(dC, gr[m]);
                    double r = 1.0;
                    if (ex > SMALL) r = maxD[m]/ex;
                    else if (ex < -SMALL) r = minD[m]/ex;
                    lim[m] = dmin(lim[m], r);
                }
            }
            for (int m = 0; m < 3; ++m) gr[m] = gr[m]*lim[m];
        }
        for (int m = 0; m < 3; ++m)
        {
            out[9*i + m] = gr[m].x;
            out[9*i + m + 3] = gr[m].y;
            out[9*i + m + 6] = gr[m].z;
        }
    });
}

// limited-corrected snGrad of a scalar on region face j: dc (phi_n - phi_o) + limiter * corr
HD inline double snGradFace(const FW& v, long j, int o, int n, const double* phi, const double* gphi, double psi)
{
    const long f = v.f0 + j;
    const double dc = v.g.nonOrthDC[f];
    double sn = dc*(phi[n] - phi[o]);
    if (psi > 0)
    {
        const double w = v.g.w[f];
        const v3 gf = w*ld3(gphi, o) + (1 - w)*ld3(gphi, n);
        double c = dot(ld3(v.g.corrVec, f), gf);
        if (psi < 1) c *= dmin(psi*fabs(sn)/((1 - psi)*fabs(c) + SMALL), 1.0);
        sn += c;
    }
    return sn;
}

HD inline double muEffC(const FW& v, long i) { return v.mu[i] + v.rho[i]*v.nut[i]; }

// explicit UEqn face fluxes (owner +X, neighbour -X) of region face j for the three components:
// limited non-orthogonal correction of the laplacian, deferred linearUpwind correction, dev2 term
struct UFace
{
    double gmag, F, X[3];
};

HD inline UFace uFace(const FW& v, long j, int o, int n, double psiU, int conv)
{
    const MeshView& g = v.g;
    const long f = v.f0 + j;
    const double w = g.w[f];
    const double dc = g.nonOrthDC[f];
    const double mo = muEffC(v, o), mn = muEffC(v, n);
    UFace r;
    r.gmag = (w*mo + (1 - w)*mn)*g.magSf[f]*dc;
    r.F = v.phi[j];
    const double* Go = v.gU + 9*long(o);
    const double* Gn = v.gU + 9*long(n);
    // non-orthogonal correction: corrVec . interpolate(grad U), one limiter from the vector magnitudes
    double c[3] = {0, 0, 0};
    if (psiU > 0)
    {
        const v3 cv = ld3(g.corrVec, f);
        for (int k = 0; k < 3; ++k)
        {
            const v3 gf = V3(w*Go[k] + (1 - w)*Gn[k], w*Go[3 + k] + (1 - w)*Gn[3 + k], w*Go[6 + k] + (1 - w)*Gn[6 + k]);
            c[k] = dot(cv, gf);
        }
        if (psiU < 1)
        {
            double sn2 = 0, c2 = 0;
            for (int k = 0; k < 3; ++k)
            {
                const double sn = dc*(v.U[3*long(n) + k] - v.U[3*long(o) + k]);
                sn2 += sn*sn;
                c2 += c[k]*c[k];
            }
            const double lim = dmin(psiU*sqrt(sn2)/((1 - psiU)*sqrt(c2) + SMALL), 1.0);
            for (int k = 0; k < 3; ++k) c[k] *= lim;
        }
    }
    const double gs = r.gmag/dc;
    const v3 S = ld3(g.Sf, f);
    const double s[3] = {S.x, S.y, S.z};
    const double trO = Go[0] + Go[4] + Go[8], trN = Gn[0] + Gn[4] + Gn[8];
    const v3 Cf = ld3(g.Cf, f);
    const v3 dO = Cf - ld3(g.C, v.c0 + o), dN = Cf - ld3(g.C, v.c0 + n);
    for (int k = 0; k < 3; ++k)
    {
        double X = gs*c[k];
        if (conv == 1)                       // linearUpwind grad(U): upwind value + (Cf - C_up) . grad(U_k)
        {
            const double* Gu = r.F >= 0 ? Go : Gn;
            const v3 d = r.F >= 0 ? dO : dN;
            X -= r.F*(d.x*Gu[k] + d.y*Gu[3 + k] + d.z*Gu[6 + k]);
        }
        // div(muEff dev2(T(grad U))): (S . T(G))_k = sum_i S_i G[k][i]; Gauss linear of the cell products
        const double tO = mo*(s[0]*Go[3*k] + s[1]*Go[3*k + 1] + s[2]*Go[3*k + 2] - (2.0/3.0)*trO*s[k]);
        const double tN = mn*(s[0]*Gn[3*k] + s[1]*Gn[3*k + 1] + s[2]*Gn[3*k + 2] - (2.0/3.0)*trN*s[k]);
        X += w*tO + (1 - w)*tN;
        r.X[k] = X;
    }
    return r;
}

} // namespace

// ============================================================================================ setup
void Flow::init(const DMesh& mesh, Energy& energy, int regionIndex, const std::vector<int>& ifFaces,
                const std::vector<char>& ifSideA)
{
    m = &mesh;
    E = &energy;
    region = regionIndex;
    const MeshIn::Region& R = mesh.regions[regionIndex];
    c0 = R.c0; nc = R.nC; f0 = R.f0; nf = R.nI; b0 = R.b0; nBr = R.nB;
    const int nIf = int(ifFaces.size());
    nS = nBr + nIf;
    // slots: region boundary faces, then interface faces
    std::vector<int> face(nS);
    std::vector<char> sideA(nS, 1);
    for (int s = 0; s < nBr; ++s) face[s] = mesh.nI + b0 + s;
    for (int k = 0; k < nIf; ++k) { face[nBr + k] = ifFaces[k]; sideA[nBr + k] = ifSideA[k]; }
    sFace = to_dev(face);
    sSideA = to_dev(sideA);
    std::vector<int> isl(mesh.nIface, -1);
    for (int k = 0; k < nIf; ++k) isl[ifFaces[k] - mesh.nIr] = nBr + k;
    ifSlot = to_dev(isl);
    sCell.resize(nS); sSf.resize(3*nS); sCf.resize(3*nS); sMagSf.resize(nS); sDc.resize(nS); sLs.resize(3*nS);
    {
        const MeshView g = view(mesh);
        const int* fc = ptr(sFace);
        const char* sa = ptr(sSideA);
        int* cell = ptr(sCell);
        double *Sf = ptr(sSf), *Cf = ptr(sCf), *ms = ptr(sMagSf), *dc = ptr(sDc), *ls = ptr(sLs);
        const int cc0 = c0, nIr = mesh.nIr;
        for_n(nS, [=] HD (long s) {
            const int f = fc[s];
            st3(Cf, s, ld3(g.Cf, f));
            ms[s] = g.magSf[f];
            if (f >= g.nI)
            {
                const int b = f - g.nI;
                cell[s] = g.own[f] - cc0;
                st3(Sf, s, ld3(g.Sf, f));
                dc[s] = g.dcB[b];
                st3(ls, s, ld3(g.lsB, b));
            }
            else if (sa[s])
            {
                cell[s] = g.own[f] - cc0;
                st3(Sf, s, ld3(g.Sf, f));
                dc[s] = g.dcIA[f - nIr];
                st3(ls, s, ld3(g.lsP, f));
            }
            else
            {
                cell[s] = g.nei[f] - cc0;
                st3(Sf, s, -ld3(g.Sf, f));
                dc[s] = g.dcIB[f - nIr];
                st3(ls, s, ld3(g.lsN, f));
            }
        });
    }
    auto z = [](dvec<double>& a, long n) { a.resize(n); fill(a, 0.0); };
    z(U, 3L*nc); z(U0, 3L*nc); z(Ub, 3L*nS); z(Ub0, 3L*nS);
    z(p, nc); z(pPrev, nc); z(pAbs, nc); z(pAbsB, nS);
    z(rho, nc); z(rho0f, nc); z(rhoB, nS); z(rhoB0, nS);
    z(phi, nf); z(phiB, nS); z(phi0, nf); z(phiB0, nS);
    z(K0, nc); z(pOld, nc); z(dpdt, nc);
    z(mu, nc); z(muB, nS); z(nut, nc); z(nutB, nS);
    z(eRho0, nc); z(eT0, nc); z(eBeta, nc); z(eRho0B, nS); z(eT0B, nS); z(eBetaB, nS);
    z(cpB, nS); z(eB, nS); z(kappaB, nS);
    uKind.resize(nS); fill(uKind, int(U_FIXED));
    pKind.resize(nS); fill(pKind, int(P_ZEROGRAD));
    z(uVal, 3L*nS); z(uVf, nS); z(pVal, nS); z(pGrad, nS); z(p0Val, nS);
    uAssign.resize(nS); fill(uAssign, char(0));
    uFixes.resize(nS); fill(uFixes, char(1));
    z(gU, 9L*nc); z(gP, 3L*nc); z(gRho, 3L*nc);
    z(src3, 3L*nc); z(bD3, 3L*nc); z(bS3, 3L*nc); z(bMin, nc);
    z(rAU, nc); z(HbyA, 3L*nc); z(HbyAB, 3L*nS);
    z(rhorAUf, nf); z(rhorAUfB, nS); z(phiHbyA, nf); z(phiHbyAB, nS); z(phig, nf); z(phigB, nS); z(pCorr, nf);
    z(wS, nS);
    MU.init(c0, nc, f0, nf);
    Mp.init(c0, nc, f0, nf);
    // reconstruct tensors: safeInv(sum over faces SfHat Sf)
    Rinv.resize(6L*nc);
    {
        FW v = fw(*this);
        double* Ri = ptr(Rinv);
        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double t[6] = {0, 0, 0, 0, 0, 0};
            for (int q = v.g.cfPtr[c]; q < v.g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int k = ent(v, q, j, o);
                const v3 S = k == 2 ? ld3(v.sSf, j) : ld3(v.g.Sf, v.f0 + j);
                const double r = 1.0/mag(S);
                t[0] += S.x*S.x*r; t[1] += S.x*S.y*r; t[2] += S.x*S.z*r;
                t[3] += S.y*S.y*r; t[4] += S.y*S.z*r; t[5] += S.z*S.z*r;
            }
            symmSafeInv(t);
            for (int k = 0; k < 6; ++k) Ri[6*i + k] = t[k];
        });
    }
}

void Flow::initSpecies(const FluidIn& fi)
{
    nSp = int(fi.species.size());
    if (nSp <= 0) return;
    inertIndex = fi.inertIndex;
    convY = fi.convY;
    convYK = fi.convYK;
    boundedY = fi.boundedY;
    limitKH = fi.limitKH;
    Tref0 = fi.species[0].Tref;
    sp = fi.species;
    std::vector<double> sd(7*nSp);
    for (int m = 0; m < nSp; ++m)
    {
        const SpecieIn& s = sp[m];
        sd[7*m + 0] = s.Cp;
        sd[7*m + 1] = s.Hsref;
        sd[7*m + 2] = s.mu;
        sd[7*m + 3] = s.Pr;
        sd[7*m + 4] = s.rho0;
        sd[7*m + 5] = s.T0;
        sd[7*m + 6] = s.beta;
    }
    spData = to_dev(sd);
    auto z = [](dvec<double>& a, long n) { a.resize(n); fill(a, 0.0); };
    z(Y, long(nSp)*nc);
    z(Y0, long(nSp)*nc);
    z(SuY, long(nSp)*nc);
    z(SpY, long(nSp)*nc);
    z(Yb, long(nSp)*nS);
    z(yVal, long(nSp)*nS);
    z(yGrad, long(nSp)*nS);
    z(yVf, long(nSp)*nS);
    yKind.resize(long(nSp)*nS);
    fill(yKind, int(B_ZEROGRAD));
}

void Flow::updateMixture(bool evalYb)
{
    if (nSp <= 0) return;
    FW v = fw(*this);
    const int ns = nSp;
    const double tr0 = Tref0;
    const double* sd = ptr(spData);
    const double* yp = ptr(Y);
    double* yb = ptr(Yb);
    if (evalYb)
    {
        const int* yk = ptr(yKind);
        const double *yv = ptr(yVal), *yg = ptr(yGrad), *yf = ptr(yVf);
        for_n(long(ns)*nS, [=] HD (long idx) {
            const int m = int(idx / v.nS);
            const long s = idx - long(m)*v.nS;
            double a, bb;
            ySlotCoeffs(yk[idx], yv[idx], yg[idx], yf[idx], v.sDc[s], v.phiB[s], a, bb);
            yb[idx] = a + bb*yp[long(m)*v.nc + v.sCell[s]];
        });
    }
    double *cp = ptr(E->cp), *ee = ptr(E->e), *kap = ptr(E->kappa);
    double *muC = ptr(mu), *r0C = ptr(eRho0), *t0C = ptr(eT0), *btC = ptr(eBeta);
    for_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        mixProps(ns, sd, tr0, yp, i, v.nc, cp[c], ee[c], kap[c], muC[i], r0C[i], t0C[i], btC[i]);
    });
    double *cpb = ptr(cpB), *eb = ptr(eB), *kapb = ptr(kappaB);
    double *ecpb = ptr(E->cpB), *eeb = ptr(E->eB), *ekapb = ptr(E->kappaB);
    double *muS = ptr(muB), *r0S = ptr(eRho0B), *t0S = ptr(eT0B), *btS = ptr(eBetaB);
    for_n(nS, [=] HD (long s) {
        mixProps(ns, sd, tr0, yb, s, v.nS, cpb[s], eb[s], kapb[s], muS[s], r0S[s], t0S[s], btS[s]);
        const int f = v.sFace[s];
        if (f >= v.g.nI)
        {
            const int b = f - v.g.nI;
            ecpb[b] = cpb[s];
            eeb[b] = eb[s];
            ekapb[b] = kapb[s];
        }
    });
}

void Flow::updateRho()
{
    FW v = fw(*this);
    for_n(nc, [=] HD (long i) {
        v.rho[i] = v.eRho0[i]*(1.0 - v.eBeta[i]*(v.T[v.c0 + i] - v.eT0[i]));
    });
    for_n(nS, [=] HD (long s) {
        const double rb = v.eRho0B[s]*(1.0 - v.eBetaB[s]*(Tslot(v, s) - v.eT0B[s]));
        v.rhoB[s] = rb;
        const int pk = v.pKind[s];
        if (pk == P_PRGH_P)
            v.pVal[s] = v.p0Val[s] - rb*ghF(v, ld3(v.sCf, s));
        else if (pk == P_PRGH_TOTALP)
        {
            const v3 ub = ld3(v.Ub, s);
            v.pVal[s] = v.p0Val[s] - (v.phiB[s] < 0.0 ? 0.5*rb*dot(ub, ub) : 0.0) - rb*ghF(v, ld3(v.sCf, s));
        }
        else if (pk == P_TOTALP)
        {
            const v3 ub = ld3(v.Ub, s);
            v.pVal[s] = v.p0Val[s] - (v.phiB[s] < 0.0 ? 0.5*rb*dot(ub, ub) : 0.0);
        }
    });
}

void Flow::updateBoundary(bool evalFlowPatches)
{
    FW v = fw(*this);
    if (evalFlowPatches && nFP > 0)
    {
        const int *s0p = ptr(fpS0), *np = ptr(fpN), *kp = ptr(fpKind);
        const double* qp = ptr(fpQ);
        for_n(nFP, [=] HD (long p) {
            const int s0 = s0p[p], n = np[p], kd = kp[p];
            const bool mass = (kd & 1) != 0, outlet = (kd & 2) != 0;
            const double q = qp[p];
            double rA = 0.0, est = 0.0;
            for (int i = 0; i < n; ++i)
            {
                const int s = s0 + i;
                const double ms = v.sMagSf[s];
                const double r = mass ? v.rhoB[s] : 1.0;
                rA += r*ms;
                if (outlet)
                {
                    const v3 nf = ld3(v.sSf, s)*(1.0/ms);
                    const double nUp = dmax(dot(nf, ld3(v.U, v.sCell[s])), 0.0);
                    est += r*ms*nUp;
                }
            }
            for (int i = 0; i < n; ++i)
            {
                const int s = s0 + i;
                const double ms = v.sMagSf[s];
                const v3 nf = ld3(v.sSf, s)*(1.0/ms);
                v3 ub;
                if (!outlet)
                {
                    ub = -nf*(q/dmax(rA, SMALL));
                }
                else
                {
                    const v3 Up = ld3(v.U, v.sCell[s]);
                    const double nUp0 = dot(nf, Up);
                    const v3 Ut = Up - nf*nUp0;
                    double nUp = dmax(nUp0, 0.0);
                    if (est > 0.5*q && fabs(est) > SMALL)
                        nUp *= fabs(q)/fabs(est);
                    else
                        nUp += (q - est)/dmax(rA, SMALL);
                    ub = Ut + nf*nUp;
                }
                v.uVal[3*s] = ub.x;
                v.uVal[3*s + 1] = ub.y;
                v.uVal[3*s + 2] = ub.z;
            }
        });
    }
    for_n(nS, [=] HD (long s) {
        const int k = v.uKind[s];
        if (k == U_INLETOUTLET || k == U_PIOV) v.uVf[s] = v.phiB[s] < 0 ? 1.0 : 0.0;
        double a[3], b[3];
        const v3 UP = ld3(v.U, v.sCell[s]);
        uCoeffs(v, s, UP, a, b);
        const v3 ub = V3(a[0] + b[0]*UP.x, a[1] + b[1]*UP.y, a[2] + b[2]*UP.z);
        v.Ub[3*s] = ub.x;
        v.Ub[3*s + 1] = ub.y;
        v.Ub[3*s + 2] = ub.z;
        const int pk = v.pKind[s];
        if (pk == P_PRGH_P)
            v.pVal[s] = v.p0Val[s] - v.rhoB[s]*ghF(v, ld3(v.sCf, s));
        else if (pk == P_PRGH_TOTALP)
            v.pVal[s] = v.p0Val[s] - (v.phiB[s] < 0.0 ? 0.5*v.rhoB[s]*dot(ub, ub) : 0.0)
                        - v.rhoB[s]*ghF(v, ld3(v.sCf, s));
        else if (pk == P_TOTALP)
            v.pVal[s] = v.p0Val[s] - (v.phiB[s] < 0.0 ? 0.5*v.rhoB[s]*dot(ub, ub) : 0.0);
    });
}

void Flow::initPhi()
{
    FW v = fw(*this);
    for_n(nf, [=] HD (long j) {
        const long f = v.f0 + j;
        const int o = v.g.own[f] - v.c0, n = v.g.nei[f] - v.c0;
        const double w = v.g.w[f];
        const v3 rU = w*v.rho[o]*ld3(v.U, o) + (1 - w)*v.rho[n]*ld3(v.U, n);
        v.phi[j] = dot(rU, ld3(v.g.Sf, f));
    });
    for_n(nS, [=] HD (long s) { v.phiB[s] = v.rhoB[s]*dot(ld3(v.Ub, s), ld3(v.sSf, s)); });
}

void Flow::initTurbulence(const FluidIn& fi)
{
    if (nSp <= 0)
    {
        fill(cpB, fi.Cp);
        fill(eB, fi.Hsref - fi.Cp*fi.Tref);
        fill(kappaB, fi.kappa);
    }
    turb = fi.turb;
    if (!turb.enabled) return;
    auto z = [](dvec<double>& a, long n) { a.resize(n); fill(a, 0.0); };
    z(k, nc); z(k0, nc); z(eps, nc); z(eps0, nc); z(alphat, nc);
    z(kb, nS); z(epsb, nS); z(alphatB, nS);
    kKind.resize(nS); fill(kKind, int(B_ZEROGRAD));
    epsKind.resize(nS); fill(epsKind, int(B_ZEROGRAD));
    nutKind.resize(nS); fill(nutKind, int(NUT_CALC));
    atKind.resize(nS); fill(atKind, int(AT_CALC));
    z(kVal, nS); z(epsVal, nS);
    z(eCmu, nS); z(eKappa, nS); z(nCmu, nS); z(nKappa, nS); z(nE, nS); z(yPlusLam, nS);
    z(atPrt, nS); z(atKappa, nS); z(atE, nS); z(sY, nS); z(sCw, nS);
    isWall.resize(nS); fill(isWall, char(0));
    z(Gturb, nc); z(GbyNu, nc); z(divU, nc); z(De, nc); z(DeB, nS);
}

void Flow::setupTurbulence(const std::vector<double>& kIn, const std::vector<double>& epsIn,
                           const std::vector<double>& nutIn, const std::vector<double>& alphatIn,
                           const std::vector<int>& kK, const std::vector<double>& kV,
                           const std::vector<int>& eK, const std::vector<double>& eV,
                           const std::vector<int>& nK, const std::vector<int>& aK,
                           const std::vector<double>& eCmuIn, const std::vector<double>& eKappaIn,
                           const std::vector<double>& nCmuIn, const std::vector<double>& nKappaIn,
                           const std::vector<double>& nEIn, const std::vector<double>& atPrtIn,
                           const std::vector<double>& atKappaIn, const std::vector<double>& atEIn,
                           const std::vector<double>& sYIn, const std::vector<double>& alphatB0)
{
    if (!turb.enabled) return;
    k = to_dev(kIn); k0 = to_dev(kIn);
    eps = to_dev(epsIn); eps0 = to_dev(epsIn);
    nut = to_dev(nutIn);
    alphat = to_dev(alphatIn);
    kKind = to_dev(kK); kVal = to_dev(kV);
    epsKind = to_dev(eK); epsVal = to_dev(eV);
    nutKind = to_dev(nK); atKind = to_dev(aK);
    eCmu = to_dev(eCmuIn); eKappa = to_dev(eKappaIn);
    nCmu = to_dev(nCmuIn); nKappa = to_dev(nKappaIn); nE = to_dev(nEIn);
    atPrt = to_dev(atPrtIn); atKappa = to_dev(atKappaIn); atE = to_dev(atEIn);
    sY = to_dev(sYIn);
    alphatB = to_dev(alphatB0);
    std::vector<double> ypl(nS);
    for (int s = 0; s < nS; ++s) ypl[s] = calcYPlusLam(nKappaIn[s], nEIn[s]);
    yPlusLam = to_dev(ypl);
    const std::vector<int> sCellH = to_host(sCell);
    std::vector<char> wallH(nS, 0);
    std::vector<int> nwCell(nc, 0);
    for (int s = 0; s < nS; ++s)
    {
        if (nK[s] == NUT_WALL)
        {
            wallH[s] = 1;
            nwCell[sCellH[s]]++;
        }
    }
    std::vector<double> cwH(nS, 0.0);
    for (int s = 0; s < nS; ++s)
        if (wallH[s]) cwH[s] = 1.0/double(nwCell[sCellH[s]]);
    std::vector<int> wcL, wcG;
    for (int i = 0; i < nc; ++i)
    {
        if (nwCell[i] > 0)
        {
            wcL.push_back(i);
            wcG.push_back(c0 + i);
        }
    }
    isWall = to_dev(wallH);
    sCw = to_dev(cwH);
    wallCells = to_dev(wcL);
    wallCellsGlobal = to_dev(wcG);
    epsWall.resize(wcL.size());
    fill(epsWall, 0.0);
}

void Flow::correctNut()
{
    if (!turb.enabled) return;
    FW v = fw(*this);
    const double *kp = ptr(k), *ep = ptr(eps);
    const double *kbp = ptr(kb), *ebp = ptr(epsb);
    const int* nK = ptr(nutKind);
    const double *ncmu = ptr(nCmu), *nkap = ptr(nKappa), *ne = ptr(nE), *ypl = ptr(yPlusLam), *sy = ptr(sY);
    const double Cmu = turb.Cmu, epsMin = turb.epsMin;
    for_n(nc, [=] HD (long i) {
        v.nut[i] = Cmu*kp[i]*kp[i]/ep[i];
    });
    for_n(nS, [=] HD (long s) {
        const int nk = nK[s];
        const int P = v.sCell[s];
        if (nk == NUT_WALL)
        {
            const double nuw = v.muB[s]/v.rhoB[s];
            const double kc = dmax(kp[P], 0.0);
            const double yp = pow(ncmu[s], 0.25)*sy[s]*sqrt(kc)/nuw;
            v.nutB[s] = yp > ypl[s] ? nuw*(yp*nkap[s]/log(dmax(ne[s]*yp, 1e-30)) - 1.0) : 0.0;
        }
        else if (nk == NUT_ZEROGRAD)
        {
            v.nutB[s] = v.nut[P];
        }
        else
        {
            v.nutB[s] = Cmu*kbp[s]*kbp[s]/dmax(ebp[s], epsMin);
        }
    });
}

void Flow::updateAlphat()
{
    if (!turb.enabled) return;
    FW v = fw(*this);
    const double* kp = ptr(k);
    double *at = ptr(alphat), *atB = ptr(alphatB);
    const int* aK = ptr(atKind);
    const double *ncmu = ptr(nCmu), *sy = ptr(sY), *aprt = ptr(atPrt), *akap = ptr(atKappa), *ae = ptr(atE);
    const double *cpb = ptr(cpB), *kapb = ptr(kappaB);
    const double Prt = turb.Prt;
    for_n(nc, [=] HD (long i) {
        at[i] = v.rho[i]*v.nut[i]/Prt;
    });
    for_n(nS, [=] HD (long s) {
        const int ak = aK[s];
        const int P = v.sCell[s];
        if (ak == AT_WALL)
        {
            atB[s] = v.rhoB[s]*v.nutB[s]/aprt[s];
        }
        else if (ak == AT_JAYATILLEKE)
        {
            const double rhow = v.rhoB[s], muw = v.muB[s];
            const double cpw = cpb[s], kw = kapb[s];
            const double alphaw = kw/cpw;
            const double y = sy[s];
            const double nuw = muw/rhow;
            const double kc = dmax(kp[P], 0.0);
            const double yp = pow(ncmu[s], 0.25)*y*sqrt(kc)/nuw;
            const v3 Uw = ld3(v.Ub, s), Up = ld3(v.U, P);
            const double magUp = mag(Up - Uw), magUw = mag(Uw);
            const double dT_sn = (Tslot(v, s) - v.T[v.c0 + P])*v.sDc[s];
            const double qDot = (alphaw + atB[s])*cpw*dT_sn;
            const double uTau = yp/y*nuw;
            const double Pr = muw/alphaw;
            const double prt = aprt[s];
            const double Prat = Pr/prt;
            const double Pjay = 9.24*(pow(Prat, 0.75) - 1.0)*(1.0 + 0.28*exp(-0.007*Prat));
            const double kap = akap[s], E = ae[s];
            const double ypt = calcYPlusTherm(Pjay, Prat, kap, E);
            const double A = qDot*rhow*uTau*y;
            double aEff;
            if (yp < ypt)
            {
                const double B = qDot*Pr*yp;
                const double C = Pr*0.5*rhow*uTau*magUp*magUp;
                aEff = A/(B + C + 1e-300);
            }
            else
            {
                const double B = qDot*prt*(1.0/kap*log(dmax(E*yp, 1e-30)) + Pjay);
                const double magUc = uTau/kap*log(dmax(E*ypt, 1e-30)) - magUw;
                const double C = 0.5*rhow*uTau*(prt*magUp*magUp + (Pr - prt)*magUc*magUc);
                aEff = A/(B + C + 1e-300);
            }
            atB[s] = dmax(0.0, aEff - alphaw);
        }
        else if (ak == AT_ZEROGRAD)
        {
            atB[s] = at[P];
        }
        else
        {
            atB[s] = v.rhoB[s]*v.nutB[s]/Prt;
        }
    });
}

void Flow::validateTurbulence()
{
    if (!turb.enabled) return;
    updateBoundary(false);
    correctNut();
    updateAlphat();
    FW v = fw(*this);
    double* Eat = ptr(E->alphat);
    double* EatB = ptr(E->alphatB);
    double* Ekw = ptr(E->kappaWall);
    const double *at = ptr(alphat), *atB = ptr(alphatB), *cpb = ptr(cpB), *kapb = ptr(kappaB);
    for_n(nc, [=] HD (long i) { Eat[v.c0 + i] = at[i]; });
    for_n(nS, [=] HD (long s) {
        const int f = v.sFace[s];
        if (f >= v.g.nI) EatB[f - v.g.nI] = atB[s];
        else Ekw[f - v.nIrG] = kapb[s] + cpb[s]*atB[s];
    });
}

void Flow::storeOld(double dt, bool frozen, bool storePrevIter)
{
    FW v = fw(*this);
    if (kOldReady)
        for_n(nc, [=] HD (long i) { const v3 u = ld3(v.U, i); v.K0[i] = 0.5*dot(u, u); });
    if (pOldReady) copy(pAbs, pOld);
    copy(U, U0);
    copy(Ub, Ub0);
    copy(rho, rho0f);
    copy(rhoB, rhoB0);
    copy(phi, phi0);
    copy(phiB, phiB0);
    if (nSp > 0) copy(Y, Y0);
    if (turb.enabled) { copy(k, k0); copy(eps, eps0); }
    if (storePrevIter) copy(p, pPrev);
    if (!frozen)
    {
        // rhoEqn (first outer corrector): rho = rho.oldTime() - dt div(phi)/V
        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double d = 0;
            for (int q = v.g.cfPtr[c]; q < v.g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                d += t == 0 ? v.phi[j] : (t == 1 ? -v.phi[j] : v.phiB[j]);
            }
            v.rho[i] = v.rho0f[i] - dt*d/v.g.V[c];
        });
    }
}

// ============================================================================================ UEqn.H
void Flow::momentum(double dt, double alpha, const SolverCtl& ctl, bool predictor, SolveInfo info[3])
{
    ProfScope prof("momentum");
    FW v = fw(*this);
    const MeshView g = v.g;
    { ProfScope pb("updateBoundary"); updateBoundary(); }
    // grad(U) with the grad(U) scheme (used by linearUpwind, dev2 and the laplacian correction)
    lsGradient3(v, v.U, v.Ub, limitKU, v.gU);
    LduView L = view(MU);
    const double psi = psiU;
    const int cv = conv;
    const bool bnd = bounded;
    std::unique_ptr<ProfScope> pa(new ProfScope("assemble"));
    // face terms once per face (the cell rows gather them): laplacian coefficient and the explicit fluxes X, kept
    // in face arrays that pressure() sets before it uses them
    double *fg = v.pCorr, *fx0 = v.phig, *fx1 = v.phiHbyA, *fx2 = v.rhorAUf;
    for_n(nf, [=] HD (long j) {
        const long f = v.f0 + j;
        const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
        const UFace u = uFace(v, j, o, n, psi, cv);
        L.upper[j] = -u.gmag + (u.F < 0 ? u.F : 0.0);
        L.lower[j] = -u.gmag - (u.F > 0 ? u.F : 0.0);
        fg[j] = u.gmag;
        fx0[j] = u.X[0];
        fx1[j] = u.X[1];
        fx2[j] = u.X[2];
    });
    const double rdt = dt > 0 ? 1.0/dt : 0.0;
    for_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        double D = 0, divPhi = 0, bA = 0, bM = 0;
        double S[3] = {0, 0, 0}, bD[3] = {0, 0, 0}, bS[3] = {0, 0, 0};
        const v3 UP = ld3(v.U, i);
        const double* GP = v.gU + 9*i;
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            if (t == 0 || t == 1)
            {
                const double F = v.phi[j];
                const double X[3] = {fx0[j], fx1[j], fx2[j]};
                if (t == 0)
                {
                    D += fg[j] + (F > 0 ? F : 0.0);
                    divPhi += F;
                    for (int k = 0; k < 3; ++k) S[k] += X[k];
                }
                else
                {
                    D += fg[j] - (F < 0 ? F : 0.0);
                    divPhi -= F;
                    for (int k = 0; k < 3; ++k) S[k] -= X[k];
                }
            }
            else
            {
                const long s = j;
                const double muB = v.muB[s] + v.rhoB[s]*v.nutB[s];
                const double gb = muB*v.sMagSf[s]*v.sDc[s];
                const double Fb = v.phiB[s];
                divPhi += Fb;
                double a[3], b[3];
                uCoeffs(v, s, UP, a, b);
                double mx = -1e300, mn = 1e300;
                for (int k = 0; k < 3; ++k)
                {
                    const double iC = gb*(1 - b[k]) + Fb*b[k];
                    bD[k] += iC;
                    bS[k] += gb*a[k] - Fb*a[k];
                    mx = dmax(mx, fabs(iC));
                    mn = dmin(mn, iC);
                }
                bA += mx;
                bM += mn;
                // dev2 on the patch face: boundary gradient with its normal part from the patch snGrad
                const v3 Sf = ld3(v.sSf, s);
                const v3 nh = Sf*(1.0/v.sMagSf[s]);
                const double nn[3] = {nh.x, nh.y, nh.z}, sv[3] = {Sf.x, Sf.y, Sf.z};
                const double up[3] = {UP.x, UP.y, UP.z};
                double Gb[9];
                for (int k = 0; k < 3; ++k)
                {
                    const double sn = v.sDc[s]*((a[k] + b[k]*up[k]) - up[k]);
                    const double ng = nn[0]*GP[k] + nn[1]*GP[3 + k] + nn[2]*GP[6 + k];
                    for (int d = 0; d < 3; ++d) Gb[3*d + k] = GP[3*d + k] + nn[d]*(sn - ng);
                }
                const double tr = Gb[0] + Gb[4] + Gb[8];
                for (int k = 0; k < 3; ++k)
                    S[k] += muB*(sv[0]*Gb[3*k] + sv[1]*Gb[3*k + 1] + sv[2]*Gb[3*k + 2] - (2.0/3.0)*tr*sv[k]);
            }
        }
        if (bnd) D -= divPhi;
        const double V = g.V[c];
        if (rdt > 0)
        {
            D += v.rho[i]*V*rdt;
            const v3 u0 = ld3(v.U0, i);
            S[0] += v.rho0f[i]*u0.x*V*rdt;
            S[1] += v.rho0f[i]*u0.y*V*rdt;
            S[2] += v.rho0f[i]*u0.z*V*rdt;
        }
        L.diag[i] = D;
        L.bAbs[i] = bA;
        v.bMin[i] = bM;
        for (int k = 0; k < 3; ++k)
        {
            v.src3[3*i + k] = S[k];
            v.bD3[3*i + k] = bD[k];
            v.bS3[3*i + k] = bS[k];
        }
    });
    pa.reset();
    // UEqn.relax() (vector: patch coefficients enter with their largest magnitude and leave with the smallest)
    if (alpha > 0)
    {
        ProfScope pr("relax");
        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double sumOff = 0;
            for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                if (t == 0) sumOff += fabs(L.upper[j]);
                else if (t == 1) sumOff += fabs(L.lower[j]);
            }
            const double D0 = L.diag[i];
            double D = dmax(fabs(D0 + L.bAbs[i]), sumOff)/alpha - v.bMin[i];
            const v3 u = ld3(v.U, i);
            v.src3[3*i] += (D - D0)*u.x;
            v.src3[3*i + 1] += (D - D0)*u.y;
            v.src3[3*i + 2] += (D - D0)*u.z;
            L.diag[i] = D;
        });
    }
    haveUEqn = true;
    for (int k = 0; k < 3; ++k) info[k] = SolveInfo();
    if (!predictor) return;
    // solve(UEqn == reconstruct((-ghf snGrad(rho) - snGrad(p_rgh)) magSf))
    ProfScope pr("predictor");
    {
        // slot values of p_rgh and rho for the gradients
        double* pbp = ptr(wS);
        for_n(nS, [=] HD (long s) { pbp[s] = pSlot(v, s, v.p[v.sCell[s]]); });
        lsGradient(v, v.p, 1, 0, pbp, 1, 0, 0.0, v.gP, 3, 0, 1);
    }
    lsGradient(v, v.rho, 1, 0, v.rhoB, 1, 0, 0.0, v.gRho, 3, 0, 1);
    // reconstruct(...) V into HbyA (free here: pressure() sets it before use)
    double* rc = v.HbyA;
    const double sP = psiSnP, sR = psiSnRho;
    std::unique_ptr<ProfScope> pc(new ProfScope("reconstruct"));
    // face values once per face (owner -> neighbour orientation, the same seen from both cells)
    double* ff = v.pCorr;
    for_n(nf, [=] HD (long j) {
        const long f = v.f0 + j;
        const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
        ff[j] = (-ghF(v, ld3(g.Cf, f))*snGradFace(v, j, o, n, v.rho, v.gRho, sR)
                 - snGradFace(v, j, o, n, v.p, v.gP, sP))*g.magSf[f];
    });
    for_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        v3 sum = V3(0, 0, 0);
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            double Ff;
            v3 S;
            if (t == 2)
            {
                S = ld3(v.sSf, j);
                const double ms = v.sMagSf[j];
                const double rP = v.rho[v.sCell[j]];
                Ff = (-ghF(v, ld3(v.sCf, j))*v.sDc[j]*(v.rhoB[j] - rP) - pSnGradSlot(v, j, v.p[v.sCell[j]]))*ms;
            }
            else
            {
                S = ld3(g.Sf, v.f0 + j);
                Ff = ff[j];
            }
            sum += S*(Ff/mag(S));
        }
        const v3 r = symmDot(v.Rinv, i, sum);
        const double V = g.V[c];
        rc[3*i] = r.x*V;
        rc[3*i + 1] = r.y*V;
        rc[3*i + 2] = r.z*V;
    });
    pc.reset();
    // component field into rAU (free here as HbyA)
    double* xp = v.rAU;
    double* bd = ptr(MU.bDiag);
    double* bs = ptr(MU.bSrc);
    double* sr = ptr(MU.src);
    for (int k = 0; k < 3; ++k)
    {
        if (!solveDir[k]) continue;
        for_n(nc, [=] HD (long i) {
            xp[i] = v.U[3*i + k];
            sr[i] = v.src3[3*i + k] + rc[3*i + k];
            bd[i] = v.bD3[3*i + k];
            bs[i] = v.bS3[3*i + k];
        });
        info[k] = solve(*m, MU, xp, ctl);
        for_n(nc, [=] HD (long i) { v.U[3*i + k] = xp[i]; });
    }
    ProfScope pb("updateBoundary");
    updateBoundary();
}

// ============================================================================================ pEqn.H
void Flow::pressure(double dt, const SolverCtl& ctl, double pRelax, int nNonOrth, SolveInfo& info, double contErr[2],
                    bool sameRho)
{
    ProfScope prof("pressure");
    FW v = fw(*this);
    const MeshView g = v.g;
    std::unique_ptr<ProfScope> ps(new ProfScope("HbyA, phiHbyA"));
    updateRho();                                      // rho = thermo.rho()
    LduView LU = view(MU);
    // rAU = 1/A, HbyA = rAU H (A and H with the component-averaged patch diagonal)
    for_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        const double avg = (v.bD3[3*i] + v.bD3[3*i + 1] + v.bD3[3*i + 2])/3.0;
        const double A = LU.diag[i] + avg;
        double H[3];
        for (int k = 0; k < 3; ++k) H[k] = v.src3[3*i + k] + v.bS3[3*i + k] - (v.bD3[3*i + k] - avg)*v.U[3*i + k];
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            if (t == 0) for (int k = 0; k < 3; ++k) H[k] -= LU.upper[j]*v.U[3*long(o) + k];
            else if (t == 1) for (int k = 0; k < 3; ++k) H[k] -= LU.lower[j]*v.U[3*long(o) + k];
        }
        v.rAU[i] = g.V[c]/A;
        for (int k = 0; k < 3; ++k) v.HbyA[3*i + k] = H[k]/A;
    });
    // constrainHbyA: patches whose U is not assignable carry U_b
    for_n(nS, [=] HD (long s) {
        const int P = v.sCell[s];
        for (int k = 0; k < 3; ++k) v.HbyAB[3*s + k] = v.uAssign[s] ? v.HbyA[3*long(P) + k] : v.Ub[3*s + k];
    });
    // grad(rho) for the limited snGrad(rho) of phig (unchanged at the next corrector: same T, so the same rho)
    if (!sameRho) lsGradient(v, v.rho, 1, 0, v.rhoB, 1, 0, 0.0, v.gRho, 3, 0, 1);
    const double rdt = 1.0/dt;
    const double sR = psiSnRho;
    for_n(nf, [=] HD (long j) {
        const long f = v.f0 + j;
        const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
        const double w = g.w[f];
        const v3 S = ld3(g.Sf, f);
        const double rf = w*v.rho[o]*v.rAU[o] + (1 - w)*v.rho[n]*v.rAU[n];
        v.rhorAUf[j] = rf;
        double ph = dot(w*v.rho[o]*ld3(v.HbyA, o) + (1 - w)*v.rho[n]*ld3(v.HbyA, n), S);
        // ddtCorr(rho, U, phi)
        const double pc = v.phi0[j] - dot(w*v.rho0f[o]*ld3(v.U0, o) + (1 - w)*v.rho0f[n]*ld3(v.U0, n), S);
        const double cc = 1.0 - dmin(fabs(pc)/(fabs(v.phi0[j]) + SMALL), 1.0);
        ph += rf*cc*pc*rdt;
        const double pg = -rf*ghF(v, ld3(g.Cf, f))*snGradFace(v, j, o, n, v.rho, v.gRho, sR)*g.magSf[f];
        v.phig[j] = pg;
        v.phiHbyA[j] = ph + pg;
    });
    for_n(nS, [=] HD (long s) {
        const int P = v.sCell[s];
        const v3 S = ld3(v.sSf, s);
        const double rf = v.rhoB[s]*v.rAU[P];
        v.rhorAUfB[s] = rf;
        double ph = v.rhoB[s]*dot(ld3(v.HbyAB, s), S);
        if (!v.uFixes[s])
        {
            const double pc = v.phiB0[s] - v.rhoB0[s]*dot(ld3(v.Ub0, s), S);
            const double cc = 1.0 - dmin(fabs(pc)/(fabs(v.phiB0[s]) + SMALL), 1.0);
            ph += rf*cc*pc*rdt;
        }
        const double pg = -rf*ghF(v, ld3(v.sCf, s))*v.sDc[s]*(v.rhoB[s] - v.rho[P])*v.sMagSf[s];
        v.phigB[s] = pg;
        v.phiHbyAB[s] = ph + pg;
        // constrainPressure (fixedFluxPressure)
        if (v.pKind[s] == P_FLUX)
            v.pGrad[s] = (v.phiHbyAB[s] - v.rhoB[s]*dot(ld3(v.Ub, s), S))/(v.sMagSf[s]*rf);
    });
    ps.reset();
    // p_rgh equation
    LduView L = view(Mp);
    const double psi = psiP;
    const int ref = pRefCell;
    const double refV = pRefValue;
    double* pb = ptr(wS);
    for (int corr = 0; corr <= nNonOrth; ++corr)
    {
        for_n(nS, [=] HD (long s) { pb[s] = pSlot(v, s, v.p[v.sCell[s]]); });
        lsGradient(v, v.p, 1, 0, pb, 1, 0, 0.0, v.gP, 3, 0, 1);
        ps.reset(new ProfScope("assemble"));
        for_n(nf, [=] HD (long j) {
            const long f = v.f0 + j;
            const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
            const double G = v.rhorAUf[j]*g.magSf[f]*g.nonOrthDC[f];
            L.upper[j] = -G;
            L.lower[j] = -G;
            double c = 0;
            if (psi > 0)
            {
                const double w = g.w[f];
                const v3 gf = w*ld3(v.gP, o) + (1 - w)*ld3(v.gP, n);
                c = dot(ld3(g.corrVec, f), gf);
                if (psi < 1)
                {
                    const double sn = g.nonOrthDC[f]*(v.p[n] - v.p[o]);
                    c *= dmin(psi*fabs(sn)/((1 - psi)*fabs(c) + SMALL), 1.0);
                }
                c *= v.rhorAUf[j]*g.magSf[f];
            }
            v.pCorr[j] = c;
        });
        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double D = 0, S = 0, bD = 0, bS = 0;
            for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                if (t == 0)
                {
                    D -= L.upper[j];
                    S += v.pCorr[j] - v.phiHbyA[j];
                }
                else if (t == 1)
                {
                    D -= L.lower[j];
                    S += -v.pCorr[j] + v.phiHbyA[j];
                }
                else
                {
                    const long s = j;
                    S -= v.phiHbyAB[s];
                    const double gb = v.rhorAUfB[s]*v.sMagSf[s];
                    if (isPFixed(v.pKind[s]))
                    {
                        bD += gb*v.sDc[s];
                        bS += gb*v.sDc[s]*v.pVal[s];
                    }
                    else if (v.pKind[s] == P_FLUX)
                    {
                        bS += gb*v.pGrad[s];
                    }
                }
            }
            S -= (v.rho[i] - v.rho0f[i])*g.V[c]*rdt;    // fvc::ddt(rho)
            if (i == ref)
            {
                S += D*refV;
                D += D;
            }
            L.diag[i] = D;
            L.src[i] = S;
            L.bDiag[i] = bD;
            L.bAbs[i] = fabs(bD);
            L.bSrc[i] = bS;
        });
        ps.reset();
        info = solve(*m, Mp, v.p, ctl, &amgP);
    }
    // phi = phiHbyA + p_rghEqn.flux()
    ProfScope pu("flux, U, continuity");
    for_n(nf, [=] HD (long j) {
        const long f = v.f0 + j;
        const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
        v.phi[j] = v.phiHbyA[j] + L.upper[j]*(v.p[n] - v.p[o]) - v.pCorr[j];
    });
    for_n(nS, [=] HD (long s) {
        const double gb = v.rhorAUfB[s]*v.sMagSf[s];
        const double pP = v.p[v.sCell[s]];
        double fl = 0;
        if (isPFixed(v.pKind[s])) fl = -gb*v.sDc[s]*(v.pVal[s] - pP);
        else if (v.pKind[s] == P_FLUX) fl = -gb*v.pGrad[s];
        v.phiB[s] = v.phiHbyAB[s] + fl;
    });
    // p_rgh.relax() (towards prevIter, stored at the start of the time step)
    if (pRelax > 0)
        for_n(nc, [=] HD (long i) { v.p[i] = v.pPrev[i] + pRelax*(v.p[i] - v.pPrev[i]); });
    // U = HbyA + rAU reconstruct((phig + p_rghEqn.flux())/rhorAUf)  [flux() after p_rgh.relax()]; the face values
    // once per face into phiHbyA (no longer needed)
    double* ff = v.phiHbyA;
    for_n(nf, [=] HD (long j) {
        const long f = v.f0 + j;
        const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
        const double fl = L.upper[j]*(v.p[n] - v.p[o]) - v.pCorr[j];
        ff[j] = (v.phig[j] + fl)/v.rhorAUf[j];
    });
    for_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        v3 sum = V3(0, 0, 0);
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            v3 S;
            double Ff;
            if (t == 2)
            {
                S = ld3(v.sSf, j);
                const double gb = v.rhorAUfB[j]*v.sMagSf[j];
                const double pP = v.p[v.sCell[j]];
                double fl = 0;
                if (isPFixed(v.pKind[j])) fl = -gb*v.sDc[j]*(v.pVal[j] - pP);
                else if (v.pKind[j] == P_FLUX) fl = -gb*v.pGrad[j];
                Ff = (v.phigB[j] + fl)/v.rhorAUfB[j];
            }
            else
            {
                S = ld3(g.Sf, v.f0 + j);
                Ff = ff[j];
            }
            sum += S*(Ff/mag(S));
        }
        const v3 r = symmDot(v.Rinv, i, sum);
        v.U[3*i] = v.HbyA[3*i] + v.rAU[i]*r.x;
        v.U[3*i + 1] = v.HbyA[3*i + 1] + v.rAU[i]*r.y;
        v.U[3*i + 2] = v.HbyA[3*i + 2] + v.rAU[i]*r.z;
    });
    updateBoundary();
    updatePAbs();
    // rhoEqn + compressibleContinuityErrors: rho from continuity against the thermodynamic rho
    {
        double tot = 0, sl = 0, gl = 0;
        double* r = v.rAU;                            // free after the U reconstruction above

        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double d = 0;
            for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                d += t == 0 ? v.phi[j] : (t == 1 ? -v.phi[j] : v.phiB[j]);
            }
            r[i] = v.rho0f[i] - dt*d/g.V[c];
        });
        tot = sum_n(nc, [=] HD (long i) { return r[i]*g.V[v.c0 + i]; });
        sl = sum_n(nc, [=] HD (long i) { return fabs(r[i] - v.rho[i])*g.V[v.c0 + i]; });
        gl = sum_n(nc, [=] HD (long i) { return (r[i] - v.rho[i])*g.V[v.c0 + i]; });
        contErr[0] = sl/tot;
        contErr[1] = gl/tot;
    }
    // dpdt = fvc::ddt(p); p.oldTime() appears at the first call after a start (copy of the current p)
    if (!pOldReady)
    {
        copy(pAbs, pOld);
        pOldReady = true;
    }
    for_n(nc, [=] HD (long i) { v.dpdt[i] = (v.pAbs[i] - v.pOld[i])*rdt; });
}

void Flow::updatePAbs()
{
    FW v = fw(*this);
    for_n(nc, [=] HD (long i) { v.pAbs[i] = v.p[i] + v.rho[i]*ghC(v, i); });
    for_n(nS, [=] HD (long s) {
        v.pAbsB[s] = pSlot(v, s, v.p[v.sCell[s]]) + v.rhoB[s]*ghF(v, ld3(v.sCf, s));
    });
    if (pRefCell >= 0)
    {
        // closed volume, psi = 0: p += pRefValue - p[pRefCell] (p_rgh unchanged)
        double pr;
        thrust::copy(pAbs.begin() + pRefCell, pAbs.begin() + pRefCell + 1, &pr);
        const double sh = pRefValue - pr;
        for_n(nc, [=] HD (long i) { v.pAbs[i] += sh; });
        for_n(nS, [=] HD (long s) { v.pAbsB[s] += sh; });
    }
}

// ============================================================================================ EEqn.H terms
void Flow::toEnergy(double dt, bool dpdtOn)
{
    FW v = fw(*this);
    const MeshView g = v.g;
    if (!kOldReady)
    {
        // K.oldTime() is created at its first use (the first EEqn after a start): a copy of the current K
        for_n(nc, [=] HD (long i) { const v3 u = ld3(v.U, i); v.K0[i] = 0.5*dot(u, u); });
        kOldReady = true;
    }
    double* Erho = ptr(E->rho);
    double* ErhoO = ptr(E->rhoOld);
    double* Ephi = ptr(E->phi);
    double* Sx = ptr(E->Sx);
    const double rdt = dt > 0 ? 1.0/dt : 0.0;
    const bool dp = dpdtOn;
    for_n(nf, [=] HD (long j) { Ephi[v.f0 + j] = v.phi[j]; });
    for_n(nS, [=] HD (long s) {
        const int f = v.sFace[s];
        if (f >= g.nI) Ephi[f] = v.phiB[s];
    });
    for_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        Erho[c] = v.rho[i];
        ErhoO[c] = v.rho0f[i];
        const v3 u = ld3(v.U, i);
        const double K = 0.5*dot(u, u);
        double divK = 0;
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            if (t == 2)
            {
                const v3 ub = ld3(v.Ub, j);
                divK += v.phiB[j]*0.5*dot(ub, ub);
            }
            else
            {
                const long f = v.f0 + j;
                const double w = g.w[f];
                const v3 uo = ld3(v.U, t == 0 ? i : o), un = ld3(v.U, t == 0 ? o : i);
                const double Kf = w*0.5*dot(uo, uo) + (1 - w)*0.5*dot(un, un);
                divK += (t == 0 ? 1.0 : -1.0)*v.phi[j]*Kf;
            }
        }
        const double V = g.V[c];
        double s = v.rho[i]*(u.x*v.gx + u.y*v.gy + u.z*v.gz)*V - divK;
        if (rdt > 0)
        {
            s -= (v.rho[i]*K - v.rho0f[i]*v.K0[i])*V*rdt;
            if (dp) s += v.dpdt[i]*V;
        }
        Sx[c] = s;
    });
    if (turb.enabled)
    {
        double* Eat = ptr(E->alphat);
        double* EatB = ptr(E->alphatB);
        double* Ekw = ptr(E->kappaWall);
        const double *at = ptr(alphat), *atB = ptr(alphatB), *cpb = ptr(cpB), *kapb = ptr(kappaB);
        for_n(nc, [=] HD (long i) { Eat[v.c0 + i] = at[i]; });
        for_n(nS, [=] HD (long s) {
            const int f = v.sFace[s];
            if (f >= g.nI) EatB[f - g.nI] = atB[s];
            else Ekw[f - v.nIrG] = kapb[s] + cpb[s]*atB[s];
        });
    }
}

// ============================================================================================ RNGkEpsilon::correct()
void Flow::turbulence(double dt, double alphaK, double alphaEps, const SolverCtl& ctlK, const SolverCtl& ctlEps,
                      SolveInfo& infoEps, SolveInfo& infoK)
{
    ProfScope prof("turbulence");
    infoEps = SolveInfo();
    infoK = SolveInfo();
    if (!turb.enabled) return;
    updateBoundary();
    FW v = fw(*this);
    const MeshView g = v.g;
    double *kp = ptr(k), *ep = ptr(eps);
    const double *kp0 = ptr(k0), *ep0 = ptr(eps0);
    double *kbp = ptr(kb), *ebp = ptr(epsb);
    const int *kK = ptr(kKind), *eK = ptr(epsKind);
    const double *kV = ptr(kVal), *eV = ptr(epsVal);
    lsGradient3(v, v.U, v.Ub, limitKU, v.gU);
    double *gbynu = ptr(GbyNu), *gturb = ptr(Gturb), *divu = ptr(divU);
    std::unique_ptr<ProfScope> pg(new ProfScope("G, divU, wall"));
    for_n(nc, [=] HD (long i) {
        const double* G = v.gU + 9*i;
        const double tr2 = (2.0/3.0)*(G[0] + G[4] + G[8]);
        double gbn = 0.0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
            {
                const double dev = G[3*a + b] + G[3*b + a] - (a == b ? tr2 : 0.0);
                gbn += G[3*a + b]*dev;
            }
        gbynu[i] = gbn;
        gturb[i] = v.nut[i]*gbn;
        const long c = v.c0 + i;
        double dv = 0.0;
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            if (t == 2)
            {
                dv += v.phiB[j]/v.rhoB[j];
            }
            else
            {
                const long f = v.f0 + j;
                const double w = g.w[f];
                const int oo = t == 0 ? int(i) : o, nn = t == 0 ? o : int(i);
                const double rf = w*v.rho[oo] + (1.0 - w)*v.rho[nn];
                dv += (t == 0 ? 1.0 : -1.0)*v.phi[j]/rf;
            }
        }
        divu[i] = dv/g.V[c];
    });

    // epsilonWallFunction::updateCoeffs: cell-wise cornerWeights sum over wall slots
    const long nWC = long(wallCells.size());
    if (nWC > 0)
    {
        const int* wc = ptr(wallCells);
        double* ew = ptr(epsWall);
        const char* isw = ptr(isWall);
        const double *cw = ptr(sCw), *ecmu = ptr(eCmu), *ekap = ptr(eKappa), *sy = ptr(sY);
        for_n(nWC, [=] HD (long m) {
            const int i = wc[m];
            const long c = v.c0 + i;
            const double kc = dmax(kp[i], 0.0);
            const double skc = sqrt(kc);
            const v3 uP = ld3(v.U, i);
            double e0 = 0.0, g0 = 0.0;
            for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                if (t != 2 || !isw[j]) continue;
                const long s = j;
                const double w = cw[s];
                const double cmu25 = pow(ecmu[s], 0.25);
                const double cmu75 = cmu25*cmu25*cmu25;
                const double ky = ekap[s]*sy[s];
                const double nuw = v.muB[s]/v.rhoB[s];
                const double magGradUw = mag(ld3(v.Ub, s) - uP)*v.sDc[s];
                e0 += w*cmu75*kc*skc/ky;
                g0 += w*(v.nutB[s] + nuw)*magGradUw*cmu25*skc/ky;
            }
            ew[m] = e0;
            gturb[i] = g0;
        });
    }
    pg.reset();

    LduView L = view(Mp);
    double* corr = ptr(pCorr);
    double* gp = ptr(gP);
    double *de = ptr(De), *deb = ptr(DeB);
    const double rdt = dt > 0.0 ? 1.0/dt : 0.0;
    const double Cmu = turb.Cmu, C1 = turb.C1, C2 = turb.C2, C3 = turb.C3;
    const double sigmak = turb.sigmak, sigmaEps = turb.sigmaEps, eta0 = turb.eta0, beta = turb.beta;

    auto assembleAndSolve = [&](double* fld, const double* fld0, double* fldb, const int* kind, const double* val,
                                bool isEps, int cv, double cvK, bool bnd, double limK, double psi, double alpha,
                                const SolverCtl& ctl, SolveInfo& info) {
        for_n(nS, [=] HD (long s) {
            double a, bb;
            tSlotCoeffs(v, s, kind[s], val[s], isEps ? kbp[s] : 0.0, a, bb);
            fldb[s] = a + bb*fld[v.sCell[s]];
        });
        const double invSig = 1.0/(isEps ? sigmaEps : sigmak);
        for_n(nc, [=] HD (long i) { de[i] = v.rho[i]*v.nut[i]*invSig + v.mu[i]; });
        for_n(nS, [=] HD (long s) { deb[s] = v.rhoB[s]*v.nutB[s]*invSig + v.muB[s]; });
        if (psi > 0.0 || cv == 1 || cv == 3) lsGradient(v, fld, 1, 0, fldb, 1, 0, limK, gp, 3, 0, 1);
        std::unique_ptr<ProfScope> pa(new ProfScope("assemble"));
        for_n(nf, [=] HD (long j) {
            const long f = v.f0 + j;
            const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
            const double w = g.w[f];
            const double dc = g.nonOrthDC[f];
            const double gmag = (w*de[o] + (1.0 - w)*de[n])*g.magSf[f]*dc;
            const double F = v.phi[j];
            if (cv <= 1)
            {
                L.upper[j] = -gmag + (F < 0.0 ? F : 0.0);
                L.lower[j] = -gmag - (F > 0.0 ? F : 0.0);
            }
            else
            {
                const double lam = cv == 3 ? limLinearFace(v, j, fld, gp, cvK) : 1.0;
                const double wf = lam*w + (1.0 - lam)*(F >= 0.0 ? 1.0 : 0.0);
                L.upper[j] = -gmag + (1.0 - wf)*F;
                L.lower[j] = -gmag - wf*F;
            }
            double c = 0.0;
            if (psi > 0.0)
            {
                const v3 gf = w*ld3(gp, o) + (1.0 - w)*ld3(gp, n);
                c = dot(ld3(g.corrVec, f), gf);
                if (psi < 1.0)
                {
                    const double sn = dc*(fld[n] - fld[o]);
                    c *= dmin(psi*fabs(sn)/((1.0 - psi)*fabs(c) + SMALL), 1.0);
                }
                c *= (gmag/dc);
            }
            if (cv == 1 && F != 0.0)
            {
                const int up = F >= 0.0 ? o : n;
                const v3 d = ld3(g.Cf, f) - ld3(g.C, v.c0 + up);
                c -= F*dot(d, ld3(gp, up));
            }
            corr[j] = c;
        });
        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double D = 0.0, S = 0.0, bD = 0.0, bA = 0.0, bS = 0.0, divPhi = 0.0;
            for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                if (t == 0)
                {
                    D -= L.lower[j];
                    S += corr[j];
                    divPhi += v.phi[j];
                }
                else if (t == 1)
                {
                    D -= L.upper[j];
                    S -= corr[j];
                    divPhi -= v.phi[j];
                }
                else
                {
                    const long s = j;
                    const double gb = deb[s]*v.sMagSf[s]*v.sDc[s];
                    const double Fb = v.phiB[s];
                    divPhi += Fb;
                    double a, bb;
                    tSlotCoeffs(v, s, kind[s], val[s], isEps ? kbp[s] : 0.0, a, bb);
                    const double iC = gb*(1.0 - bb) + Fb*bb;
                    const double bC = (gb - Fb)*a;
                    bD += iC;
                    bA += fabs(iC);
                    bS += bC;
                }
            }
            if (bnd) D -= divPhi;
            const double V = g.V[c];
            if (rdt > 0.0)
            {
                D += v.rho[i]*V*rdt;
                S += v.rho0f[i]*fld0[i]*V*rdt;
            }
            double Su, Sp;
            if (isEps)
            {
                const double eta = sqrt(fabs(gbynu[i]))*kp[i]/ep[i];
                const double R = (eta*(1.0 - eta/eta0))/(beta*eta*eta*eta + 1.0);
                Su = (C1 - R)*v.rho[i]*gbynu[i]*Cmu*kp[i]*V;
                Sp = -C2*v.rho[i]*ep[i]/kp[i]*V;
                const double cd = ((2.0/3.0)*C1 - C3)*v.rho[i]*divu[i]*V;
                Sp -= dmax(cd, 0.0);
                Su -= dmin(cd, 0.0)*ep[i];
            }
            else
            {
                Su = v.rho[i]*gturb[i]*V;
                Sp = -v.rho[i]*ep[i]/kp[i]*V;
                const double ck = (2.0/3.0)*v.rho[i]*divu[i]*V;
                Sp -= dmax(ck, 0.0);
                Su -= dmin(ck, 0.0)*kp[i];
            }
            S += Su;
            D -= Sp;
            L.diag[i] = D;
            L.src[i] = S;
            L.bDiag[i] = bD;
            L.bAbs[i] = bA;
            L.bSrc[i] = bS;
        });
        if (alpha > 0.0)
        {
            for_n(nc, [=] HD (long i) {
                const long c = v.c0 + i;
                double sumOff = 0.0;
                for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
                {
                    int j, o;
                    const int t = ent(v, q, j, o);
                    if (t == 0) sumOff += fabs(L.upper[j]);
                    else if (t == 1) sumOff += fabs(L.lower[j]);
                }
                const double D0 = L.diag[i];
                const double D = dmax(fabs(D0 + L.bAbs[i]), sumOff)/alpha - L.bDiag[i];
                L.src[i] += (D - D0)*fld[i];
                L.diag[i] = D;
            });
        }
        if (isEps && !wallCellsGlobal.empty())
            setValues(*this->m, Mp, wallCellsGlobal, epsWall, fld);
        pa.reset();
        info = solve(*this->m, Mp, fld, ctl);
        ProfScope pb("bound");
        const double vmin = isEps ? turb.epsMin : turb.kMin;
        const double mn = min_n(nc, [=] HD (long i) { return fld[i]; });
        if (mn < vmin)
        {
            for_n(nS, [=] HD (long s) {
                double a, bb;
                tSlotCoeffs(v, s, kind[s], val[s], isEps ? kbp[s] : 0.0, a, bb);
                fldb[s] = a + bb*fld[v.sCell[s]];
            });
            double* out = ptr(rAU);
            for_n(nc, [=] HD (long i) {
                if (fld[i] >= vmin) { out[i] = fld[i]; return; }
                const long c = v.c0 + i;
                double sv = 0.0, sa = 0.0;
                for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
                {
                    int j, o;
                    const int t = ent(v, q, j, o);
                    if (t == 2)
                    {
                        const double ms = v.sMagSf[j];
                        sv += dmax(fldb[j], vmin)*ms;
                        sa += ms;
                    }
                    else
                    {
                        const long f = v.f0 + j;
                        const int oo = t == 0 ? int(i) : o, nn = t == 0 ? o : int(i);
                        const double w = g.w[f];
                        const double ms = g.magSf[f];
                        const double vf = w*dmax(fld[oo], vmin) + (1.0 - w)*dmax(fld[nn], vmin);
                        sv += vf*ms;
                        sa += ms;
                    }
                }
                out[i] = dmax(sv/sa, vmin);
            });
            copy(rAU, isEps ? eps : k);
        }
        for_n(nS, [=] HD (long s) {
            double a, bb;
            tSlotCoeffs(v, s, kind[s], val[s], isEps ? kbp[s] : 0.0, a, bb);
            fldb[s] = a + bb*fld[v.sCell[s]];
        });
    };

    assembleAndSolve(ep, ep0, ebp, eK, eV, true, turb.convEps, turb.convEpsK, turb.boundedEps, turb.limitKEps,
                     turb.psiEps, alphaEps, ctlEps, infoEps);
    assembleAndSolve(kp, kp0, kbp, kK, kV, false, turb.convK, turb.convKK, turb.boundedK, turb.limitKK,
                     turb.psiK, alphaK, ctlK, infoK);
    correctNut();
    updateAlphat();
}

// ============================================================================================ YEqn.H
void Flow::species(double dt, const std::vector<double>& alphaY, const SolverCtl& ctl, std::vector<SolveInfo>& info)
{
    ProfScope prof("species");
    info.assign(nSp, SolveInfo());
    if (nSp <= 0) return;
    FW v = fw(*this);
    const MeshView g = v.g;
    const int ns = nSp, inert = inertIndex, cv = convY, cc0 = c0;
    const double kC = convYK, lKH = limitKH;
    const bool bnd = boundedY;
    double *cp = ptr(E->cp), *ee = ptr(E->e), *tC = ptr(E->T);
    double* he = ptr(rAU);
    for_n(nc, [=] HD (long i) { he[i] = cp[cc0 + i]*tC[cc0 + i] + ee[cc0 + i]; });

    double* yp = ptr(Y);
    const double* yp0 = ptr(Y0);
    double* yb = ptr(Yb);
    const int* yk = ptr(yKind);
    const double *yv = ptr(yVal), *yg = ptr(yGrad), *yf = ptr(yVf);
    for_n(long(ns)*nS, [=] HD (long idx) {
        const int m = int(idx / v.nS);
        const long s = idx - long(m)*v.nS;
        double a, bb;
        ySlotCoeffs(yk[idx], yv[idx], yg[idx], yf[idx], v.sDc[s], v.phiB[s], a, bb);
        yb[idx] = a + bb*yp[long(m)*v.nc + v.sCell[s]];
    });

    // multivariate / univariate face interpolation weights wY
    double* wy = ptr(phig);
    double* gp = ptr(gP);
    if (cv == 0)
    {
        for_n(nf, [=] HD (long j) { wy[j] = v.phi[j] >= 0.0 ? 1.0 : 0.0; });
    }
    else if (cv == 2)
    {
        for_n(nf, [=] HD (long j) { wy[j] = g.w[v.f0 + j]; });
    }
    else if (cv == 3)
    {
        fill(phig, 1.0);
        for (int m = 0; m < ns; ++m)
        {
            const double* ym = yp + long(m)*nc;
            const double* ybm = yb + long(m)*nS;
            lsGradient(v, ym, 1, 0, ybm, 1, 0, sp[m].limitK, gp, 3, 0, 1);
            for_n(nf, [=] HD (long j) { wy[j] = dmin(wy[j], limLinearFace(v, j, ym, gp, kC)); });
        }
        double* heb = ptr(rhorAUfB);
        const double *cpb = ptr(cpB), *eb = ptr(eB);
        for_n(nS, [=] HD (long s) { heb[s] = cpb[s]*Tslot(v, s) + eb[s]; });
        double* gh = ptr(gRho);
        lsGradient(v, he, 1, 0, heb, 1, 0, lKH, gh, 3, 0, 1);
        for_n(nf, [=] HD (long j) {
            const double lam = dmin(wy[j], limLinearFace(v, j, he, gh, kC));
            wy[j] = lam*g.w[v.f0 + j] + (1.0 - lam)*(v.phi[j] >= 0.0 ? 1.0 : 0.0);
        });
    }

    LduView L = view(Mp);
    double* corr = ptr(pCorr);
    const double rdt = dt > 0.0 ? 1.0/dt : 0.0;
    for (int m = 0; m < ns; ++m)
    {
        if (m == inert) continue;
        double* ym = yp + long(m)*nc;
        const double* ym0 = yp0 + long(m)*nc;
        double* ybm = yb + long(m)*nS;
        const double* su = ptr(SuY) + long(m)*nc;
        const double* spv = ptr(SpY) + long(m)*nc;
        const double psi = sp[m].psiY;
        const double limK = sp[m].limitK;
        const double alpha = m < int(alphaY.size()) ? alphaY[m] : 0.0;
        if (psi > 0.0 || cv == 1) lsGradient(v, ym, 1, 0, ybm, 1, 0, limK, gp, 3, 0, 1);
        for_n(nf, [=] HD (long j) {
            const long f = v.f0 + j;
            const int o = g.own[f] - v.c0, n = g.nei[f] - v.c0;
            const double w = g.w[f];
            const double dc = g.nonOrthDC[f];
            const double gmag = (w*muEffC(v, o) + (1.0 - w)*muEffC(v, n))*g.magSf[f]*dc;
            const double F = v.phi[j];
            if (cv == 1)
            {
                L.upper[j] = -gmag + (F < 0.0 ? F : 0.0);
                L.lower[j] = -gmag - (F > 0.0 ? F : 0.0);
            }
            else
            {
                const double wf = wy[j];
                L.upper[j] = -gmag + (1.0 - wf)*F;
                L.lower[j] = -gmag - wf*F;
            }
            double c = 0.0;
            if (psi > 0.0)
            {
                const v3 gf = w*ld3(gp, o) + (1.0 - w)*ld3(gp, n);
                c = dot(ld3(g.corrVec, f), gf);
                if (psi < 1.0)
                {
                    const double sn = dc*(ym[n] - ym[o]);
                    c *= dmin(psi*fabs(sn)/((1.0 - psi)*fabs(c) + SMALL), 1.0);
                }
                c *= (gmag/dc);
            }
            if (cv == 1 && F != 0.0)
            {
                const int up = F >= 0.0 ? o : n;
                const v3 d = ld3(g.Cf, f) - ld3(g.C, v.c0 + up);
                c -= F*dot(d, ld3(gp, up));
            }
            corr[j] = c;
        });
        for_n(nc, [=] HD (long i) {
            const long c = v.c0 + i;
            double D = 0.0, S = 0.0, bD = 0.0, bA = 0.0, bS = 0.0, divPhi = 0.0;
            for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
            {
                int j, o;
                const int t = ent(v, q, j, o);
                if (t == 0)
                {
                    D -= L.lower[j];
                    S += corr[j];
                    divPhi += v.phi[j];
                }
                else if (t == 1)
                {
                    D -= L.upper[j];
                    S -= corr[j];
                    divPhi -= v.phi[j];
                }
                else
                {
                    const long s = j;
                    const double muB = v.muB[s] + v.rhoB[s]*v.nutB[s];
                    const double gb = muB*v.sMagSf[s]*v.sDc[s];
                    const double Fb = v.phiB[s];
                    divPhi += Fb;
                    double a, bb;
                    const long idx = long(m)*v.nS + s;
                    ySlotCoeffs(yk[idx], yv[idx], yg[idx], yf[idx], v.sDc[s], Fb, a, bb);
                    const double iC = gb*(1.0 - bb) + Fb*bb;
                    const double bC = (gb - Fb)*a;
                    bD += iC;
                    bA += fabs(iC);
                    bS += bC;
                }
            }
            if (bnd) D -= divPhi;
            const double V = g.V[c];
            if (rdt > 0.0)
            {
                D += v.rho[i]*V*rdt;
                S += v.rho0f[i]*ym0[i]*V*rdt;
            }
            S += su[i];
            D -= spv[i];
            L.diag[i] = D;
            L.src[i] = S;
            L.bDiag[i] = bD;
            L.bAbs[i] = bA;
            L.bSrc[i] = bS;
        });
        if (alpha > 0.0)
        {
            for_n(nc, [=] HD (long i) {
                const long c = v.c0 + i;
                double sumOff = 0.0;
                for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
                {
                    int j, o;
                    const int t = ent(v, q, j, o);
                    if (t == 0) sumOff += fabs(L.upper[j]);
                    else if (t == 1) sumOff += fabs(L.lower[j]);
                }
                const double D0 = L.diag[i];
                const double D = dmax(fabs(D0 + L.bAbs[i]), sumOff)/alpha - L.bDiag[i];
                L.src[i] += (D - D0)*ym[i];
                L.diag[i] = D;
            });
        }
        info[m] = solve(*this->m, Mp, ym, ctl);
        for_n(nc, [=] HD (long i) { ym[i] = dmax(ym[i], 0.0); });
        for_n(nS, [=] HD (long s) {
            double a, bb;
            const long idx = long(m)*v.nS + s;
            ySlotCoeffs(yk[idx], yv[idx], yg[idx], yf[idx], v.sDc[s], v.phiB[s], a, bb);
            ybm[s] = dmax(a + bb*ym[v.sCell[s]], 0.0);
        });
    }
    if (inert >= 0 && inert < ns)
    {
        for_n(nc, [=] HD (long i) {
            double yt = 0.0;
            for (int m = 0; m < ns; ++m) if (m != inert) yt += yp[long(m)*v.nc + i];
            yp[long(inert)*v.nc + i] = dmax(1.0 - yt, 0.0);
        });
        for_n(nS, [=] HD (long s) {
            double yt = 0.0;
            for (int m = 0; m < ns; ++m) if (m != inert) yt += yb[long(m)*v.nS + s];
            yb[long(inert)*v.nS + s] = dmax(1.0 - yt, 0.0);
        });
    }
    // thermo.correct(): update mixture properties and recompute T from unchanged he = cp_new*T + e_new
    updateMixture(false);
    for_n(nc, [=] HD (long i) { tC[cc0 + i] = (he[i] - ee[cc0 + i])/cp[cc0 + i]; });
}

double Flow::courantNo(double dt, double& meanCo)
{
    if (nc == 0 || nf == 0) { meanCo = 0.0; return 0.0; }
    FW v = fw(*this);
    const MeshView g = v.g;
    const double maxCo = 0.5*dt*max_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        double sp = 0.0;
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            if (t == 0 || t == 1) sp += fabs(v.phi[j]);
            else if (t == 2) sp += fabs(v.phiB[j]);
        }
        return (sp/v.rho[i])/g.V[c];
    }, 0.0);
    const double sumPhiRho = sum_n(nc, [=] HD (long i) {
        const long c = v.c0 + i;
        double sp = 0.0;
        for (int q = g.cfPtr[c]; q < g.cfPtr[c + 1]; ++q)
        {
            int j, o;
            const int t = ent(v, q, j, o);
            if (t == 0 || t == 1) sp += fabs(v.phi[j]);
            else if (t == 2) sp += fabs(v.phiB[j]);
        }
        return sp/v.rho[i];
    });
    const double vTot = sum_n(nc, [=] HD (long i) { return g.V[v.c0 + i]; });
    meanCo = vTot > 0.0 ? 0.5*dt*(sumPhiRho/vTot) : 0.0;
    return maxCo;
}

} // namespace gc
